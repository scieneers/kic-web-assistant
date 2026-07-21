"""Unit tests for the Drupal loader's HTTP robustness layer.

Background: ingest runs died on a single ConnectTimeout to ki-campus.org
(requests.get without timeout/retries). _get now wraps every request with
timeouts, session-level retries and an outer exponential-backoff loop, and
fetch_data degrades to {} for related-entity lookups instead of failing the
whole run.
"""

from unittest.mock import MagicMock

import pytest
import requests

import src.loaders.drupal as drupal_module
from src.loaders.drupal import Drupal


@pytest.fixture()
def drupal(monkeypatch) -> Drupal:
    monkeypatch.setattr(Drupal, "get_oauth_token", lambda self, base_url: None)
    monkeypatch.setattr(drupal_module.env, "DRUPAL_AUTH_REQUIRED", False, raising=False)
    monkeypatch.setattr(drupal_module, "HTTP_REQUEST_DELAY_SECONDS", 0)
    instance = Drupal()
    return instance


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(drupal_module.time, "sleep", sleeps.append)
    return sleeps


def _ok_response(payload: dict | None = None) -> MagicMock:
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = payload if payload is not None else {"data": []}
    return response


class TestGetRetry:
    def test_returns_response_on_first_success(self, drupal, monkeypatch):
        response = _ok_response()
        monkeypatch.setattr(drupal.session, "get", MagicMock(return_value=response))
        assert drupal._get("https://ki-campus.org/jsonapi/node/course") is response

    def test_retries_on_connect_timeout_then_succeeds(self, drupal, monkeypatch, no_sleep):
        response = _ok_response()
        mock_get = MagicMock(side_effect=[requests.exceptions.ConnectTimeout("boom"), response])
        monkeypatch.setattr(drupal.session, "get", mock_get)

        assert drupal._get("https://ki-campus.org/jsonapi/node/course") is response
        assert mock_get.call_count == 2
        assert len(no_sleep) == 1

    def test_backoff_grows_exponentially(self, drupal, monkeypatch, no_sleep):
        response = _ok_response()
        mock_get = MagicMock(
            side_effect=[
                requests.exceptions.ConnectTimeout("boom"),
                requests.exceptions.ReadTimeout("boom"),
                requests.exceptions.ConnectionError("boom"),
                response,
            ]
        )
        monkeypatch.setattr(drupal.session, "get", mock_get)

        drupal._get("https://ki-campus.org/whatever")
        assert no_sleep == [
            drupal_module.HTTP_BACKOFF_SECONDS,
            drupal_module.HTTP_BACKOFF_SECONDS * 2,
            drupal_module.HTTP_BACKOFF_SECONDS * 4,
        ]

    def test_raises_after_max_attempts(self, drupal, monkeypatch):
        mock_get = MagicMock(side_effect=requests.exceptions.ConnectTimeout("boom"))
        monkeypatch.setattr(drupal.session, "get", mock_get)

        with pytest.raises(requests.exceptions.ConnectTimeout):
            drupal._get("https://ki-campus.org/whatever")
        assert mock_get.call_count == drupal_module.HTTP_MAX_ATTEMPTS

    def test_sends_timeout_and_headers(self, drupal, monkeypatch):
        mock_get = MagicMock(return_value=_ok_response())
        monkeypatch.setattr(drupal.session, "get", mock_get)

        drupal._get("https://ki-campus.org/whatever")
        _, kwargs = mock_get.call_args
        assert kwargs["timeout"] == (drupal_module.HTTP_CONNECT_TIMEOUT, drupal_module.HTTP_READ_TIMEOUT)
        assert kwargs["headers"] is drupal.header


class TestFetchDataDegradesGracefully:
    def test_returns_empty_dict_when_all_retries_exhausted(self, drupal, monkeypatch):
        monkeypatch.setattr(
            drupal.session, "get", MagicMock(side_effect=requests.exceptions.ConnectTimeout("boom"))
        )
        assert drupal.fetch_data("https://ki-campus.org/jsonapi/node/lecturer/abc") == {}

    def test_returns_empty_dict_on_non_200(self, drupal, monkeypatch):
        response = MagicMock()
        response.status_code = 404
        response.text = "not found"
        monkeypatch.setattr(drupal.session, "get", MagicMock(return_value=response))
        assert drupal.fetch_data("https://ki-campus.org/jsonapi/node/lecturer/abc") == {}

    def test_lecturer_lookup_failure_skips_instead_of_crashing(self, drupal, monkeypatch):
        monkeypatch.setattr(
            drupal.session, "get", MagicMock(side_effect=requests.exceptions.ConnectTimeout("boom"))
        )
        names = drupal.get_lecturers([{"id": "160e1d6d-667e-45ca-a1a3-55bc98268de4"}])
        assert names == []

    def test_lecture_book_chapters_tolerate_empty_fetch(self, drupal, monkeypatch):
        monkeypatch.setattr(drupal, "fetch_data", lambda url: {})
        assert drupal.process_chapters({}) == ""
        assert drupal.process_lectures({}) == ""
