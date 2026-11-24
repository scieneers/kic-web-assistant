"""Unit tests for Drupal's optional-vs-required authentication gating.

DRUPAL_AUTH_REQUIRED lets prod deployments fail loudly when credentials are
missing or invalid, instead of silently falling back to public-only content
(which risks stale-deleting protected courses in get_data.py, since the run
would then "successfully" see less than what's actually indexed).

Credential fields are saved/restored manually (via object.__getattribute__)
rather than through monkeypatch.setattr: EnvHelper.__getattribute__ raises
AttributeError for any field still at its "UNSET" sentinel, which would make
monkeypatch treat the field as never having existed and `delattr` it on
teardown — breaking the shared `env` singleton for every test after it.
"""

import pytest

from src.env import env
from src.loaders.drupal import Drupal

_CREDENTIAL_FIELDS = ["DRUPAL_CLIENT_ID", "DRUPAL_CLIENT_SECRET", "DRUPAL_USERNAME", "DRUPAL_PASSWORD"]


@pytest.fixture(autouse=True)
def clean_drupal_env():
    fields = _CREDENTIAL_FIELDS + ["DRUPAL_AUTH_REQUIRED"]
    original = {name: object.__getattribute__(env, name) for name in fields}
    for name in _CREDENTIAL_FIELDS:
        setattr(env, name, "UNSET")
    setattr(env, "DRUPAL_AUTH_REQUIRED", False)
    yield
    for name, value in original.items():
        setattr(env, name, value)


def _set_credentials() -> None:
    for name in _CREDENTIAL_FIELDS:
        setattr(env, name, f"test-{name.lower()}")


class TestDrupalAuthOptional:
    def test_no_credentials_and_not_required_runs_unauthenticated(self):
        drupal = Drupal()
        assert "Authorization" not in drupal.header

    def test_no_credentials_and_required_raises(self):
        setattr(env, "DRUPAL_AUTH_REQUIRED", True)
        with pytest.raises(RuntimeError, match="DRUPAL_AUTH_REQUIRED"):
            Drupal()

    def test_credentials_set_but_oauth_fails_and_required_raises(self, monkeypatch):
        _set_credentials()
        setattr(env, "DRUPAL_AUTH_REQUIRED", True)
        monkeypatch.setattr(Drupal, "get_oauth_token", lambda self, base_url: None)
        with pytest.raises(RuntimeError, match="DRUPAL_AUTH_REQUIRED"):
            Drupal()

    def test_credentials_set_but_oauth_fails_and_not_required_logs_and_continues(self, monkeypatch, caplog):
        _set_credentials()
        monkeypatch.setattr(Drupal, "get_oauth_token", lambda self, base_url: None)
        drupal = Drupal()
        assert "Authorization" not in drupal.header

    def test_credentials_set_and_oauth_succeeds_sets_header_regardless_of_required(self, monkeypatch):
        _set_credentials()
        setattr(env, "DRUPAL_AUTH_REQUIRED", True)
        monkeypatch.setattr(Drupal, "get_oauth_token", lambda self, base_url: "tok123")
        drupal = Drupal()
        assert drupal.header["Authorization"] == "Bearer tok123"
