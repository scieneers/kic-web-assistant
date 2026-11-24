"""Integration tests for the Moodle API client.

All tests call real external services and require valid credentials (az login).
Run with: pytest src/tests/loaders/test_moodle_integration.py -m integration
"""

import os

import pytest
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

from src.loaders.moodle import CourseTopic, MoodleCourse, Moodle


@pytest.fixture
def production_moodle():
    """Moodle instance authenticated with Production credentials from Lab Key Vault."""
    key_vault_name = os.environ.get("KEY_VAULT_NAME", "kicwa-keyvault-lab")
    key_vault_uri = f"https://{key_vault_name}.vault.azure.net/"
    credential = DefaultAzureCredential()
    secret_client = SecretClient(vault_url=key_vault_uri, credential=credential)

    prod_url = secret_client.get_secret("DATA-SOURCE-PRODUCTION-MOODLE-URL").value
    prod_token = secret_client.get_secret("DATA-SOURCE-PRODUCTION-MOODLE-TOKEN").value

    moodle = Moodle()
    moodle.base_url = prod_url
    moodle.api_endpoint = f"{prod_url}webservice/rest/server.php"
    moodle.token = prod_token
    moodle.function_params["wstoken"] = prod_token
    moodle.download_params["token"] = prod_token

    return moodle


@pytest.mark.integration
def test_get_token():
    """Default Moodle() picks up a non-empty API token from the environment."""
    moodle = Moodle()
    assert moodle.token, "Moodle token must not be empty"
    assert len(moodle.token) > 10


@pytest.mark.integration
def test_get_topics():
    """Course contents for course ID 16 parse into valid CourseTopic objects."""
    # https://ki-campus-test.fernuni-hagen.de/course/view.php?id=16
    moodle = Moodle()
    topics = moodle.get_course_contents(16)

    assert isinstance(topics, list)
    assert len(topics) > 0, "Course 16 must have at least one topic"
    assert all(isinstance(t, CourseTopic) for t in topics)
    assert all(t.id is not None for t in topics)
    assert all(isinstance(t.name, str) and len(t.name) > 0 for t in topics)


@pytest.mark.integration
def test_count_courses(production_moodle):
    """Production Moodle returns a non-empty list of well-formed MoodleCourse objects."""
    courses = production_moodle.get_courses()

    assert isinstance(courses, list)
    assert len(courses) > 0, "Production Moodle must have at least one visible course"
    assert all(isinstance(c, MoodleCourse) for c in courses)
    sample = courses[:10]
    assert all(c.id is not None for c in sample)
    assert all(c.fullname for c in sample)
    assert all(c.shortname for c in sample)
