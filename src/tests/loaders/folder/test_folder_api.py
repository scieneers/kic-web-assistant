"""Integration tests for Moodle folder module API."""

import os

import pytest

from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

from src.loaders.APICaller import APICaller
from src.loaders.moodle import Moodle

pytestmark = pytest.mark.integration

TEST_COURSE_ID = 134  # "Die Welt der KI entdecken"


@pytest.fixture(scope="module")
def moodle():
    key_vault_name = os.environ.get("KEY_VAULT_NAME", "kicwa-keyvault-lab")
    key_vault_uri = f"https://{key_vault_name}.vault.azure.net/"
    credential = DefaultAzureCredential()
    secret_client = SecretClient(vault_url=key_vault_uri, credential=credential)

    prod_url = secret_client.get_secret("DATA-SOURCE-PRODUCTION-MOODLE-URL").value
    prod_token = secret_client.get_secret("DATA-SOURCE-PRODUCTION-MOODLE-TOKEN").value

    m = Moodle()
    m.base_url = prod_url
    m.api_endpoint = f"{prod_url}webservice/rest/server.php"
    m.token = prod_token
    m.function_params["wstoken"] = prod_token
    m.download_params["token"] = prod_token
    return m


def test_mod_folder_api_returns_dict(moodle):
    caller = APICaller(
        url=moodle.api_endpoint,
        params={**moodle.function_params, "courseids[0]": TEST_COURSE_ID},
        wsfunction="mod_folder_get_folders_by_courses",
    )
    response = caller.getJSON()
    assert isinstance(response, dict), "Response should be a dict"
    assert "exception" not in response, f"API returned exception: {response.get('message')}"


def test_mod_folder_api_has_folders_key(moodle):
    caller = APICaller(
        url=moodle.api_endpoint,
        params={**moodle.function_params, "courseids[0]": TEST_COURSE_ID},
        wsfunction="mod_folder_get_folders_by_courses",
    )
    response = caller.getJSON()
    assert "folders" in response, "Response should contain 'folders' key"


def test_folder_modules_have_required_fields(moodle):
    caller = APICaller(
        url=moodle.api_endpoint,
        params={**moodle.function_params, "courseids[0]": TEST_COURSE_ID},
        wsfunction="mod_folder_get_folders_by_courses",
    )
    response = caller.getJSON()
    folders = response.get("folders", [])
    if not folders:
        pytest.skip(f"No folder modules found in course {TEST_COURSE_ID}")
    for folder in folders:
        assert "id" in folder
        assert "name" in folder
        assert "coursemodule" in folder
        assert "course" in folder


def test_core_course_contents_has_folder_modules(moodle):
    caller = APICaller(
        url=moodle.api_endpoint,
        params=moodle.function_params,
        wsfunction="core_course_get_contents",
        courseid=TEST_COURSE_ID,
    )
    topics = caller.getJSON()
    assert isinstance(topics, list), "core_course_get_contents should return a list"

    folder_modules = [
        module
        for topic in topics
        for module in topic.get("modules", [])
        if module.get("modname") == "folder"
    ]
    assert len(folder_modules) > 0, f"Expected folder modules in course {TEST_COURSE_ID}"


def test_folder_files_accessible_via_core_api(moodle):
    """Folders expose their files through core_course_get_contents contents array."""
    caller = APICaller(
        url=moodle.api_endpoint,
        params=moodle.function_params,
        wsfunction="core_course_get_contents",
        courseid=TEST_COURSE_ID,
    )
    topics = caller.getJSON()
    folder_modules = [
        module
        for topic in topics
        for module in topic.get("modules", [])
        if module.get("modname") == "folder"
    ]
    if not folder_modules:
        pytest.skip("No folder modules found")
    folders_with_files = [m for m in folder_modules if m.get("contents")]
    assert len(folders_with_files) > 0, "Expected at least one folder with file contents"
