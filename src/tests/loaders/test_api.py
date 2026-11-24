"""Integration tests: Moodle module type API availability and course content discovery."""

import os

import pytest

from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

from src.loaders.APICaller import APICaller
from src.loaders.moodle import Moodle

pytestmark = pytest.mark.integration

# Maps module type → (test_course_id, specific_wsfunction, wsfunction_params_factory)
# wsfunction is None when only core_course_get_contents is needed.
MODULE_TYPE_CONFIGS = [
    ("folder",   134, "mod_folder_get_folders_by_courses",    lambda cid: {"courseids[0]": cid}),
    ("book",      41, "mod_book_get_books_by_courses",         lambda cid: {"courseids[0]": cid}),
    ("hvp",       99, "mod_hvp_get_hvps_by_courses",           lambda cid: {"courseids[0]": cid}),
    ("label",    152, "mod_label_get_labels_by_courses",       lambda cid: {"courseids[0]": cid}),
    ("lesson",    51, "mod_lesson_get_lessons_by_courses",     lambda cid: {"courseids[0]": cid}),
    ("resource", 152, "mod_resource_get_resources_by_courses", lambda cid: {"courseids[0]": cid}),
    ("url",      180, "mod_url_get_urls_by_courses",           lambda cid: {"courseids[0]": cid}),
    ("glossary", 313, "mod_glossary_get_glossaries_by_courses",lambda cid: {"courseids[0]": cid}),
    ("quiz",     313, "mod_quiz_get_quizzes_by_courses",       lambda cid: {"courseids[0]": cid}),
    # TODO: find a course that actually contains forum / assign modules and update the course ID
    # ("forum",    ???, "mod_forum_get_forums_by_courses",       lambda cid: {"courseids[0]": cid}),
    # ("assign",   ???, "mod_assign_get_assignments",            lambda cid: {"courseids[0]": cid}),
    ("data",     313, "mod_data_get_databases_by_courses",     lambda cid: {"courseids[0]": cid}),
]

MODULE_TYPE_IDS = [cfg[0] for cfg in MODULE_TYPE_CONFIGS]


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


@pytest.mark.parametrize(
    "modname,course_id,wsfunction,params_factory",
    MODULE_TYPE_CONFIGS,
    ids=MODULE_TYPE_IDS,
)
def test_module_type_in_course(moodle, modname, course_id, wsfunction, params_factory):
    """core_course_get_contents should find at least one module of the expected type."""
    caller = APICaller(
        url=moodle.api_endpoint,
        params=moodle.function_params,
        wsfunction="core_course_get_contents",
        courseid=course_id,
    )
    topics = caller.getJSON()
    assert isinstance(topics, list), "core_course_get_contents should return a list"

    found = [
        module
        for topic in topics
        for module in topic.get("modules", [])
        if module.get("modname") == modname
    ]
    assert len(found) > 0, f"Expected at least one '{modname}' module in course {course_id}"


@pytest.mark.parametrize(
    "modname,course_id,wsfunction,params_factory",
    MODULE_TYPE_CONFIGS,
    ids=MODULE_TYPE_IDS,
)
def test_module_specific_api(moodle, modname, course_id, wsfunction, params_factory):
    """Specific module API should return a valid response or be skipped if not enabled."""
    try:
        caller = APICaller(
            url=moodle.api_endpoint,
            params={**moodle.function_params, **params_factory(course_id)},
            wsfunction=wsfunction,
        )
        response = caller.getJSON()
    except Exception as e:
        err = str(e).lower()
        if "accessexception" in err or "invalidrecord" in err:
            pytest.skip(f"{wsfunction} not available on this server: {e}")
        raise

    assert response is not None, f"{wsfunction} returned None"
    if isinstance(response, dict):
        assert "exception" not in response, (
            f"{wsfunction} returned exception: {response.get('message')}"
        )
