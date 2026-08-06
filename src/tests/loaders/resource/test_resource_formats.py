"""Integration tests for HTML, Audio, and ZIP resource extraction."""

import os

import pytest

from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

from src.loaders.moodle import Moodle

pytestmark = pytest.mark.integration

# (course_id, module_id, expected_type_attr, description)
FORMAT_CASES = [
    (127, 10759, "is_audio", "MP3 audio file"),
    (191, 22060, "is_zip",   "ZIP archive"),
    (350, 29185, "is_html",  "HTML file"),
]


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


def _find_module(moodle, course_id, module_id):
    topics = moodle.get_course_contents(course_id)
    for topic in topics:
        for module in topic.modules:
            if module.id == module_id:
                return module
    return None


@pytest.mark.parametrize("course_id,module_id,type_attr,description", FORMAT_CASES)
def test_resource_format_extracted(moodle, course_id, module_id, type_attr, description):
    module = _find_module(moodle, course_id, module_id)
    assert module is not None, f"Module {module_id} not found in course {course_id}"

    err = moodle.extract_resource(module)
    assert err is None, f"{description}: extraction error: {err}"
    assert module.resource is not None, f"{description}: resource should not be None"
    assert getattr(module.resource, type_attr), f"{description}: expected {type_attr}=True"


@pytest.mark.parametrize("course_id,module_id,type_attr,description", FORMAT_CASES)
def test_resource_format_document(moodle, course_id, module_id, type_attr, description):
    module = _find_module(moodle, course_id, module_id)
    if module is None:
        pytest.skip(f"Module {module_id} not found")

    moodle.extract_resource(module)
    doc = module.to_document(course_id)
    assert doc is not None
    assert len(doc.text) > 0, f"{description}: document text should not be empty"
