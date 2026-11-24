"""Integration test for Resource module extraction (PDF)."""

import os

import pytest

from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

from src.loaders.moodle import Moodle

pytestmark = pytest.mark.integration

# Course 121, Module 14458 — Resource module with PDF
TEST_COURSE_ID = 121
TEST_MODULE_ID = 14458


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


@pytest.fixture(scope="module")
def resource_module(moodle):
    topics = moodle.get_course_contents(TEST_COURSE_ID)
    for topic in topics:
        for module in topic.modules:
            if module.id == TEST_MODULE_ID:
                return module
    pytest.fail(f"Module {TEST_MODULE_ID} not found in course {TEST_COURSE_ID}")


def test_resource_module_found(resource_module):
    assert resource_module is not None
    assert resource_module.id == TEST_MODULE_ID
    assert resource_module.modname == "resource"


def test_resource_has_file(resource_module):
    assert resource_module.contents, "Resource module should have file contents"
    assert resource_module.contents[0].fileurl, "File should have a download URL"


def test_resource_extraction_succeeds(moodle, resource_module):
    err = moodle.extract_resource(resource_module)
    assert err is None, f"Extraction returned error: {err}"
    assert resource_module.resource is not None, "resource attribute should be set after extraction"


def test_resource_is_supported(resource_module):
    assert resource_module.resource.is_supported, "PDF should be a supported resource type"
    assert resource_module.resource.is_pdf, "This resource should be a PDF"


def test_resource_text_extracted(resource_module):
    assert resource_module.resource.extracted_text, "Should have extracted text from PDF"
    assert len(resource_module.resource.extracted_text) > 100, "Extracted text should be substantial"


def test_resource_to_document(resource_module):
    doc = resource_module.to_document(TEST_COURSE_ID)
    assert doc is not None
    assert len(doc.text) > 0, "Document text should not be empty"
    assert "--- Dokument" in doc.text, "Document should contain resource content section"
