import logging
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
import os

from src.loaders.moodle import Moodle

logger = logging.getLogger(__name__)


def get_production_moodle() -> Moodle:
    """Initialize Moodle instance with production credentials"""
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


def test_extract_embedded_h5p():
    """Test extracting H5P Interactive Book content embedded within a Moodle page via extract_page()."""
    moodle = get_production_moodle()
    
    # Get course contents and find the target page module (ID: 25877 in course 322)
    topics = moodle.get_course_contents(322)
    
    page_module = None
    for topic in topics:
        for module in topic.modules:
            if module.id == 25877:
                page_module = module
                break
    
    assert page_module is not None, "Page module 25877 not found"
    logger.info(f"Found page: {page_module.name} (ID: {page_module.id})")
    
    # Call extract_page to process the page including embedded H5P content
    err_message = moodle.extract_page(page_module)
    assert err_message is None, f"extract_page returned an error: {err_message}"
    
    # Verify extraction results
    logger.info(f"\n{'='*60}")
    logger.info("Extraction Results:")
    logger.info(f"{'='*60}")
    
    # Check module text
    if page_module.text:
        logger.info(f"Page text extracted: {len(page_module.text)} characters")
    
    # Check transcripts (from embedded videos)
    if page_module.transcripts:
        logger.info(f"Transcripts extracted: {len(page_module.transcripts)}")
    
    # Check embedded H5P content
    if page_module.h5p_content_type:
        logger.info(f"H5P content type: {page_module.h5p_content_type}")
    
    if page_module.interactive_video:
        interactions = page_module.interactive_video.get("interactions", [])
        logger.info(f"H5P interactions extracted: {len(interactions)}")
        
        # Save extracted H5P content to output file
        output_dir = Path(__file__).parent / "document_outputs"
        output_dir.mkdir(exist_ok=True)
        
        h5p_type = page_module.h5p_content_type or "unknown"
        safe_library = h5p_type.replace(".", "_")
        output_file = output_dir / f"h5p_{safe_library}_{page_module.id}.txt"
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"Source: Page {page_module.id} - {page_module.name}\n")
            f.write(f"H5P Type: {h5p_type}\n")
            f.write("=" * 60 + "\n\n")
            for idx, interaction in enumerate(interactions):
                f.write(f"--- Embedded H5P Content {idx + 1} ---\n")
                f.write(interaction)
                f.write("\n\n")
        
        logger.info(f"Saved extracted H5P content to: {output_file}")
    else:
        logger.warning("No H5P content was extracted")
    
    # Assertions for test validation
    assert page_module.interactive_video is not None, "Expected embedded H5P content to be extracted"
    assert len(page_module.interactive_video.get("interactions", [])) > 0, "Expected at least one H5P interaction"
    
    logger.info("\nTest passed: Embedded H5P content successfully extracted via extract_page()")


if __name__ == "__main__":
    test_extract_embedded_h5p()