"""
Test script to fetch glossary entries using Moodle Web Service API.
Tests for Course 313, Module 26353 (glossary)
"""

import pytest
pytestmark = pytest.mark.integration

import json
import logging
import os
from pathlib import Path

from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

from src.loaders.APICaller import APICaller
from src.loaders.moodle import Moodle

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@pytest.fixture
def moodle():
    return setup_production_moodle()


def setup_production_moodle():
    """Setup Moodle client with Production credentials from Lab Key Vault"""
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


def test_glossary_apis(course_id: int, module_id: int):
    """Test various glossary API endpoints"""
    
    logger.info("=" * 80)
    logger.info("GLOSSARY API TEST")
    logger.info("=" * 80)
    logger.info(f"Course ID:  {course_id}")
    logger.info(f"Module ID:  {module_id}")
    logger.info("")
    
    # Setup
    moodle = setup_production_moodle()
    api_endpoint = moodle.api_endpoint
    token = moodle.token
    logger.info(f"Using Moodle: {moodle.base_url}")
    function_params = {
        "wstoken": token,
        "moodlewsrestformat": "json",
    }
    
    output_dir = Path(__file__).parent / "loaded_data" / "glossary" / "module_content"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    glossary_id = None
    target_glossary = None
    
    # Test 1: Get glossaries by courses (specialized API)
    logger.info("[1] Testing: mod_glossary_get_glossaries_by_courses")
    try:
        caller = APICaller(
            url=api_endpoint,
            params=function_params,
            wsfunction="mod_glossary_get_glossaries_by_courses",
            **{"courseids[0]": course_id}
        )
        glossaries = caller.getJSON()
        
        logger.info(f"    ✓ Response received")
        logger.info(f"    Glossaries found: {len(glossaries.get('glossaries', []))}")
        
        # Save full response
        with open(output_dir / "1_glossaries_by_courses.json", "w", encoding="utf-8") as f:
            json.dump(glossaries, f, indent=2, ensure_ascii=False)
        logger.info(f"    ✓ Saved to 1_glossaries_by_courses.json")
        
        # Find our specific glossary
        for g in glossaries.get("glossaries", []):
            if g.get("coursemodule") == module_id:
                target_glossary = g
                glossary_id = g.get("id")
                logger.info(f"\n    📖 Found target glossary:")
                logger.info(f"       Name: {g.get('name')}")
                logger.info(f"       ID: {glossary_id}")
                logger.info(f"       Intro: {g.get('intro', '')[:100]}...")
                break
        
        if not target_glossary:
            logger.error(f"    ✗ Glossary with coursemodule={module_id} not found!")
        
    except Exception as e:
        logger.warning(f"    ✗ Specialized API failed: {str(e)[:200]}...")
        logger.info(f"\n[FALLBACK] Trying core_course_get_contents instead...")
        
        # Fallback: Use standard course contents API
        try:
            caller = APICaller(
                url=api_endpoint,
                params=function_params,
                wsfunction="core_course_get_contents",
                courseid=course_id
            )
            course_contents = caller.getJSON()
            
            # Save full course contents
            with open(output_dir / "1_fallback_course_contents.json", "w", encoding="utf-8") as f:
                json.dump(course_contents, f, indent=2, ensure_ascii=False)
            logger.info(f"    ✓ Saved full course contents to 1_fallback_course_contents.json")
            
            # Find glossary module in topics
            logger.info(f"\n    🔍 Searching for module {module_id} in course topics...")
            found_module = None
            for topic in course_contents:
                for module in topic.get("modules", []):
                    if module.get("id") == module_id:
                        found_module = module
                        logger.info(f"    ✓ Found module in topic: {topic.get('name')}")
                        logger.info(f"       Module name: {module.get('name')}")
                        logger.info(f"       Module type: {module.get('modname')}")
                        logger.info(f"       URL: {module.get('url')}")
                        
                        # Save module data
                        with open(output_dir / "1_fallback_glossary_module.json", "w", encoding="utf-8") as f:
                            json.dump(module, f, indent=2, ensure_ascii=False)
                        logger.info(f"    ✓ Saved module details to 1_fallback_glossary_module.json")
                        
                        # Check what fields are available
                        logger.info(f"\n    📋 Available fields in module:")
                        for key in module.keys():
                            value = module[key]
                            if isinstance(value, (dict, list)):
                                logger.info(f"       • {key}: {type(value).__name__} (len={len(value)})")
                            else:
                                logger.info(f"       • {key}: {value}")
                        
                        # Extract instance ID (glossary ID)
                        if "instance" in module:
                            glossary_id = module.get("instance")
                            logger.info(f"\n    ✓ Extracted glossary ID from instance field: {glossary_id}")
                        
                        break
                if found_module:
                    break
            
            if not found_module:
                logger.error(f"    ✗ Module {module_id} not found in course contents!")
                return
                
        except Exception as fallback_error:
            logger.error(f"    ✗ Fallback also failed: {fallback_error}")
            import traceback
            traceback.print_exc()
            return
    
    if not glossary_id:
        logger.error("\n❌ Could not determine glossary ID - cannot fetch entries!")
        return
    
    # Test 2: Get entries by letter (ALL)
    logger.info("\n[2] Testing: mod_glossary_get_entries_by_letter (letter=ALL)")
    try:
        caller = APICaller(
            url=api_endpoint,
            params=function_params,
            wsfunction="mod_glossary_get_entries_by_letter",
            id=glossary_id,
            letter="ALL",
            **{"from": 0, "limit": 100}  # Max 100 Einträge
        )
        entries_response = caller.getJSON()
        
        entries = entries_response.get("entries", [])
        logger.info(f"    ✓ Entries found: {len(entries)}")
        logger.info(f"    Total count: {entries_response.get('count', 'N/A')}")
        
        # Save full response
        with open(output_dir / "2_entries_by_letter_ALL.json", "w", encoding="utf-8") as f:
            json.dump(entries_response, f, indent=2, ensure_ascii=False)
        logger.info(f"    ✓ Saved to 2_entries_by_letter_ALL.json")
        
        # Show first 3 entries
        logger.info("\n    📝 First 3 entries:")
        for idx, entry in enumerate(entries[:3]):
            logger.info(f"\n       [{idx+1}] Concept: {entry.get('concept')}")
            definition = entry.get('definition', '')
            # Remove HTML tags for preview
            import re
            definition_text = re.sub(r'<[^>]+>', '', definition)
            logger.info(f"           Definition: {definition_text[:100]}...")
            logger.info(f"           ID: {entry.get('id')}")
            logger.info(f"           Author: {entry.get('userfullname', 'N/A')}")
            
            # Check for attachments
            if entry.get('attachments'):
                logger.info(f"           Attachments: {len(entry['attachments'])} file(s)")
                for att in entry['attachments']:
                    logger.info(f"             - {att.get('filename')}")
        
    except Exception as e:
        logger.error(f"    ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 3: Get entries by date
    logger.info("\n[3] Testing: mod_glossary_get_entries_by_date")
    try:
        caller = APICaller(
            url=api_endpoint,
            params=function_params,
            wsfunction="mod_glossary_get_entries_by_date",
            id=glossary_id,
            order="CREATION",
            sort="DESC",
            **{"from": 0, "limit": 10}
        )
        date_entries = caller.getJSON()
        
        logger.info(f"    ✓ Entries found: {len(date_entries.get('entries', []))}")
        
        # Save
        with open(output_dir / "3_entries_by_date.json", "w", encoding="utf-8") as f:
            json.dump(date_entries, f, indent=2, ensure_ascii=False)
        logger.info(f"    ✓ Saved to 3_entries_by_date.json")
        
    except Exception as e:
        logger.error(f"    ✗ Failed: {e}")
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Glossary Name: {target_glossary.get('name')}")
    logger.info(f"Glossary ID: {glossary_id}")
    logger.info(f"Total Entries: {entries_response.get('count', 'N/A')}")
    logger.info(f"\nAll data saved to: {output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    # Test with known glossary
    test_glossary_apis(course_id=313, module_id=26353)
