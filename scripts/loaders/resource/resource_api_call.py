"""
Test script to explore the resource module type using Moodle API.
Tests mod_resource_get_resources_by_course for Course 121, Module 14458
"""

import json
import logging
import os
import sys
from pathlib import Path

from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.loaders.APICaller import APICaller
from src.loaders.moodle import Moodle

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_production_moodle():
    """Setup Moodle with Production Credentials from Azure Key Vault."""
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
    
    logger.info(f"✅ Connected to Production Moodle: {prod_url}")
    return moodle


def test_get_resources_by_course(moodle: Moodle, course_id: int = 121):
    """
    Test mod_resource_get_resources_by_course API call.
    
    Args:
        course_id: Course ID to get resources from (default: 121)
    """
    print("\n" + "="*80)
    print(f"TEST: mod_resource_get_resources_by_course for Course {course_id}")
    print("="*80 + "\n")
    
    try:
        # API Call
        caller = APICaller(
            url=moodle.api_endpoint,
            params={**moodle.function_params, "courseids[0]": course_id},
            wsfunction="mod_resource_get_resources_by_courses"
        )
        
        response = caller.getJSON()
        
        # Display response
        print(f"✅ API call successful!\n")
        print(f"Response type: {type(response)}")
        print(f"Response keys: {response.keys() if isinstance(response, dict) else 'N/A'}")
        print(f"\nFull Response:")
        print("-" * 80)
        print(json.dumps(response, indent=2, ensure_ascii=False))
        print("-" * 80)
        
        # Analyze resources
        if isinstance(response, dict) and "resources" in response:
            resources = response["resources"]
            print(f"\n📊 Found {len(resources)} resources")
            
            for i, resource in enumerate(resources):
                print(f"\n--- Resource {i+1} ---")
                print(f"ID: {resource.get('id')}")
                print(f"Course Module: {resource.get('coursemodule')}")
                print(f"Course: {resource.get('course')}")
                print(f"Name: {resource.get('name')}")
                print(f"Intro: {resource.get('intro', 'N/A')[:100]}...")
                
                # Check for files
                if 'contentfiles' in resource:
                    files = resource['contentfiles']
                    print(f"Files: {len(files)}")
                    for file in files:
                        print(f"  - {file.get('filename')} ({file.get('filesize')} bytes)")
                        print(f"    Type: {file.get('mimetype')}")
                        print(f"    URL: {file.get('fileurl', 'N/A')[:80]}...")
        
        return response
        
    except Exception as e:
        logger.error(f"❌ API call failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_specific_resource_module(moodle: Moodle, module_id: int = 14458):
    """
    Test getting a specific resource module.
    
    Args:
        moodle: Moodle instance
        module_id: Module ID to inspect (default: 14458)
    """
    print("\n" + "="*80)
    print(f"TEST: Inspect Specific Resource Module {module_id}")
    print("="*80 + "\n")
    
    # Get course contents to find the module
    # First, we need to find which course this module belongs to
    # Let's assume it's course 121
    course_id = 121
    
    try:
        topics = moodle.get_course_contents(course_id)
        
        # Find the specific module
        target_module = None
        for topic in topics:
            for module in topic.modules:
                if module.id == module_id:
                    target_module = module
                    break
            if target_module:
                break
        
        if target_module:
            print(f"✅ Found module {module_id}\n")
            print(f"Module Details:")
            print(f"  ID: {target_module.id}")
            print(f"  Name: {target_module.name}")
            print(f"  Type (modname): {target_module.modname}")
            print(f"  Visible: {target_module.visible}")
            print(f"  URL: {target_module.url}")
            
            if hasattr(target_module, 'instance'):
                print(f"  Instance: {target_module.instance}")
            
            if target_module.contents:
                print(f"\n  Contents ({len(target_module.contents)} items):")
                for i, content in enumerate(target_module.contents):
                    print(f"\n  Content {i+1}:")
                    print(f"    Filename: {content.filename}")
                    print(f"    Type: {content.type}")
                    print(f"    Fileurl: {content.fileurl[:80] if content.fileurl else 'N/A'}...")
            
            return target_module
        else:
            print(f"❌ Module {module_id} not found in course {course_id}")
            return None
            
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Run all resource module tests."""
    print("\n" + "="*80)
    print("RESOURCE MODULE API TESTS")
    print("="*80)
    
    # Setup Production Moodle
    moodle = setup_production_moodle()
    
    # Test 1: Get all resources for course 121
    resources_response = test_get_resources_by_course(moodle, course_id=121)
    
    # Test 2: Inspect specific module 14458
    module = test_specific_resource_module(moodle, module_id=14458)
    
    print("\n" + "="*80)
    print("TESTS COMPLETED")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
