"""
Test script to trace the extraction process for a single Moodle course:
"Introduction to Machine Learning Part 1: Foundations"
"""

import json
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
import os

from src.loaders.moodle import Moodle


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
    moodle.download_params["token"] = prod_token  # ← FIX: Update download token!
    
    return moodle


def find_course_by_name(course_name: str):
    """Find course ID by name"""
    moodle = setup_production_moodle()
    courses = moodle.get_courses()
    
    for course in courses:
        if course_name.lower() in course.fullname.lower():
            print(f"\n{'='*70}")
            print(f"Found Course:")
            print(f"  ID: {course.id}")
            print(f"  Name: {course.fullname}")
            print(f"  URL: {course.url}")
            print(f"{'='*70}\n")
            return course.id
    
    print(f"Course '{course_name}' not found!")
    return None


def test_get_course_contents():
    """Test Step 1: Get course structure with topics and modules"""
    
    # Find the course
    course_name = "KI-Explorables für die Schule"
    course_id = find_course_by_name(course_name)
    
    if not course_id:
        return
    
    # Get course contents
    moodle = setup_production_moodle()
    topics = moodle.get_course_contents(course_id)
    
    print(f"\n{'='*70}")
    print(f"COURSE STRUCTURE")
    print(f"{'='*70}\n")
    
    print(f"Total Topics: {len(topics)}\n")
    
    for i, topic in enumerate(topics, 1):
        print(f"Topic {i}: {topic.name}")
        print(f"  ID: {topic.id}")
        print(f"  Summary: {topic.summary[:100]}..." if topic.summary else "  Summary: None")
        print(f"  Modules: {len(topic.modules)}")
        
        for j, module in enumerate(topic.modules, 1):
            print(f"    Module {j}: {module.name}")
            print(f"      Type: {module.type}")
            print(f"      ID: {module.id}")
            print(f"      Visible: {module.visible}")
            print(f"      URL: {module.url}")
            if module.contents:
                print(f"      Contents: {len(module.contents)} items")
        print()
    
    # Save to JSON for detailed inspection
    topics_data = [topic.model_dump(mode='json') for topic in topics]
    output_file = "outputs/ml_course_structure.json"
    os.makedirs("outputs", exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(topics_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*70}")
    print(f"Detailed structure saved to: {output_file}")
    print(f"{'='*70}")


def test_get_h5p_module_ids():
    """Test Step 2: Get H5P module IDs with download URLs"""
    
    course_name = "KI-Explorables für die Schule"
    course_id = find_course_by_name(course_name)
    
    if not course_id:
        return
    
    moodle = setup_production_moodle()
    h5p_activities = moodle.get_h5p_module_ids(course_id)
    
    print(f"\n{'='*70}")
    print(f"H5P ACTIVITIES")
    print(f"{'='*70}\n")
    
    print(f"Total H5P Activities: {len(h5p_activities)}\n")
    
    for i, activity in enumerate(h5p_activities, 1):
        print(f"H5P Activity {i}:")
        print(f"  Course Module ID: {activity.coursemodule}")
        print(f"  File URL: {activity.fileurl}")
        print(f"  Filename: {activity.filename}")
        print()
    
    return h5p_activities


def test_extract_single_module():
    """Test Step 3: Extract content from a single h5pactivity module (ID 74)"""
    
    course_name = "KI-Explorables für die Schule"
    course_id = find_course_by_name(course_name)
    
    if not course_id:
        return
    
    moodle = setup_production_moodle()
    
    # Step 1: Get course structure
    topics = moodle.get_course_contents(course_id)
    
    # Step 2: Get H5P download URLs
    h5p_activity_ids = moodle.get_h5p_module_ids(course_id)
    
    # Find the specific H5P activity for module 74
    target_h5p = None
    for activity in h5p_activity_ids:
        if activity.coursemodule == 74:
            target_h5p = activity
            break
    
    if not target_h5p:
        print("H5P activity for module 74 not found!")
        return
    
    print(f"\n{'='*70}")
    print(f"H5P ACTIVITY FOR MODULE 74")
    print(f"{'='*70}\n")
    print(f"Course Module: {target_h5p.coursemodule}")
    print(f"File URL: {target_h5p.fileurl}")
    print(f"Filename: {target_h5p.filename}")
    
    # Find the specific module (ID 74)
    target_module = None
    target_topic = None
    
    for topic in topics:
        for module in topic.modules:
            if module.id == 74:
                target_module = module
                target_topic = topic
                break
        if target_module:
            break
    
    if not target_module:
        print("Module ID 74 not found!")
        return
    
    print(f"\n{'='*70}")
    print(f"EXTRACTING MODULE 74")
    print(f"{'='*70}\n")
    
    print(f"Module Name: {target_module.name}")
    print(f"Module Type: {target_module.type}")
    print(f"Module URL: {target_module.url}\n")
    
    # Step 3: Extract content
    print("Before extraction:")
    print(f"  Transcripts: {target_module.transcripts}")
    print(f"  Text: {target_module.text}\n")
    
    # Run the extraction for this topic
    try:
        failed_modules = moodle.get_module_contents(target_topic, h5p_activity_ids)
        
        print("After extraction:")
        print(f"  Transcripts: {len(target_module.transcripts)} transcript(s)")
        if target_module.transcripts:
            for transcript in target_module.transcripts:
                if transcript:
                    print(f"    Language: {transcript.language}")
                    print(f"    Text length: {len(transcript.text)} characters")
                    print(f"    Preview: {transcript.text[:200]}...")
                else:
                    print(f"    None (extraction failed)")
        print(f"  Text: {target_module.text[:200] if target_module.text else None}...")
        
        if failed_modules:
            print(f"\n⚠️  Failed modules: {len(failed_modules)}")
            for failed in failed_modules:
                print(f"    - {failed.module_name} (ID: {failed.module_id}): {failed.reason}")
        
        # Save extracted module to JSON
        output_file = "outputs/ml_module_74_extracted.json"
        os.makedirs("outputs", exist_ok=True)
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(target_module.model_dump(mode='json'), f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*70}")
        print(f"Extracted module saved to: {output_file}")
        print(f"{'='*70}")
    
    except Exception as e:
        print(f"\n❌ Extraction failed with error:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run all tests
    print("=" * 70)
    print("STEP 1: Get Course Structure")
    print("=" * 70)
    test_get_course_contents()
    
    print("\n\n")
    print("=" * 70)
    print("STEP 2: Get H5P Module IDs")
    print("=" * 70)
    test_get_h5p_module_ids()
    
    print("\n\n")
    print("=" * 70)
    print("STEP 3: Extract Content")
    print("=" * 70)
    test_extract_single_module()