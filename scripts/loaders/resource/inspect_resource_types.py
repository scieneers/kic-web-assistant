"""
Test script to analyze all resource module types across all courses.
Identifies what file types are used in resource modules (PDF, DOCX, etc.)
Saves API responses for specific formats (zip, wav, html, m4a, mp3, txt).
"""

import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path

from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.loaders.moodle import Moodle
from src.loaders.APICaller import APICaller

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Dateiformate, für die API-Antworten gespeichert werden sollen
FORMATS_TO_ANALYZE = ['zip', 'wav', 'html', 'm4a', 'mp3', 'txt']
MAX_SAMPLES_PER_FORMAT = 5  # Maximal 5 Beispiele pro Format speichern


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


def analyze_resource_types():
    """
    Analyze all resource modules across all courses to identify file types.
    Saves API responses for specific file formats (zip, wav, html, m4a, mp3, txt).
    """
    print("\n" + "="*80)
    print("RESOURCE MODULE TYPE ANALYSIS")
    print("="*80 + "\n")
    
    moodle = setup_production_moodle()
    
    # Get all courses
    print("📚 Fetching all courses...")
    courses = moodle.get_courses()
    print(f"✅ Found {len(courses)} courses\n")
    
    # Statistics collectors
    resource_types = Counter()  # file extensions
    resource_modules = []  # List of all resource modules with details
    
    total_resources = 0
    courses_with_resources = 0
    
    # Create output directory for API responses
    output_dir = Path(__file__).parent / "api_responses"
    output_dir.mkdir(exist_ok=True)
    
    # Track saved samples per format
    saved_samples = {fmt: 0 for fmt in FORMATS_TO_ANALYZE}
    
    # Iterate through all courses
    for i, course in enumerate(courses):
        logger.info(f"Processing course {i+1}/{len(courses)}: {course.id} - {course.fullname}")
        
        try:
            topics = moodle.get_course_contents(course.id)
            
            course_has_resources = False
            
            for topic in topics:
                for module in topic.modules:
                    # Check if it's a resource module
                    if module.modname == "resource":
                        course_has_resources = True
                        total_resources += 1
                        
                        # Analyze content files
                        if module.contents:
                            for content in module.contents:
                                # Get file type from content.type (bereinigt)
                                file_type = content.type.split('?')[0] if content.type else "unknown"
                                resource_types[file_type] += 1
                                
                                # Get filename extension
                                file_ext = None
                                if content.filename:
                                    file_ext = content.filename.split('.')[-1].lower()
                                    resource_types[file_ext] += 1
                                
                                # Store details
                                resource_modules.append({
                                    "course_id": course.id,
                                    "course_name": course.fullname,
                                    "module_id": module.id,
                                    "module_name": module.name,
                                    "filename": content.filename,
                                    "type": file_type,
                                    "extension": file_ext,
                                    "fileurl": content.fileurl[:100] if content.fileurl else None
                                })
                                
                                # Speichere API-Response für spezielle Dateiformate
                                if file_ext in FORMATS_TO_ANALYZE:
                                    if saved_samples[file_ext] < MAX_SAMPLES_PER_FORMAT:
                                        try:
                                            logger.info(f"Saving API response for {file_ext} file: {content.filename}")
                                            
                                            # API-Call für mod_resource_get_resources_by_courses
                                            # Moodle API erwartet courseids[0], courseids[1], etc.
                                            caller = APICaller(
                                                url=moodle.api_endpoint,
                                                params={**moodle.function_params, "courseids[0]": course.id},
                                                wsfunction="mod_resource_get_resources_by_courses"
                                            )
                                            response = caller.getJSON()
                                            
                                            # Speichere Response
                                            filename = f"{file_ext}_module_{module.id}_course_{course.id}.json"
                                            filepath = output_dir / filename
                                            
                                            with open(filepath, 'w', encoding='utf-8') as f:
                                                json.dump(response, f, indent=2, ensure_ascii=False)
                                            
                                            saved_samples[file_ext] += 1
                                            logger.info(f"✅ Saved: {filepath}")
                                            
                                        except Exception as e:
                                            logger.error(f"Error saving API response for module {module.id}: {e}")
            
            if course_has_resources:
                courses_with_resources += 1
                
        except Exception as e:
            logger.error(f"Error processing course {course.id}: {e}")
            continue
    
    # Print results
    print("\n" + "="*80)
    print("ANALYSIS RESULTS")
    print("="*80 + "\n")
    
    print(f"📊 Total Courses: {len(courses)}")
    print(f"📊 Courses with Resources: {courses_with_resources}")
    print(f"📊 Total Resource Modules: {total_resources}\n")
    
    print("="*80)
    print("API RESPONSES SAVED FOR SPECIAL FORMATS")
    print("="*80)
    for fmt in FORMATS_TO_ANALYZE:
        count = saved_samples[fmt]
        print(f"  {fmt:10s}: {count}/{MAX_SAMPLES_PER_FORMAT} samples saved")
    print()
    
    print("="*80)
    print("FILE TYPES FOUND (by extension/type)")
    print("="*80)
    for file_type, count in resource_types.most_common():
        print(f"  {file_type:20s}: {count:4d} occurrences")
    
    # Print sample resources for each type
    print("\n" + "="*80)
    print("SAMPLE RESOURCES BY TYPE")
    print("="*80)
    
    types_shown = set()
    for resource in resource_modules:
        file_type = resource["type"]
        if file_type not in types_shown:
            types_shown.add(file_type)
            print(f"\n📄 Type: {file_type}")
            print(f"   Course: {resource['course_name']}")
            print(f"   Module: {resource['module_name']}")
            print(f"   File: {resource['filename']}")
            print(f"   URL: {resource['fileurl']}...")
            
            if len(types_shown) >= 10:  # Show max 10 different types
                break
    
    # Save detailed results to JSON
    output_file = Path(__file__).parent / "resource_types_analysis.json"
    
    results = {
        "total_courses": len(courses),
        "courses_with_resources": courses_with_resources,
        "total_resource_modules": total_resources,
        "file_types": dict(resource_types.most_common()),
        "sample_resources": resource_modules[:50],  # Save first 50 as samples
        "api_responses_saved": {fmt: saved_samples[fmt] for fmt in FORMATS_TO_ANALYZE}
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Detailed results saved to: {output_file}")
    print(f"✅ API responses saved to: {output_dir}/")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80 + "\n")
    
    return results


if __name__ == "__main__":
    analyze_resource_types()
