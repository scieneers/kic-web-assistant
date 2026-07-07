"""
Test script to discover all Moodle module types (modname) and download example files.
For each unique modname, downloads one example module with all available files.
Output: tests/loaders/loaded_data/<modname>/ directories
"""

import json
import logging
from pathlib import Path
import sys
import tempfile
import zipfile
import os

from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.loaders.moodle import Moodle
from src.loaders.APICaller import APICaller

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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


def setup_output_directory(modname: str):
    """Create output directory for a specific modname"""
    output_dir = Path(__file__).parent / "loaded_data" / modname
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_json(data, filepath: Path, filename: str):
    """Save data as formatted JSON"""
    with open(filepath / filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"✓ Saved {filename}")


def download_file(url: str, params: dict, output_path: Path, description: str):
    """Download a file from URL to output_path"""
    try:
        caller = APICaller(url=url, params=params)
        with open(output_path, "wb") as f:
            # Get raw response content
            caller.get()
            f.write(caller.response.content)
        logger.info(f"✓ Downloaded {description} ({output_path.stat().st_size} bytes)")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to download {description}: {e}")
        return False


def extract_zip_contents(zip_path: Path, output_dir: Path):
    """Extract ZIP file and document its contents"""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # List all files
            file_list = zip_ref.namelist()
            logger.info(f"ZIP contains {len(file_list)} files")
            
            # Save file list
            save_json(file_list, output_dir, "zip_file_list.json")
            
            # Extract all
            extract_dir = output_dir / "extracted"
            extract_dir.mkdir(exist_ok=True)
            zip_ref.extractall(extract_dir)
            logger.info(f"✓ Extracted ZIP to {extract_dir}")
            
            return file_list
    except Exception as e:
        logger.error(f"✗ Failed to extract ZIP: {e}")
        return []


def download_module_files(moodle, module, course, output_dir):
    """Download all available files for a given module"""
    logger.info(f"\n  → Downloading files for module: {module.name} (ID: {module.id})")
    
    # Save module metadata
    module_data = {
        "id": module.id,
        "name": module.name,
        "modname": module.modname,
        "type": module.type,
        "url": str(module.url) if module.url else None,
        "visible": module.visible,
        "course_id": course.id,
        "course_name": course.fullname,
    }
    save_json(module_data, output_dir, "module_metadata.json")
    
    # Check for contents[] array
    if module.contents:
        logger.info(f"    ✓ Module has {len(module.contents)} content files")
        contents_data = []
        for idx, content in enumerate(module.contents):
            content_info = {
                "index": idx,
                "filename": content.filename,
                "fileurl": content.fileurl,
                "type": content.type,
            }
            contents_data.append(content_info)
            logger.info(f"      [{idx}] {content.filename}")
        
        save_json(contents_data, output_dir, "contents_array.json")
        
        # Download first 3 files as examples (to avoid huge downloads)
        for idx, content in enumerate(module.contents[:3]):
            if content.fileurl:
                filename = content.filename or f"content_{idx}"
                file_path = output_dir / f"content_{idx}_{filename}"
                download_file(
                    content.fileurl,
                    moodle.download_params,
                    file_path,
                    f"content[{idx}]"
                )
    else:
        logger.info("    ✗ No contents[] array")
        save_json({"contents": None}, output_dir, "contents_array.json")
    
    # Handle h5pactivity modules
    if module.modname == "h5pactivity":
        logger.info("    ↳ Fetching H5P activity data...")
        try:
            h5p_activities = moodle.get_h5p_module_ids(course.id)
            target_activity = None
            for activity in h5p_activities:
                if activity.coursemodule == module.id:
                    target_activity = activity
                    break
            
            if target_activity:
                h5p_data = {
                    "id": target_activity.id,
                    "coursemodule": target_activity.coursemodule,
                    "filename": str(target_activity.filename),
                    "fileurl": str(target_activity.fileurl),
                }
                save_json(h5p_data, output_dir, "h5p_activity_metadata.json")
                
                # Download H5P package
                h5p_path = output_dir / target_activity.filename
                if download_file(
                    target_activity.fileurl,
                    moodle.download_params,
                    h5p_path,
                    "H5P package"
                ):
                    # Extract content.json only (not full ZIP to save space)
                    try:
                        with zipfile.ZipFile(h5p_path, 'r') as zip_ref:
                            if "content/content.json" in zip_ref.namelist():
                                content_json = json.loads(zip_ref.read("content/content.json"))
                                save_json(content_json, output_dir, "h5p_content.json")
                                logger.info("      ✓ Extracted content.json")
                    except Exception as e:
                        logger.warning(f"      ✗ Could not extract content.json: {e}")
        except Exception as e:
            logger.warning(f"    ✗ H5P extraction failed: {e}")


def main():
    logger.info("=" * 80)
    logger.info("Module Type Discovery - All modnames Analysis")
    logger.info("=" * 80)
    
    # Setup with Production credentials
    logger.info("\n[Setup] Connecting to Production Moodle...")
    try:
        moodle = setup_production_moodle()
        logger.info("✓ Connected to Production Moodle")
    except Exception as e:
        logger.error(f"✗ Failed to setup Moodle: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 1: Discover all module types
    logger.info("\n[Step 1] Discovering all module types across all courses...")
    
    try:
        courses = moodle.get_courses()
        logger.info(f"✓ Found {len(courses)} visible courses")
    except Exception as e:
        logger.error(f"✗ Failed to get courses: {e}")
        import traceback
        traceback.print_exc()
        return
    
    if not courses:
        logger.error("✗ No courses found! Check your Moodle credentials.")
        return
    
    # Collect all unique modnames with example modules
    modname_examples = {}  # {modname: (module, course)}
    total_modules = 0
    
    for idx, course in enumerate(courses):
        try:
            logger.info(f"  [{idx+1}/{len(courses)}] Scanning: {course.fullname[:60]}...")
            topics = moodle.get_course_contents(course.id)
            
            for topic in topics:
                for module in topic.modules:
                    total_modules += 1
                    modname = module.modname
                    
                    # Store first example of each modname
                    if modname not in modname_examples:
                        modname_examples[modname] = (module, course)
                        logger.info(f"    ✓ Found new type: {modname} (Module ID: {module.id})")
        
        except Exception as e:
            logger.warning(f"  ✗ Error scanning course {course.id}: {e}")
            continue
    
    logger.info(f"\n✓ Discovery complete!")
    logger.info(f"  Total modules scanned: {total_modules}")
    logger.info(f"  Unique modnames found: {len(modname_examples)}")
    logger.info(f"  Types: {sorted(modname_examples.keys())}")
    
    # Step 2: Download example files for each modname
    logger.info("\n" + "=" * 80)
    logger.info("[Step 2] Downloading example files for each modname...")
    logger.info("=" * 80)
    
    for modname in sorted(modname_examples.keys()):
        module, course = modname_examples[modname]
        output_dir = setup_output_directory(modname)
        
        logger.info(f"\n📦 Processing modname: {modname}")
        logger.info(f"  Example module: {module.name[:60]}")
        logger.info(f"  From course: {course.fullname[:60]}")
        logger.info(f"  Output: {output_dir}")
        
        try:
            download_module_files(moodle, module, course, output_dir)
        except Exception as e:
            logger.error(f"  ✗ Failed to download files: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Step 3: Summary
    logger.info("\n" + "=" * 80)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total modules scanned: {total_modules}")
    logger.info(f"Unique modnames found: {len(modname_examples)}")
    logger.info(f"\nModule types discovered:")
    
    for modname in sorted(modname_examples.keys()):
        output_dir = setup_output_directory(modname)
        file_count = len(list(output_dir.glob("*")))
        logger.info(f"  - {modname:<20} → {file_count} files in {output_dir.name}/")
    
    logger.info(f"\nAll data saved to: {Path(__file__).parent / 'loaded_data'}/")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
