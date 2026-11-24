"""
Download all files for a single Moodle module by ID.
Output: tests/loaders/module_data/<module_id>/
"""

import json
import logging
from pathlib import Path
import sys
import zipfile
import os

from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.loaders.moodle import Moodle
from src.loaders.APICaller import APICaller

# ============================================================================
# CONFIGURATION - Change this to test different modules
# ============================================================================
MODULE_ID = 2195  # ← Change this to test different modules
# ============================================================================

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_production_moodle():
    """Setup Moodle with Production credentials from Azure Key Vault"""
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


def save_json(data, filepath: Path, filename: str):
    """Save data as formatted JSON"""
    with open(filepath / filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"  ✓ Saved {filename}")


def download_file(url: str, params: dict, output_path: Path, description: str):
    """Download a file from URL to output_path"""
    try:
        caller = APICaller(url=url, params=params)
        caller.get()
        with open(output_path, "wb") as f:
            f.write(caller.response.content)
        size = output_path.stat().st_size
        logger.info(f"  ✓ Downloaded {description} ({size:,} bytes)")
        return True
    except Exception as e:
        logger.error(f"  ✗ Failed to download {description}: {e}")
        return False


def main():
    logger.info("=" * 80)
    logger.info(f"Loading Module ID: {MODULE_ID}")
    logger.info("=" * 80)
    
    # Setup output directory
    output_dir = Path(__file__).parent / "module_data" / str(MODULE_ID)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}\n")
    
    # Setup Moodle connection
    logger.info("[1] Connecting to Production Moodle...")
    try:
        moodle = setup_production_moodle()
        logger.info("  ✓ Connected\n")
    except Exception as e:
        logger.error(f"  ✗ Connection failed: {e}")
        return
    
    # Find module in courses
    logger.info(f"[2] Searching for module {MODULE_ID} in all courses...")
    try:
        courses = moodle.get_courses()
        logger.info(f"  ✓ Found {len(courses)} courses")
    except Exception as e:
        logger.error(f"  ✗ Failed to get courses: {e}")
        return
    
    target_module = None
    target_course = None
    
    for course in courses:
        try:
            topics = moodle.get_course_contents(course.id)
            for topic in topics:
                for module in topic.modules:
                    if module.id == MODULE_ID:
                        target_module = module
                        target_course = course
                        break
                if target_module:
                    break
            if target_module:
                break
        except Exception as e:
            continue
    
    if not target_module:
        logger.error(f"  ✗ Module {MODULE_ID} not found!")
        return
    
    logger.info(f"  ✓ Found in course: {target_course.fullname}")
    logger.info(f"  ✓ Module: {target_module.name}")
    logger.info(f"  ✓ Type: {target_module.modname}\n")
    
    # Save module metadata
    logger.info("[3] Saving module metadata...")
    module_data = {
        "id": target_module.id,
        "name": target_module.name,
        "modname": target_module.modname,
        "type": target_module.type,
        "url": str(target_module.url) if target_module.url else None,
        "visible": target_module.visible,
        "course_id": target_course.id,
        "course_name": target_course.fullname,
    }
    save_json(module_data, output_dir, "module_metadata.json")
    print()
    
    # Check contents[] array
    logger.info("[4] Checking contents[] array...")
    if target_module.contents:
        logger.info(f"  ✓ Found {len(target_module.contents)} content files")
        
        contents_data = []
        for idx, content in enumerate(target_module.contents):
            contents_data.append({
                "index": idx,
                "filename": content.filename,
                "fileurl": content.fileurl,
                "type": content.type,
            })
        
        save_json(contents_data, output_dir, "contents_array.json")
        
        # Download all content files
        logger.info(f"  → Downloading {len(target_module.contents)} files...")
        for idx, content in enumerate(target_module.contents):
            if content.fileurl:
                filename = content.filename or f"content_{idx}"
                file_path = output_dir / f"{idx}_{filename}"
                download_file(
                    content.fileurl,
                    moodle.download_params,
                    file_path,
                    filename
                )
    else:
        logger.info("  ✗ No contents[] array")
        save_json({"contents": None}, output_dir, "contents_array.json")
    
    print()
    
    # Handle H5P modules
    if target_module.modname == "h5pactivity":
        logger.info("[5] Processing H5P activity...")
        try:
            h5p_activities = moodle.get_h5p_module_ids(target_course.id)
            target_activity = None
            
            for activity in h5p_activities:
                if activity.coursemodule == MODULE_ID:
                    target_activity = activity
                    break
            
            if target_activity:
                logger.info(f"  ✓ Found H5P activity")
                
                # Save metadata
                h5p_data = {
                    "id": target_activity.id,
                    "coursemodule": target_activity.coursemodule,
                    "filename": str(target_activity.filename),
                    "fileurl": str(target_activity.fileurl),
                }
                save_json(h5p_data, output_dir, "h5p_metadata.json")
                
                # Download H5P package
                h5p_path = output_dir / target_activity.filename
                if download_file(
                    target_activity.fileurl,
                    moodle.download_params,
                    h5p_path,
                    "H5P package"
                ):
                    # Extract content.json
                    try:
                        with zipfile.ZipFile(h5p_path, 'r') as zip_ref:
                            if "content/content.json" in zip_ref.namelist():
                                content_json = json.loads(
                                    zip_ref.read("content/content.json")
                                )
                                save_json(content_json, output_dir, "h5p_content.json")
                    except Exception as e:
                        logger.warning(f"  ✗ Could not extract content.json: {e}")
            else:
                logger.warning("  ✗ H5P activity not found")
        except Exception as e:
            logger.error(f"  ✗ H5P processing failed: {e}")
    
    # Summary
    print()
    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Module ID: {MODULE_ID}")
    logger.info(f"Name: {target_module.name}")
    logger.info(f"Type: {target_module.modname}")
    logger.info(f"Course: {target_course.fullname}")
    logger.info(f"\nFiles saved to: {output_dir}")
    logger.info("\nDownloaded files:")
    
    total_size = 0
    for file in sorted(output_dir.iterdir()):
        if file.is_file():
            size = file.stat().st_size
            total_size += size
            logger.info(f"  - {file.name:<40} {size:>10,} bytes")
    
    logger.info(f"\nTotal size: {total_size:,} bytes")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
