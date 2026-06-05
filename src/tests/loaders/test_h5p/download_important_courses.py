"""
Download course contents for all important courses.

This script downloads the complete course content (all modules) for each course
listed in IMPORTANT_COURSES.txt and saves them as JSON files in the 
IMPORTANT_COURSES directory structure.

Directory structure:
    IMPORTANT_COURSES/
        <course_id>/
            module_<module_id>.json
            module_<module_id>.json
            ...
"""

import json
import os
import sys
from pathlib import Path

# IMPORTANT: Set to PRODUCTION to use production Moodle credentials
os.environ["ENVIRONMENT"] = "PRODUCTION"

# Add src to path
sys.path.insert(0, str(Path(__file__).parents[4]))

from src.env import env
from src.loaders.APICaller import APICaller


def load_important_course_ids() -> list[int]:
    """Load course IDs from IMPORTANT_COURSES.txt"""
    courses_file = Path(__file__).parents[4] / "IMPORTANT_COURSES.txt"
    
    with open(courses_file, "r") as f:
        content = f.read().strip()
        # Parse list format: [99, 106, 313, ...]
        course_ids = json.loads(content)
    
    return course_ids


def get_course_contents(api_endpoint: str, function_params: dict, course_id: int) -> list[dict]:
    """Get all course contents/modules for a course."""
    caller = APICaller(
        url=api_endpoint,
        params=function_params,
        wsfunction="core_course_get_contents",
        courseid=course_id,
    )
    course_contents = caller.getJSON()
    return course_contents


def save_module_to_json(module: dict, course_dir: Path):
    """Save a single module to JSON file."""
    module_id = module.get("id", "unknown")
    filename = f"module_{module_id}.json"
    filepath = course_dir / filename
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(module, f, indent=2, ensure_ascii=False)


def main():
    print("=== DOWNLOAD IMPORTANT COURSES ===\n")
    
    # Verify PRODUCTION environment
    print(f"Environment: {os.environ.get('ENVIRONMENT', 'DEV')}")
    
    # Setup API connection (using PRODUCTION Moodle secrets)
    base_url = env.DATA_SOURCE_MOODLE_URL
    api_endpoint = f"{base_url}webservice/rest/server.php"
    token = env.DATA_SOURCE_MOODLE_TOKEN
    function_params = {
        "wstoken": token,
        "moodlewsrestformat": "json",
    }
    
    print(f"Moodle URL: {base_url}")
    print(f"Token configured: {'✓' if token and token != 'UNSET' else '✗'}")
    
    # Verify it's production Moodle
    if "staging" in base_url.lower():
        print("\n⚠️  WARNING: Using STAGING Moodle, not PRODUCTION!")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    else:
        print("✓ Using PRODUCTION Moodle")
    
    print()
    
    # Load important course IDs
    course_ids = load_important_course_ids()
    print(f"Found {len(course_ids)} important courses: {course_ids}\n")
    
    # Create output directory
    output_base = Path(__file__).parent / "IMPORTANT_COURSES"
    output_base.mkdir(exist_ok=True)
    
    # Download each course
    total_modules = 0
    for i, course_id in enumerate(course_ids, 1):
        print(f"[{i}/{len(course_ids)}] Processing Course {course_id}...")
        sys.stdout.flush()
        
        try:
            # Create course directory
            course_dir = output_base / str(course_id)
            course_dir.mkdir(exist_ok=True)
            
            # Get course contents
            course_contents = get_course_contents(api_endpoint, function_params, course_id)
            
            # Process each topic
            modules_count = 0
            for topic in course_contents:
                modules = topic.get("modules", [])
                for module in modules:
                    save_module_to_json(module, course_dir)
                    modules_count += 1
            
            total_modules += modules_count
            print(f"  ✓ Saved {modules_count} modules to {course_dir}")
            
        except Exception as e:
            print(f"  ✗ Error processing course {course_id}: {e}")
    
    print(f"\n=== SUMMARY ===")
    print(f"Courses processed: {len(course_ids)}")
    print(f"Total modules saved: {total_modules}")
    print(f"Output directory: {output_base}")
    print("\n✅ Download complete!")


if __name__ == "__main__":
    main()
