"""
Download and analyze a specific H5P module.

Usage: Set COURSE_ID and MODULE_ID, then run the script.
The script will download the H5P package and save the content.json with analysis.
"""

import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path

from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

sys.path.insert(0, str(Path(__file__).parents[4]))

from src.loaders.APICaller import APICaller
from src.loaders.moodle import Moodle

# ⚙️ KONFIGURATION
COURSE_ID = 41
MODULE_ID = 19126


def setup_production_moodle():
    """Setup Moodle mit Production Credentials aus Azure Key Vault."""
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


def main():
    print(f"=== Download H5P Module ===")
    print(f"Course: {COURSE_ID}")
    print(f"Module: {MODULE_ID}\n")
    
    # Setup
    print("🔐 Connecting to Production Moodle...")
    moodle = setup_production_moodle()
    print("✓ Connected\n")
    
    # Get H5P activities for course
    print(f"📚 Loading H5P activities for course {COURSE_ID}...")
    try:
        h5p_activities = moodle.get_h5p_module_ids(COURSE_ID)
        print(f"✓ Found {len(h5p_activities)} H5P activities\n")
    except Exception as e:
        print(f"✗ Error loading H5P activities: {e}")
        return
    
    # Find the specific module
    activity = next((a for a in h5p_activities if a.coursemodule == MODULE_ID), None)
    
    if not activity:
        print(f"✗ Module {MODULE_ID} is not an H5P activity in course {COURSE_ID}!")
        print(f"\nAvailable H5P modules in course {COURSE_ID}:")
        for a in h5p_activities[:10]:  # Show first 10
            print(f"  - Module {a.coursemodule}: {a.filename}")
        if len(h5p_activities) > 10:
            print(f"  ... and {len(h5p_activities) - 10} more")
        return
    
    print(f"✓ Found H5P module: {activity.filename}")
    print(f"📦 Downloading package...\n")
    
    # Download and extract
    h5pfile_call = APICaller(url=activity.fileurl, params=moodle.download_params)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        local_filename = h5pfile_call.getFile(activity.filename, tmp_dir)
        
        with zipfile.ZipFile(local_filename, "r") as zip_ref:
            # Extrahiere h5p.json und content.json
            zip_ref.extract("h5p.json", tmp_dir)
            zip_ref.extract("content/content.json", tmp_dir)
        
        h5p_json_path = Path(tmp_dir) / "h5p.json"
        content_json_path = Path(tmp_dir) / "content" / "content.json"
        
        # Lade beide Dateien
        with open(h5p_json_path, "r", encoding="utf-8") as f:
            h5p_data = json.load(f)
        
        with open(content_json_path, "r", encoding="utf-8") as f:
            content = json.load(f)
        
        # Save to output directory
        output_dir = Path(__file__).parent / "h5p_downloads"
        output_dir.mkdir(exist_ok=True)
        
        h5p_output_file = output_dir / f"course_{COURSE_ID}_module_{MODULE_ID}_h5p.json"
        content_output_file = output_dir / f"course_{COURSE_ID}_module_{MODULE_ID}_content.json"
        
        with open(h5p_output_file, "w", encoding="utf-8") as f:
            json.dump(h5p_data, f, indent=2, ensure_ascii=False)
        
        with open(content_output_file, "w", encoding="utf-8") as f:
            json.dump(content, f, indent=2, ensure_ascii=False)
        
        print(f"✓ h5p.json saved to: {h5p_output_file}")
        print(f"✓ Content saved to: {content_output_file}\n")
        
        # Analyze content
        print("=== Content Analysis ===\n")
        
        if "interactiveVideo" in content:
            iv = content["interactiveVideo"]
            
            # Check for video
            if "video" in iv and "files" in iv["video"]:
                video_url = iv["video"]["files"][0]["path"]
                print(f"Video URL: {video_url}")
            
            # Check for interactions
            interaction_list = []
            if "assets" in iv and "interactions" in iv["assets"]:
                interaction_list = iv["assets"]["interactions"]
                print(f"Interactions: {len(interaction_list)} (found in assets.interactions)")
            elif "interactions" in iv:
                interaction_list = iv["interactions"]
                print(f"Interactions: {len(interaction_list)} (found in interactions)")
            else:
                print("Interactions: None found")
            
            if interaction_list:
                print(f"\n=== Interaction Details ===\n")
                for idx, interaction in enumerate(interaction_list, 1):
                    library = interaction.get("action", {}).get("library", "UNKNOWN")
                    print(f"[{idx}] {library}")
                    
                    # Show specific details for each type
                    if "MultiChoice" in library or "SingleChoiceSet" in library:
                        params = interaction.get("action", {}).get("params", {})
                        question = params.get("question", "")[:60]
                        print(f"    Question: {question}...")
                    
                    elif "TrueFalse" in library:
                        params = interaction.get("action", {}).get("params", {})
                        question = params.get("question", "")[:60]
                        correct = params.get("correct", "")
                        print(f"    Question: {question}...")
                        print(f"    Answer: {correct}")
                    
                    elif "Blanks" in library:
                        params = interaction.get("action", {}).get("params", {})
                        text = params.get("text", "")[:60]
                        print(f"    Text: {text}...")
                    
                    elif "DragQuestion" in library:
                        params = interaction.get("action", {}).get("params", {})
                        task = params.get("question", {}).get("task", {})
                        dropzones = task.get("dropZones", [])
                        elements = task.get("elements", [])
                        print(f"    Dropzones: {len(dropzones)}")
                        print(f"    Elements: {len(elements)}")
                    
                    elif "Text" in library:
                        params = interaction.get("action", {}).get("params", {})
                        text = params.get("text", "")[:60]
                        print(f"    Text: {text}...")
        else:
            print("⚠️  Not an Interactive Video H5P!")
            print(f"Content keys: {list(content.keys())}")
        
        print(f"\n✅ Done! Check {content_output_file}")


if __name__ == "__main__":
    main()
