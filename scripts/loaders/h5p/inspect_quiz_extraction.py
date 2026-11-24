"""
Test-Skript: Lädt alle H5P-Dateien für einen Kurs herunter und analysiert sie

Dieses Skript macht einen API-Call für alle H5P Activities eines Kurses,
lädt alle Packages herunter und extrahiert die content.json Dateien.
"""

import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path

from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

from src.loaders.APICaller import APICaller
from src.loaders.moodle import Moodle

# ⚙️ KONFIGURATION
COURSE_ID = 6
MODULE_ID = 2195  # Ziel-Modul zum Highlighten
OUTPUT_DIR = Path(__file__).parent / "output"


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


def download_and_extract_all(moodle: Moodle, course_id: int):
    """
    Lädt alle H5P Activities eines Kurses herunter und extrahiert content.json.
    """
    print(f"\n{'='*80}")
    print(f"DOWNLOAD & EXTRACTION: Alle H5P Activities von Kurs {course_id}")
    print(f"{'='*80}\n")
    sys.stdout.flush()
    
    # API Call
    print(f"🔍 Lade H5P Activities...")
    sys.stdout.flush()
    h5p_activities = moodle.get_h5p_module_ids(course_id)
    print(f"   ✓ {len(h5p_activities)} H5P Activities gefunden\n")
    sys.stdout.flush()
    
    # Erstelle Output-Verzeichnis
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Statistik
    successful_downloads = 0
    failed_downloads = 0
    files_with_interactions = 0
    
    # Lade jede Activity herunter
    for idx, activity in enumerate(h5p_activities, start=1):
        module_id = activity.coursemodule
        filename = activity.filename
        
        is_target = (module_id == MODULE_ID)
        marker = " ⭐ TARGET" if is_target else ""
        
        print(f"[{idx}/{len(h5p_activities)}] Modul {module_id}{marker}")
        print(f"    Datei: {filename}")
        sys.stdout.flush()
        
        try:
            # Download H5P Package
            h5pfile_call = APICaller(url=activity.fileurl, params=moodle.download_params)
            
            with tempfile.TemporaryDirectory() as tmp_dir:
                local_filename = h5pfile_call.getFile(activity.filename, tmp_dir)
                
                # Extrahiere content.json
                with zipfile.ZipFile(local_filename, "r") as zip_ref:
                    zip_ref.extract("content/content.json", tmp_dir)
                
                content_json_path = Path(tmp_dir) / "content" / "content.json"
                
                with open(content_json_path, "r", encoding="utf-8") as f:
                    content = json.load(f)
                
                # Speichere content.json
                output_file = OUTPUT_DIR / f"content_module_{module_id}.json"
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(content, f, indent=2, ensure_ascii=False)
                
                # Prüfe auf Interaktionen
                has_interactions = False
                interaction_count = 0
                
                if "interactiveVideo" in content:
                    iv = content["interactiveVideo"]
                    if "interactions" in iv:
                        has_interactions = True
                        interaction_count = len(iv["interactions"])
                        files_with_interactions += 1
                
                status = f"✓ Gespeichert"
                if has_interactions:
                    status += f" | {interaction_count} Interaktionen gefunden"
                
                print(f"    {status}")
                print(f"    → {output_file.name}")
                sys.stdout.flush()
                
                successful_downloads += 1
                
        except Exception as e:
            print(f"    ✗ Fehler: {type(e).__name__}: {str(e)}")
            sys.stdout.flush()
            failed_downloads += 1
        
        print("")
        sys.stdout.flush()
    
    # Zusammenfassung
    print(f"{'='*80}")
    print("📊 ZUSAMMENFASSUNG")
    print(f"{'='*80}\n")
    print(f"Gesamt: {len(h5p_activities)} H5P Activities")
    print(f"✓ Erfolgreich: {successful_downloads}")
    print(f"✗ Fehlgeschlagen: {failed_downloads}")
    print(f"📋 Mit Interaktionen: {files_with_interactions}")
    print(f"\n💾 Alle Dateien gespeichert in: {OUTPUT_DIR.absolute()}")
    sys.stdout.flush()


def main():
    print("=" * 80)
    print("H5P DOWNLOAD & EXTRACTION TEST")
    print("=" * 80)
    sys.stdout.flush()
    
    # Setup
    print("\n🔐 Verbinde mit Production Moodle...")
    sys.stdout.flush()
    moodle = setup_production_moodle()
    print("✓ Mit Production Moodle verbunden\n")
    sys.stdout.flush()
    
    # Download & Extract
    download_and_extract_all(moodle, COURSE_ID)
    
    print("\n" + "=" * 80)
    print("✅ TEST ABGESCHLOSSEN")
    print("=" * 80)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
