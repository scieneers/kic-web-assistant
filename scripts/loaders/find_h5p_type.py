"""
Findet einen spezifischen H5P-Typ in Moodle-Kursen.

Usage:
    1. H5P_TYPE_TO_FIND anpassen (z.B. "H5P.QuestionSet")
    2. Script ausführen
    
Das Script durchsucht zunächst die IMPORTANT_COURSES und gibt an,
ob der H5P-Typ dort vorkommt oder nicht.
"""

import json
import os
import sys
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path

from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

sys.path.insert(0, str(Path(__file__).parents[3]))

from src.loaders.APICaller import APICaller
from src.loaders.moodle import Moodle

# ⚙️ KONFIGURATION
H5P_TYPE_TO_FIND = "H5P.CoursePresentation"  # <--- Hier den gesuchten H5P-Typ eintragen

# IMPORTANT_COURSES aus Datei laden
IMPORTANT_COURSES_FILE = Path(__file__).parents[3] / "documentation" / "IMPORTANT_COURSES.txt"


def load_important_courses():
    """Lädt Course-IDs aus IMPORTANT_COURSES.txt"""
    with open(IMPORTANT_COURSES_FILE, "r", encoding="utf-8") as f:
        content = f.read().strip()
        # Parse als JSON-Array
        return json.loads(content)


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


def extract_h5p_type(h5p_zip_path: str) -> str | None:
    """Extrahiert H5P-Typ aus h5p.json im ZIP."""
    try:
        with zipfile.ZipFile(h5p_zip_path, "r") as zip_ref:
            with zip_ref.open("h5p.json") as f:
                h5p_data = json.load(f)
                return h5p_data.get("mainLibrary", "")
    except Exception:
        return None


def get_all_courses(moodle: Moodle) -> list[int]:
    """Holt alle verfügbaren Kurs-IDs vom Moodle-Server."""
    try:
        courses = moodle.get_courses()
        return [course.id for course in courses]
    except Exception as e:
        print(f"⚠️ Fehler beim Abrufen aller Kurse: {e}")
        return []


def find_h5p_in_course(moodle: Moodle, course_id: int, target_type: str, verbose: bool = True):
    """
    Sucht nach einem H5P-Typ in einem Kurs.
    
    Returns:
        dict mit 'found': bool, 'modules': list[dict]
    """
    if verbose:
        print(f"\n📚 Durchsuche Kurs {course_id}...")
    
    try:
        # Hole H5P Activities
        h5p_activities = moodle.get_h5p_module_ids(course_id)
        if verbose:
            print(f"   ✓ {len(h5p_activities)} H5P Activities gefunden")
        
        if not h5p_activities:
            return {"found": False, "modules": []}
        
        matching_modules = []
        
        for activity in h5p_activities:
            try:
                # Download H5P Package
                h5pfile_call = APICaller(url=activity.fileurl, params=moodle.download_params)
                
                with tempfile.TemporaryDirectory() as tmp_dir:
                    local_filename = h5pfile_call.getFile(activity.filename, tmp_dir)
                    h5p_type = extract_h5p_type(local_filename)
                    
                    if h5p_type and target_type in h5p_type:
                        matching_modules.append({
                            "module_id": activity.coursemodule,
                            "filename": activity.filename,
                            "h5p_type": h5p_type
                        })
                        if verbose:
                            print(f"   ✅ GEFUNDEN! Module {activity.coursemodule}: {h5p_type}")
            
            except Exception as e:
                if verbose:
                    print(f"   ⚠️  Fehler bei Module {activity.coursemodule}: {e}")
                continue
        
        return {
            "found": len(matching_modules) > 0,
            "modules": matching_modules
        }
    
    except Exception as e:
        if verbose:
            print(f"   ✗ Fehler beim Laden des Kurses: {e}")
        return {"found": False, "modules": []}


def main():
    print("=" * 80)
    print(f"H5P-TYP SUCHE: {H5P_TYPE_TO_FIND}")
    print("=" * 80)
    
    # Setup
    print("\n🔐 Verbinde mit Production Moodle...")
    moodle = setup_production_moodle()
    print("✓ Verbunden\n")
    
    # Lade wichtige Kurse
    important_courses = load_important_courses()
    print(f"📋 IMPORTANT_COURSES: {important_courses}")
    print(f"   Anzahl: {len(important_courses)} Kurse\n")
    
    print("=" * 80)
    print("SCHRITT 1: IMPORTANT_COURSES durchsuchen")
    print("=" * 80)
    
    results = {}
    courses_with_type = []
    
    for course_id in important_courses:
        result = find_h5p_in_course(moodle, course_id, H5P_TYPE_TO_FIND)
        results[course_id] = result
        
        if result["found"]:
            courses_with_type.append(course_id)
    
    # Zusammenfassung
    print("\n" + "=" * 80)
    print("ZUSAMMENFASSUNG - IMPORTANT_COURSES")
    print("=" * 80)
    
    print(f"\n✅ H5P-Typ '{H5P_TYPE_TO_FIND}' gefunden in: {len(courses_with_type)} von {len(important_courses)} Kursen")
    
    if courses_with_type:
        print(f"\n📍 Kurse mit {H5P_TYPE_TO_FIND}:")
        for course_id in courses_with_type:
            modules = results[course_id]["modules"]
            print(f"\n   Kurs {course_id}:")
            for module in modules:
                print(f"      - Module {module['module_id']}: {module['h5p_type']}")
                print(f"        Datei: {module['filename']}")
    else:
        print(f"\n❌ '{H5P_TYPE_TO_FIND}' wurde in keinem der IMPORTANT_COURSES gefunden.")
    
    # Falls nicht gefunden, alle Kurse durchsuchen
    if not courses_with_type:
        print("\n" + "=" * 80)
        print("SCHRITT 2: Durchsuche ALLE Kurse bis H5P-Typ gefunden wird")
        print("=" * 80)
        
        # Hole alle verfügbaren Kurse
        print("\n🔍 Lade alle verfügbaren Kurse...")
        all_course_ids = get_all_courses(moodle)
        print(f"   ✓ {len(all_course_ids)} Kurse verfügbar")
        
        # Filter bereits durchsuchte Kurse
        remaining_courses = [c for c in all_course_ids if c not in results]
        print(f"   📋 {len(remaining_courses)} Kurse noch nicht durchsucht\n")
        
        found_in_search = False
        courses_checked = 0
        
        for course_id in remaining_courses:
            courses_checked += 1
            # Weniger Ausgabe für schnellere Suche
            result = find_h5p_in_course(moodle, course_id, H5P_TYPE_TO_FIND, verbose=False)
            results[course_id] = result
            
            if result["found"]:
                found_in_search = True
                courses_with_type.append(course_id)
                print(f"\n✅ GEFUNDEN in Kurs {course_id} (nach {courses_checked} Kursen)!")
                for module in result["modules"]:
                    print(f"   - Module {module['module_id']}: {module['h5p_type']}")
                    print(f"     Datei: {module['filename']}")
                break
            
            # Progress Update alle 10 Kurse
            if courses_checked % 10 == 0:
                print(f"   ... {courses_checked}/{len(remaining_courses)} Kurse durchsucht")
        
        print("\n" + "-" * 80)
        if not found_in_search:
            print(f"❌ '{H5P_TYPE_TO_FIND}' existiert in KEINEM der {len(all_course_ids)} Kurse!")
        print("-" * 80)
    
    print("\n" + "=" * 80)
    print("FERTIG!")
    print("=" * 80)
    
    if courses_with_type:
        print(f"\n✅ Insgesamt gefunden in {len(courses_with_type)} Kurs(en)")
        for course_id in courses_with_type:
            print(f"   - Kurs {course_id}: {len(results[course_id]['modules'])} Module")


if __name__ == "__main__":
    main()
