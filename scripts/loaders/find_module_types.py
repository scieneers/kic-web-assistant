"""
Suchskript: Findet spezifische Modultypen in IMPORTANT_COURSES.

Sucht nach Modultypen: hvp, label, lesson, url, data
Gibt am Ende eine saubere Liste aller Funde aus.
"""

import json
import logging
import os
from pathlib import Path

from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

from src.loaders.moodle import Moodle

# Logging Setup - nur Warnings für Moodle/Azure
logging.basicConfig(
    level=logging.WARNING, 
    format='%(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)

# Gesuchte Modultypen
TARGET_MODULE_TYPES = ["hvp","label", "lesson", "url", "data"]

# Important Courses
IMPORTANT_COURSES_FILE = Path(__file__).parent.parent.parent.parent / "documentation" / "IMPORTANT_COURSES.txt"



def load_all_courses(moodle: 'Moodle') -> list[int]:
    """Lädt alle Kurs-IDs aus Moodle."""
    try:
        courses = moodle.get_courses()
        return [course.id for course in courses]
    except Exception as e:
        logger.error(f"Fehler beim Laden aller Kurse: {e}")
        return []


def setup_production_moodle():
    """Setup Moodle mit Production Credentials aus Azure Key Vault."""
    try:
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
    
    except Exception as e:
        logger.error(f"❌ Fehler beim Setup der Moodle-Verbindung: {e}")
        logger.error("   Stellen Sie sicher, dass Sie mit Azure authentifiziert sind (az login)")
        raise


def search_modules_in_course(moodle: Moodle, course_id: int, target_types: list[str]) -> list[dict]:
    """
    Sucht nach spezifischen Modultypen in einem Kurs.
    
    Returns:
        Liste von Dicts: [{"course_id": int, "module_id": int, "module_type": str, "module_name": str}, ...]
    """
    found_modules = []
    
    try:
        topics = moodle.get_course_contents(course_id)
        
        for topic in topics:
            for module in topic.modules:
                if module.modname in target_types:
                    found_modules.append({
                        "course_id": course_id,
                        "module_id": module.id,
                        "module_type": module.modname,
                        "module_name": module.name,
                        "module_url": str(module.url) if module.url else None
                    })
    
    except Exception as e:
        logger.warning(f"Fehler beim Durchsuchen von Kurs {course_id}: {e}")
    
    return found_modules


def print_results_limited(modules: list[dict], target_types: list[str], max_per_type: int = 5):
    """Gibt für jeden Modultyp maximal max_per_type Module mit URL aus."""
    print("\n" + "=" * 80)
    print("ERGEBNISSE (max. 5 pro Typ)")
    print("=" * 80)
    if not modules:
        print("\n❌ Keine Module der gesuchten Typen gefunden.\n")
        return
    by_type = {t: [] for t in target_types}
    for module in modules:
        mod_type = module['module_type']
        if mod_type in by_type and module.get('module_url'):
            if len(by_type[mod_type]) < max_per_type:
                by_type[mod_type].append(module)
    for mod_type in target_types:
        mods = by_type[mod_type]
        print(f"📌 {mod_type.upper()} ({len(mods)} Modul(e)):")
        print("-" * 80)
        for module in mods:
            print(f"  Kurs-ID:  {module['course_id']}")
            print(f"  Modul-ID: {module['module_id']}")
            print(f"  Name:     {module['module_name']}")
            print(f"  URL:      {module['module_url']}")
            print()
    print("=" * 80)



def main():
    print("\n" + "=" * 80)
    print("MODULTYPEN-SUCHE IN ALLEN KURSEN")
    print("=" * 80)
    print(f"Gesuchte Typen: {', '.join(TARGET_MODULE_TYPES)}")
    try:
        # Setup
        moodle = setup_production_moodle()
        # Lade alle Kurse
        all_courses = load_all_courses(moodle)
        print(f"Kurse: {all_courses}")
        print("=" * 80)
        print("\n🔍 Durchsuche alle Kurse...\n")
        all_found_modules = []
        for i, course_id in enumerate(all_courses, start=1):
            print(f"[{i}/{len(all_courses)}] Kurs {course_id}...", end=" ")
            found = search_modules_in_course(moodle, course_id, TARGET_MODULE_TYPES)
            all_found_modules.extend(found)
            print(f"{'✓ ' + str(len(found)) + ' gefunden' if found else '—'}")
        # Ergebnisse ausgeben (max. 5 pro Typ)
        print_results_limited(all_found_modules, TARGET_MODULE_TYPES, max_per_type=5)
    except Exception as e:
        logger.error(f"\n❌ Fehler: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
