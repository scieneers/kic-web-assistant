"""
Test-Skript: Inspiziert URL-Modul (Kurs 27, Modul 2893)

Ruft core_course_get_contents für Kurs 27 ab und zeigt das JSON
für das URL-Modul 2893 an. Gibt die wichtigsten Felder und ggf. die verlinkte URL aus.
"""

import json
import logging
import os
from pathlib import Path

from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

from src.loaders.APICaller import APICaller

# Logging Setup
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Target
COURSE_ID = 27
MODULE_ID = 2893


def setup_production_credentials():
    """Hole Production Moodle Credentials aus Azure Key Vault."""
    try:
        key_vault_name = os.environ.get("KEY_VAULT_NAME", "kicwa-keyvault-lab")
        key_vault_uri = f"https://{key_vault_name}.vault.azure.net/"
        
        credential = DefaultAzureCredential()
        secret_client = SecretClient(vault_url=key_vault_uri, credential=credential)
        
        prod_url = secret_client.get_secret("DATA-SOURCE-PRODUCTION-MOODLE-URL").value
        prod_token = secret_client.get_secret("DATA-SOURCE-PRODUCTION-MOODLE-TOKEN").value
        
        return prod_url, prod_token
    
    except Exception as e:
        logger.error(f"❌ Fehler beim Setup: {e}")
        raise


def main():
    print("\n" + "=" * 80)
    print(f"URL-MODUL INSPEKTION")
    print("=" * 80)
    print(f"Kurs-ID:  {COURSE_ID}")
    print(f"Modul-ID: {MODULE_ID}")
    print("=" * 80)
    
    # Setup
    print("\n🔑 Hole Credentials aus Azure Key Vault...")
    prod_url, prod_token = setup_production_credentials()
    
    api_endpoint = f"{prod_url}webservice/rest/server.php"
    function_params = {
        "wstoken": prod_token,
        "moodlewsrestformat": "json",
    }
    
    # API Call: core_course_get_contents
    print(f"📡 API Call: core_course_get_contents für Kurs {COURSE_ID}...")
    caller = APICaller(
        url=api_endpoint,
        params=function_params,
        wsfunction="core_course_get_contents",
        courseid=COURSE_ID,
    )
    course_contents = caller.getJSON()
    print(f"✓ {len(course_contents)} Topics geladen\n")
    # Finde Modul 2893
    print(f"🔍 Suche Modul {MODULE_ID}...\n")
    target_module = None
    for topic in course_contents:
        for module in topic.get("modules", []):
            if module.get("id") == MODULE_ID:
                target_module = module
                break
        if target_module:
            break
    if not target_module:
        print(f"❌ Modul {MODULE_ID} nicht gefunden!")
        return
    print("=" * 80)
    print(f"MODUL {MODULE_ID} - JSON-STRUKTUR (core_course_get_contents)")
    print("=" * 80)
    print(json.dumps(target_module, indent=2, ensure_ascii=False))
    print("\n" + "=" * 80)
    print("\nZUSAMMENFASSUNG (core_course_get_contents):")
    print("-" * 80)
    print(f"Modul-ID:   {target_module.get('id')}")
    print(f"Modul-Name: {target_module.get('name')}")
    print(f"Modul-Typ:  {target_module.get('modname')}")
    print(f"URL:        {target_module.get('url')}")
    print(f"Visible:    {target_module.get('visible')}")
    # Extrahiere die verlinkte URL aus contents
    if target_module.get('contents'):
        for content in target_module['contents']:
            if content.get('type') == 'url':
                print(f"Verlinkte URL: {content.get('fileurl')}")
            elif content.get('type') == 'file':
                print(f"Datei: {content.get('filename')} ({content.get('fileurl')})")
    else:
        print("Keine contents im Modul gefunden.")
    print("\n" + "=" * 80)

    # --- mod_url_get_urls_by_courses ---
    print("\n\n--- mod_url_get_urls_by_courses ---")
    try:
        # Moodle REST API erwartet courseids[0]=27, nicht courseids=[27]
        url_params = function_params.copy()
        url_params["courseids[0]"] = COURSE_ID
        url_caller = APICaller(
            url=api_endpoint,
            params=url_params,
            wsfunction="mod_url_get_urls_by_courses",
        )
        url_modules = url_caller.getJSON()
        print(f"Antwort von mod_url_get_urls_by_courses: {json.dumps(url_modules, indent=2, ensure_ascii=False)}")
        # Suche das Zielmodul
        found = False
        for mod in url_modules.get('urls', []) if isinstance(url_modules, dict) else url_modules:
            if mod.get('cmid') == MODULE_ID:
                found = True
                print("\nZUSAMMENFASSUNG (mod_url_get_urls_by_courses):")
                print("-" * 80)
                print(f"cmid:        {mod.get('cmid')}")
                print(f"name:        {mod.get('name')}")
                print(f"intro:       {mod.get('intro')}")
                print(f"externalurl: {mod.get('externalurl')}")
                print(f"visible:     {mod.get('visible')}")
                break
        if not found:
            print(f"Modul {MODULE_ID} nicht in mod_url_get_urls_by_courses gefunden.")
    except Exception as e:
        print(f"Fehler beim mod_url_get_urls_by_courses-Call: {e}")


if __name__ == "__main__":
    main()
