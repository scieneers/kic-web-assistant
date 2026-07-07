"""
Test-Skript: Testet alle möglichen Book-WS-Funktionen auf Verfügbarkeit.

Prüft nur, ob die Funktion freigeschaltet ist (keine JSON-Analyse).
"""

import pytest
pytestmark = pytest.mark.integration

import logging
import os

from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

from src.loaders.APICaller import APICaller
from src.loaders.moodle import Moodle

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ⚙️ KONFIGURATION
TEST_COURSE_ID = 41
TEST_BOOK_ID = 83  # Aus mod_book_get_books_by_courses für Kurs 41
TEST_COURSEMODULE_ID = 19488  # Aus core_course_get_contents


@pytest.fixture
def moodle():
    return setup_production_moodle()


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


def test_wsfunction(moodle: Moodle, wsfunction: str, **params):
    """
    Testet eine WS-Funktion auf Verfügbarkeit.
    
    Args:
        moodle: Moodle instance
        wsfunction: Name der zu testenden Funktion
        **params: Optionale Parameter für die Funktion
        
    Returns:
        bool: True wenn erlaubt, False wenn blockiert
    """
    try:
        caller = APICaller(
            url=moodle.api_endpoint,
            params={**moodle.function_params, **params},
            wsfunction=wsfunction
        )
        
        response = caller.getJSON()
        
        # Prüfe auf Fehler in Response
        if isinstance(response, dict):
            if "exception" in response:
                logger.error(f"❌ {wsfunction}: {response.get('message', 'Unbekannter Fehler')}")
                return False
            if "warnings" in response and response["warnings"]:
                logger.warning(f"⚠️  {wsfunction}: Warnings vorhanden")
        
        logger.info(f"✅ {wsfunction}: ERLAUBT")
        
        # Zeige grundlegende Struktur
        if isinstance(response, dict):
            logger.info(f"   Response Keys: {list(response.keys())}")
        elif isinstance(response, list):
            logger.info(f"   Response: Liste mit {len(response)} Einträgen")
        
        return True
        
    except Exception as e:
        error_msg = str(e)
        
        # Parse Fehlertyp
        if "accessexception" in error_msg.lower():
            logger.error(f"❌ {wsfunction}: NICHT FREIGESCHALTET (accessexception)")
        elif "invalidparameter" in error_msg.lower():
            logger.warning(f"⚠️  {wsfunction}: ERLAUBT, aber ungültige Parameter")
            return True  # Funktion existiert, Parameter waren falsch
        else:
            logger.error(f"❌ {wsfunction}: {error_msg[:100]}")
        
        return False


def main():
    """Hauptfunktion: Testet alle Book-WS-Funktionen."""
    logger.info("="*80)
    logger.info("BOOK WS-FUNKTIONEN TEST")
    logger.info("="*80)
    
    moodle = setup_production_moodle()
    
    # Liste aller zu testenden Book-Funktionen
    test_cases = [
        # ===== MOD_BOOK Funktionen =====
        {
            "name": "mod_book_get_books_by_courses",
            "params": {"courseids[0]": TEST_COURSE_ID},
            "description": "Hole alle Books eines Kurses"
        },
        {
            "name": "mod_book_view_book",
            "params": {"bookid": TEST_BOOK_ID},
            "description": "View/Tracking für ein Book"
        },
        {
            "name": "mod_book_get_book_by_id",
            "params": {"bookid": TEST_BOOK_ID},
            "description": "Hole ein Book nach ID"
        },
        {
            "name": "mod_book_get_book_by_coursemodule",
            "params": {"cmid": TEST_COURSEMODULE_ID},
            "description": "Hole Book nach CourseModule ID"
        },
        
        # ===== CORE Funktionen (generisch, könnten für Book funktionieren) =====
        {
            "name": "core_course_get_module",
            "params": {"id": TEST_COURSEMODULE_ID},
            "description": "Hole Modul-Details (generisch)"
        },
        {
            "name": "core_course_get_contents",
            "params": {"courseid": TEST_COURSE_ID},
            "description": "Hole Kurs-Inhalte (bereits bekannt)"
        },
        
        # ===== Mögliche Chapter-Funktionen =====
        {
            "name": "mod_book_get_chapters",
            "params": {"bookid": TEST_BOOK_ID},
            "description": "Hole Kapitel eines Books"
        },
        {
            "name": "mod_book_get_chapter_content",
            "params": {"chapterid": 287},  # Aus structure.json
            "description": "Hole Kapitel-Inhalt"
        },
        
        # ===== Weitere mögliche Funktionen =====
        {
            "name": "mod_book_get_book_info",
            "params": {"bookid": TEST_BOOK_ID},
            "description": "Hole Book-Informationen"
        },
        {
            "name": "mod_book_export_book",
            "params": {"bookid": TEST_BOOK_ID},
            "description": "Export Book als Paket"
        },
    ]
    
    logger.info(f"\nTeste {len(test_cases)} WS-Funktionen...\n")
    
    allowed_functions = []
    blocked_functions = []
    
    for test in test_cases:
        logger.info(f"\n{'─'*60}")
        logger.info(f"Test: {test['name']}")
        logger.info(f"Beschreibung: {test['description']}")
        logger.info(f"Parameter: {test['params']}")
        logger.info(f"{'─'*60}")
        
        is_allowed = test_wsfunction(moodle, test['name'], **test['params'])
        
        if is_allowed:
            allowed_functions.append(test['name'])
        else:
            blocked_functions.append(test['name'])
    
    # Zusammenfassung
    logger.info("\n" + "="*80)
    logger.info("ZUSAMMENFASSUNG")
    logger.info("="*80)
    
    logger.info(f"\n✅ ERLAUBTE FUNKTIONEN ({len(allowed_functions)}):")
    for func in allowed_functions:
        logger.info(f"   - {func}")
    
    logger.info(f"\n❌ BLOCKIERTE FUNKTIONEN ({len(blocked_functions)}):")
    for func in blocked_functions:
        logger.info(f"   - {func}")
    
    logger.info("\n" + "="*80)
    logger.info("EMPFEHLUNG:")
    logger.info("="*80)
    
    if allowed_functions:
        logger.info("\nVerwende diese Funktionen für Book-Extraktion:")
        for func in allowed_functions:
            logger.info(f"   → {func}")
    else:
        logger.info("\n⚠️  Nur core_course_get_contents verfügbar!")
        logger.info("   → Kapitel müssen aus contents-Array extrahiert werden")
        logger.info("   → HTML-Dateien downloaden und parsen")


if __name__ == "__main__":
    main()
