"""
Test-Skript: Untersucht die mod_book_get_books_by_courses API.

Dieses Skript testet:
1. Was mod_book_get_books_by_courses zurückgibt
2. Wie Book-Inhalte (Kapitel) abgerufen werden können
3. Vergleich mit core_course_get_contents für Book-Module
"""

import pytest
pytestmark = pytest.mark.integration

import json
import logging
import os
from pathlib import Path

from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

from src.loaders.APICaller import APICaller
from src.loaders.moodle import Moodle

# Logging Setup
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s:%(name)s: %(message)s')
logger = logging.getLogger(__name__)

# ⚙️ KONFIGURATION
TEST_COURSE_ID = 41  # Beispiel-Kurs zum Testen
OUTPUT_DIR = Path(__file__).parent / "api_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


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


def test_book_api(moodle: Moodle, course_id: int):
    """
    Testet mod_book_get_books_by_courses API.
    
    Args:
        moodle: Moodle instance mit Production Setup
        course_id: ID des Kurses zum Testen
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"TEST 1: mod_book_get_books_by_courses für Kurs {course_id}")
    logger.info(f"{'='*80}")
    
    try:
        # API Call: mod_book_get_books_by_courses
        caller = APICaller(
            url=moodle.api_endpoint,
            params={**moodle.function_params, "courseids[0]": course_id},
            wsfunction="mod_book_get_books_by_courses"
        )
        
        books_response = caller.getJSON()
        
        # Speichere vollständige Response
        output_file = OUTPUT_DIR / f"mod_book_get_books_by_courses_course_{course_id}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(books_response, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ API Response gespeichert: {output_file}")
        
        # Analysiere Response
        books = books_response.get("books", [])
        logger.info(f"\n📚 Gefundene Books: {len(books)}")
        
        if books:
            logger.info("\n--- Book Übersicht ---")
            for idx, book in enumerate(books, 1):
                logger.info(f"\nBook {idx}:")
                logger.info(f"  ID: {book.get('id')}")
                logger.info(f"  Name: {book.get('name')}")
                logger.info(f"  CourseModule ID: {book.get('coursemodule')}")
                logger.info(f"  Course ID: {book.get('course')}")
                logger.info(f"  Intro: {book.get('intro', '')[:100]}...")
                logger.info(f"  Numbering: {book.get('numbering')}")
                logger.info(f"  CustomTitles: {book.get('customtitles')}")
                logger.info(f"  Revision: {book.get('revision')}")
                logger.info(f"  TimeModified: {book.get('timemodified')}")
                
                # Wichtig: Wo sind die Kapitel/Inhalte?
                logger.info(f"  Alle Felder: {list(book.keys())}")
        
        return books
        
    except Exception as e:
        logger.error(f"❌ Fehler bei mod_book_get_books_by_courses: {e}")
        return []


def test_core_course_get_contents_for_books(moodle: Moodle, course_id: int):
    """
    Testet core_course_get_contents und filtert Book-Module.
    
    Args:
        moodle: Moodle instance mit Production Setup
        course_id: ID des Kurses zum Testen
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"TEST 2: core_course_get_contents - Book Module")
    logger.info(f"{'='*80}")
    
    try:
        # API Call: core_course_get_contents
        caller = APICaller(
            url=moodle.api_endpoint,
            params=moodle.function_params,
            wsfunction="core_course_get_contents",
            courseid=course_id
        )
        
        course_contents = caller.getJSON()
        
        # Finde alle Book-Module
        book_modules = []
        for topic in course_contents:
            for module in topic.get("modules", []):
                if module.get("modname") == "book":
                    book_modules.append(module)
        
        logger.info(f"\n📚 Gefundene Book Module in core_course_get_contents: {len(book_modules)}")
        
        if book_modules:
            # Speichere Book-Module
            output_file = OUTPUT_DIR / f"core_course_get_contents_books_course_{course_id}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(book_modules, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Book Module gespeichert: {output_file}")
            
            logger.info("\n--- Book Module Details (aus core_course_get_contents) ---")
            for idx, module in enumerate(book_modules, 1):
                logger.info(f"\nBook Modul {idx}:")
                logger.info(f"  ID: {module.get('id')}")
                logger.info(f"  Name: {module.get('name')}")
                logger.info(f"  Instance: {module.get('instance')}")
                logger.info(f"  ModName: {module.get('modname')}")
                logger.info(f"  URL: {module.get('url')}")
                
                # WICHTIG: contents Feld - enthält Kapitel-Dateien?
                contents = module.get('contents', [])
                logger.info(f"  Contents (Kapitel?): {len(contents)}")
                
                if contents:
                    logger.info(f"\n  --- Contents in diesem Book ---")
                    for content_idx, content_item in enumerate(contents, 1):
                        logger.info(f"\n    Item {content_idx}:")
                        logger.info(f"      Filename: {content_item.get('filename')}")
                        logger.info(f"      Filepath: {content_item.get('filepath')}")
                        logger.info(f"      Type: {content_item.get('type')}")
                        logger.info(f"      Filesize: {content_item.get('filesize', 0)} bytes")
                        logger.info(f"      FileURL: {content_item.get('fileurl')}")
                        logger.info(f"      MimeType: {content_item.get('mimetype')}")
                        logger.info(f"      Author: {content_item.get('author')}")
                else:
                    logger.info(f"  ⚠️ Keine Contents - Kapitel müssen über andere API abgerufen werden!")
        
        return book_modules
        
    except Exception as e:
        logger.error(f"❌ Fehler bei core_course_get_contents: {e}")
        return []


def test_book_chapters(moodle: Moodle, book_id: int):
    """
    Testet das Abrufen von Book-Kapiteln.
    
    Book-Kapitel könnten über verschiedene APIs verfügbar sein:
    - mod_book_view_book (nur Tracking)
    - Direkt aus book-Objekt (wenn chapters-Feld vorhanden)
    - Über spezielle Chapter-API
    
    Args:
        moodle: Moodle instance
        book_id: ID des Books (coursemodule ID)
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"TEST 3: Book Kapitel abrufen (Book ID: {book_id})")
    logger.info(f"{'='*80}")
    
    # Versuche 1: mod_book_view_book (meist nur für Tracking, gibt keine Kapitel zurück)
    try:
        logger.info("\nVersuch 1: mod_book_view_book")
        caller = APICaller(
            url=moodle.api_endpoint,
            params=moodle.function_params,
            wsfunction="mod_book_view_book",
            bookid=book_id
        )
        
        view_response = caller.getJSON()
        logger.info(f"✅ mod_book_view_book Response: {view_response}")
        
        output_file = OUTPUT_DIR / f"mod_book_view_book_id_{book_id}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(view_response, f, indent=2, ensure_ascii=False)
        
    except Exception as e:
        logger.warning(f"⚠️ mod_book_view_book fehlgeschlagen: {e}")
    
    # Hinweis: Kapitel müssen eventuell aus contents-Array gelesen werden
    # oder sind im mod_book_get_books_by_courses bereits enthalten
    logger.info("\n💡 HINWEIS:")
    logger.info("   Book-Kapitel könnten in mod_book_get_books_by_courses enthalten sein")
    logger.info("   Oder müssen aus contents-Array in core_course_get_contents gelesen werden")


def test_content_download(moodle: Moodle, fileurl: str, filename: str):
    """
    Testet das Herunterladen einer Book-Datei/Kapitel.
    
    Args:
        moodle: Moodle instance
        fileurl: URL der Datei
        filename: Name der Datei
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"TEST 4: Content-Download Test")
    logger.info(f"{'='*80}")
    logger.info(f"Datei: {filename}")
    logger.info(f"URL: {fileurl}")
    
    try:
        caller = APICaller(url=fileurl, params=moodle.download_params)
        caller.get()
        
        content_type = caller.response.headers.get('Content-Type', '')
        content_length = caller.response.headers.get('Content-Length', 0)
        
        logger.info(f"\n✅ Download erfolgreich!")
        logger.info(f"  Content-Type: {content_type}")
        logger.info(f"  Content-Length: {content_length} bytes")
        logger.info(f"  Status Code: {caller.response.status_code}")
        
        # Erste 500 chars als Preview (für HTML-Content)
        if 'html' in content_type.lower():
            text_preview = caller.response.text[:500]
            logger.info(f"\n  HTML Preview (erste 500 chars):")
            logger.info(f"  {text_preview[:200]}...")
        else:
            preview = caller.response.content[:200]
            logger.info(f"\n  Preview (erste 200 bytes): {preview[:100]}...")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Fehler beim Download: {e}")
        return False


def compare_apis(books_from_mod_api, books_from_core_api):
    """
    Vergleicht die Rückgaben beider APIs.
    
    Args:
        books_from_mod_api: Ergebnis von mod_book_get_books_by_courses
        books_from_core_api: Book-Module aus core_course_get_contents
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"VERGLEICH: mod_book_get_books_by_courses vs core_course_get_contents")
    logger.info(f"{'='*80}")
    
    logger.info(f"\nAnzahl Books:")
    logger.info(f"  mod_book_get_books_by_courses: {len(books_from_mod_api)}")
    logger.info(f"  core_course_get_contents (book): {len(books_from_core_api)}")
    
    # Vergleiche verfügbare Felder
    if books_from_mod_api and books_from_core_api:
        logger.info(f"\nVerfügbare Felder im Vergleich:")
        logger.info(f"  mod_book API: {list(books_from_mod_api[0].keys())}")
        logger.info(f"  core_course API: {list(books_from_core_api[0].keys())}")
        
        # Wichtig: Welche API hat die Kapitel-Daten?
        mod_has_contents = 'contents' in books_from_mod_api[0]
        core_has_contents = 'contents' in books_from_core_api[0]
        
        logger.info(f"\n📚 Kapitel-Daten verfügbar:")
        logger.info(f"  mod_book API hat 'contents': {mod_has_contents}")
        logger.info(f"  core_course API hat 'contents': {core_has_contents}")
        
        if core_has_contents:
            logger.info(f"\n✅ HINWEIS: core_course_get_contents liefert contents-Array")
            logger.info(f"   Aber: Bei Books könnten das nur Export-Dateien sein, nicht Kapitel-Inhalte!")
        
        if mod_has_contents:
            logger.info(f"\n✅ HINWEIS: mod_book API liefert 'contents'")


def main():
    """Hauptfunktion: Führt alle Tests aus."""
    logger.info("🔧 Setup Production Moodle...")
    moodle = setup_production_moodle()
    logger.info("✅ Moodle Setup abgeschlossen\n")
    
    # Test 1: mod_book_get_books_by_courses
    books_from_mod_api = test_book_api(moodle, TEST_COURSE_ID)
    
    # Test 2: core_course_get_contents (Filter: book)
    books_from_core_api = test_core_course_get_contents_for_books(moodle, TEST_COURSE_ID)
    
    # Test 3: Book Kapitel abrufen (falls Books gefunden)
    if books_from_core_api:
        first_book_instance = books_from_core_api[0].get('instance')
        if first_book_instance:
            test_book_chapters(moodle, first_book_instance)
    
    # Test 4: Content-Download (wenn Contents gefunden)
    if books_from_core_api and books_from_core_api[0].get('contents'):
        first_content = books_from_core_api[0]['contents'][0]
        test_content_download(
            moodle, 
            first_content.get('fileurl'), 
            first_content.get('filename')
        )
    
    # Vergleich der APIs
    compare_apis(books_from_mod_api, books_from_core_api)
    
    logger.info(f"\n{'='*80}")
    logger.info("✅ ALLE TESTS ABGESCHLOSSEN")
    logger.info(f"{'='*80}")
    logger.info(f"\n📁 Ausgabedateien: {OUTPUT_DIR}")
    logger.info("\nNächste Schritte:")
    logger.info("1. JSON-Dateien in api_outputs/ prüfen")
    logger.info("2. Herausfinden: Wo sind die Kapitel-Inhalte?")
    logger.info("3. Entscheiden: Welche API verwenden?")
    logger.info("4. extract_book() Methode in moodle.py implementieren")


if __name__ == "__main__":
    main()
