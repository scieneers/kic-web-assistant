"""
Debug-Skript: Testet direkt die Book-Extraktion für Modul 19488.
"""

import logging
import os
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

from src.loaders.moodle import Moodle

# Logging Setup - DEBUG Level für maximale Ausgabe
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s:%(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Konfiguration
COURSE_ID = 41
TARGET_MODULE_ID = 19488  # Book: Programming Exercise 0


def setup_production_moodle():
    """Setup Moodle mit Production Credentials."""
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
    logger.info("="*80)
    logger.info("DEBUG: Book-Extraktion für Modul 19488")
    logger.info("="*80)
    
    moodle = setup_production_moodle()
    logger.info("✓ Moodle Setup abgeschlossen\n")
    
    # Lade Kurs
    courses = moodle.get_courses()
    course = next((c for c in courses if c.id == COURSE_ID), None)
    if not course:
        logger.error(f"Kurs {COURSE_ID} nicht gefunden!")
        return
    
    logger.info(f"✓ Kurs geladen: {course.fullname}\n")
    
    # Lade Module-Intros
    logger.info("Lade Module-Intros...")
    moodle._load_module_intros_for_course(COURSE_ID)
    logger.info(f"✓ {len(moodle.module_intros_cache)} Intros im Cache\n")
    
    # Lade Course Contents
    logger.info("Lade Course Contents...")
    course.topics = moodle.get_course_contents(COURSE_ID)
    logger.info(f"✓ {len(course.topics)} Topics geladen\n")
    
    # Finde das Ziel-Modul
    target_module = None
    for topic in course.topics:
        for module in topic.modules:
            if module.id == TARGET_MODULE_ID:
                target_module = module
                logger.info(f"✓ Modul gefunden: {module.name} (ID: {module.id})")
                logger.info(f"  ModName: {module.modname}")
                logger.info(f"  Type: {module.type}")
                logger.info(f"  Instance: {module.instance}")
                logger.info(f"  Contents: {len(module.contents) if module.contents else 0}\n")
                break
    
    if not target_module:
        logger.error(f"Modul {TARGET_MODULE_ID} nicht gefunden!")
        return
    
    # Debug: Zeige Contents
    logger.info("--- Contents des Moduls ---")
    for idx, content in enumerate(target_module.contents, 1):
        logger.info(f"Content {idx}:")
        logger.info(f"  Type: {type(content).__name__}")
        logger.info(f"  Type-Feld: {content.type}")
        logger.info(f"  Filename: {content.filename}")
        logger.info(f"  Hat filepath-Attribut: {hasattr(content, 'filepath')}")
        if hasattr(content, 'filepath'):
            logger.info(f"  Filepath: {content.filepath}")
        else:
            logger.info(f"  FEHLT: filepath")
        if hasattr(content, 'fileurl'):
            logger.info(f"  FileURL: {content.fileurl[:80]}..." if content.fileurl else "  FileURL: None")
        if hasattr(content, 'content'):
            logger.info(f"  Content: {str(content.content)[:100]}..." if content.content else "  Content: None")
        # Zeige ALLE Attribute
        logger.info(f"  Alle Attribute: {dir(content)}")
        logger.info("")
    
    # Extrahiere Book
    print("\n" + "="*80)
    print("STARTE extract_book()")
    print(f"Module ID: {target_module.id}")
    print(f"Module Name: {target_module.name}")
    print(f"Anzahl Contents: {len(target_module.contents) if target_module.contents else 0}")
    print("="*80 + "\n")
    
    # Inspiziere Contents
    if target_module.contents:
        for i, content in enumerate(target_module.contents[:3]):  # Erste 3
            print(f"\nContent {i}:")
            print(f"  type: {content.type}")
            print(f"  filename: {content.filename}")
            fileurl = content.fileurl if content.fileurl else "None"
            if fileurl != "None" and len(fileurl) > 80:
                print(f"  fileurl: {fileurl[:80]}...")
            else:
                print(f"  fileurl: {fileurl}")
            print(f"  Attribute: {[a for a in dir(content) if not a.startswith('_')]}")
    
    try:
        error = moodle.extract_book(target_module)
        
        if error:
            print(f"[X] Fehler bei extract_book: {error}")
        else:
            print("[OK] extract_book() abgeschlossen ohne Fehler\n")
    except Exception as e:
        print(f"[X] EXCEPTION in extract_book: {e}")
        import traceback
        traceback.print_exc()
    
    # Prüfe Ergebnis
    logger.info("="*80)
    logger.info("ERGEBNIS")
    logger.info("="*80)
    
    if target_module.book:
        book = target_module.book
        logger.info(f"[OK] Book erstellt:")
        logger.info(f"  Book ID: {book.book_id}")
        logger.info(f"  Module ID: {book.module_id}")
        logger.info(f"  Intro: {'Ja' if book.intro else 'Nein'}")
        logger.info(f"  Kapitel: {book.total_chapters}")
        logger.info(f"  Videos: {book.total_videos}")
        logger.info(f"  Anhänge: {book.total_attachments}\n")
        
        # Zeige Kapitel-Details
        logger.info("--- Kapitel-Details ---")
        for chapter in book.chapters:
            logger.info(f"\nKapitel: {chapter.title} (ID: {chapter.chapter_id})")
            logger.info(f"  HTML-Text: {len(chapter.html_text) if chapter.html_text else 0} Zeichen")
            logger.info(f"  Videos: {len(chapter.transcripts)}")
            logger.info(f"  Anhänge: {len(chapter.attachments)}")
        
        # Generiere Document
        logger.info("\n" + "="*80)
        logger.info("DOCUMENT-GENERIERUNG")
        logger.info("="*80 + "\n")
        
        doc = target_module.to_document(COURSE_ID)
        logger.info(f"Document Text ({len(doc.text)} Zeichen):")
        logger.info("-"*80)
        logger.info(doc.text[:1000] + "..." if len(doc.text) > 1000 else doc.text)
        logger.info("-"*80)
        
    else:
        logger.error("[X] KEIN BOOK ERSTELLT!")
        logger.error("  module.book ist None")


if __name__ == "__main__":
    main()
