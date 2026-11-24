"""
Test-Skript: Zeigt die finalen LlamaIndex Documents für einen Kurs und ein Modul.

Dieses Skript lädt einen kompletten Kurs mit allen Modulen und zeigt:
1. Das Kurs-Übersichtsdokument
2. Ein spezifisches Modul-Dokument (z.B. Modul 2195)

So sehen wir, welche Inhalte aktuell in die Vektordatenbank gelangen.
"""

import json
import logging
import os
from pathlib import Path

import pandas as pd
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

from src.loaders.moodle import Moodle

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s: %(message)s')
logger = logging.getLogger(__name__)

# ⚙️ KONFIGURATION
COURSE_ID = 41  # Kurs, der Modul 2195 enthält (Introduction to Machine Learning Part 1)
TARGET_MODULE_ID = 19126  # Das Modul mit Quiz-Fragen zum Testen
OUTPUT_DIR = Path(__file__).parent / "document_outputs"
PROCESS_ONLY_TARGET_MODULE = True  # Wenn True: Nur TARGET_MODULE_ID verarbeiten, sonst alle Module


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


def find_course_in_excel(course_name_hint: str, excel_path: Path) -> int | None:
    """
    Sucht die Course-ID in der Excel-Liste.
    """
    logger.info(f"🔍 Suche Kurs in Excel: '{course_name_hint}'...")
    
    if not excel_path.exists():
        logger.error(f"  ✗ Excel-Datei nicht gefunden: {excel_path}")
        return None
    
    df = pd.read_excel(excel_path)
    
    if "Live Courses" not in df.columns or "courseID" not in df.columns:
        logger.error("  ✗ Excel muss Spalten 'Live Courses' und 'courseID' enthalten!")
        return None
    
    mask = df["Live Courses"].str.contains(course_name_hint, case=False, na=False)
    matches = df[mask]
    
    if matches.empty:
        logger.warning(f"  ⚠️  Kein Kurs gefunden, der '{course_name_hint}' enthält!")
        return None
    
    course_id = int(matches.iloc[0]["courseID"])
    course_name = matches.iloc[0]["Live Courses"]
    
    logger.info(f"  ✓ Gefunden: {course_name} (ID: {course_id})")
    return course_id


def load_course_with_documents(
    moodle: Moodle,
    course_id: int,
    target_module_id: int | None = TARGET_MODULE_ID,
    process_only_target_module: bool = PROCESS_ONLY_TARGET_MODULE,
):
    """Lädt einen Kurs komplett mit allen Topics und Modulen.

    Gibt ein MoodleCourse-Objekt zurück, aus dem anschließend
    LlamaIndex-Documents generiert werden können.
    """
    logger.info(f"\n📚 Lade Kurs {course_id} mit allen Inhalten...")

    # Hole Course-Basis-Daten
    courses = moodle.get_courses()
    course = next((c for c in courses if c.id == course_id), None)

    if not course:
        logger.error(f"  ✗ Kurs {course_id} nicht gefunden!")
        return None

    logger.info(f"  ✓ Kurs geladen: {course.fullname}")

    # Lade Module-Intros (Resource, Glossary, Page, etc.) für den Kurs
    # WICHTIG: Muss VOR get_course_contents() aufgerufen werden (wie in Produktion)
    moodle._load_module_intros_for_course(course_id)
    logger.info(f"  ✓ Module-Intros geladen ({len(moodle.module_intros_cache)} im Cache)")

    # Lade Topics/Modules
    course.topics = moodle.get_course_contents(course_id)
    logger.info(f"  ✓ {len(course.topics)} Topics geladen")

    # Extrahiere Texte aus Modulen (HTML-Pages, H5P, etc.)
    total_modules = sum(len(topic.modules) for topic in course.topics)
    logger.info(f"  📦 Verarbeite {total_modules} Module...")

    # Hole H5P Activities für den Kurs
    h5p_activities = moodle.get_h5p_module_ids(course_id)
    logger.info(f"  ✓ {len(h5p_activities)} H5P-Activities gefunden")

    # Extrahiere Inhalte aus allen Topics (wie in der echten Pipeline)
    for topic in course.topics:
        # Optional: Nur ein spezifisches Modul verarbeiten
        if process_only_target_module and target_module_id is not None:
            # Filtere Module im Topic
            original_modules = topic.modules
            topic.modules = [m for m in original_modules if m.id == target_module_id]

            if len(topic.modules) > 0:
                logger.info(f"  📌 Verarbeite nur Modul {target_module_id} aus Topic '{topic.name}'")
                moodle.get_module_contents(topic, h5p_activities)

            # Stelle Original-Liste wieder her (für spätere Nutzung)
            topic.modules = original_modules
        else:
            moodle.get_module_contents(topic, h5p_activities)

    logger.info(f"  ✓ Alle Module verarbeitet\n")
    return course


def save_document_to_file(doc, filename: str, output_dir: Path):
    """
    Speichert ein LlamaIndex Document als lesbare Datei.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / filename
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("DOCUMENT TEXT\n")
        f.write("=" * 80 + "\n\n")
        f.write(doc.text)
        f.write("\n\n")
        f.write("=" * 80 + "\n")
        f.write("METADATA\n")
        f.write("=" * 80 + "\n\n")
        f.write(json.dumps(doc.metadata, indent=2, ensure_ascii=False))
    
    logger.info(f"  💾 Gespeichert: {output_file}")


def run_ingestion(
    course_id: int = COURSE_ID,
    target_module_id: int | None = TARGET_MODULE_ID,
    process_only_target_module: bool = PROCESS_ONLY_TARGET_MODULE,
    output_dir: Path = OUTPUT_DIR,
):
    """Führt den kompletten Ingestion-Prozess für einen Kurs aus.

    - Verwendet die produktive Moodle-Konfiguration (wie im echten Loader)
    - Lädt Kurs + Module
    - Erzeugt LlamaIndex-Documents
    - Schreibt die wichtigsten Documents als TXT-Dateien nach ``output_dir``

    Rückgabe: Liste der erzeugten Documents.
    """
    logger.info("=" * 80)
    logger.info("DOCUMENT EXTRACTION TEST - MODUL 2195")
    logger.info("=" * 80)

    # Setup
    moodle = setup_production_moodle()

    # Lade Kurs komplett
    logger.info(f"\n📚 Lade Kurs {course_id}...")
    course = load_course_with_documents(
        moodle,
        course_id,
        target_module_id=target_module_id,
        process_only_target_module=process_only_target_module,
    )
    if not course:
        logger.error("❌ Kurs konnte nicht geladen werden!")
        return []

    # Generiere Documents
    logger.info("📄 Generiere LlamaIndex Documents...\n")
    documents = course.to_document()

    logger.info(f"  ✓ {len(documents)} Documents erstellt:")
    logger.info("    - 1x Kurs-Übersicht")
    logger.info(f"    - {len(documents) - 1}x Modul-Documents\n")

    # Speichere Kurs-Übersichtsdokument (erstes Document)
    logger.info("💾 Speichere Dokumente...")
    if documents:
        course_doc = documents[0]
        save_document_to_file(
            course_doc,
            f"course_{course_id}_overview.txt",
            output_dir,
        )

    # Finde und speichere Modul-Dokument (falls target_module_id gesetzt ist)
    if target_module_id is not None:
        module_doc = next(
            (doc for doc in documents[1:] if doc.metadata.get("module_id") == target_module_id),
            None,
        )
        if module_doc:
            save_document_to_file(
                module_doc,
                f"module_{target_module_id}_document.txt",
                output_dir,
            )
        else:
            logger.warning(f"  ⚠️  Modul {target_module_id} nicht in Documents gefunden!")

    # Speichere auch die ersten 3 anderen Module als Beispiele
    logger.info("\n📋 Speichere zusätzlich erste 3 Module als Beispiele...")
    for i, doc in enumerate(documents[1:4], start=1):
        module_id = doc.metadata.get("module_id", f"unknown_{i}")
        save_document_to_file(
            doc,
            f"module_{module_id}_example.txt",
            output_dir,
        )

    logger.info("\n" + "=" * 80)
    logger.info("✅ FERTIG!")
    logger.info(f"Ausgabe in: {output_dir.absolute()}")
    logger.info("=" * 80)

    return documents


def main():
    # CLI-Entry: verhält sich wie bisher, nutzt Default-Konfiguration
    run_ingestion()


if __name__ == "__main__":
    main()
