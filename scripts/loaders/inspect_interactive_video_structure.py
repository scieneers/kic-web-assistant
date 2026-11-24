"""
Analyse-Skript für Interactive Video Struktur.

Lädt ein spezifisches H5P-Modul herunter und analysiert dessen content.json,
um zu sehen, welche zusätzlichen Inhalte extrahierbar sind (neben Transkripten).

Ausgabe: JSON-Dateien mit verschiedenen Aspekten des Moduls.
"""

import json
import logging
import os
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

from src.loaders.APICaller import APICaller
from src.loaders.moodle import Moodle

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# ⚙️ KONFIGURATION: Welches H5P-Modul soll analysiert werden?
TARGET_MODULE_ID = 2195  # Die Modul-ID (coursemodule ID)
COURSE_NAME_HINT = "Introduction to Machine Learning Part 1"  # Optional: Kurs-Name für Suche
EXCEL_PATH = Path("documentation/Oct_Nov_KIC-course completion rate.xlsx")  # Excel mit Kurs-Liste
OUTPUT_DIR = Path(__file__).parent / "interactive_video_analysis"


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
    Sucht die Course-ID in der Excel-Liste anhand eines Namenshinweises.
    
    Args:
        course_name_hint: Teil des Kursnamens zur Suche
        excel_path: Pfad zur Excel-Datei mit Spalten 'Live Courses' und 'courseID'
        
    Returns:
        Course-ID oder None
    """
    logger.info(f"🔍 Suche Kurs in Excel: '{course_name_hint}'...")
    
    if not excel_path.exists():
        logger.error(f"  ❌ Excel-Datei nicht gefunden: {excel_path}")
        return None
    
    df = pd.read_excel(excel_path)
    
    # Spalten prüfen
    if "Live Courses" not in df.columns or "courseID" not in df.columns:
        logger.error("  ❌ Excel muss Spalten 'Live Courses' und 'courseID' enthalten!")
        return None
    
    # Suche nach Kurs (Case-insensitive substring match)
    mask = df["Live Courses"].str.contains(course_name_hint, case=False, na=False)
    matches = df[mask]
    
    if matches.empty:
        logger.warning(f"  ⚠️  Kein Kurs gefunden, der '{course_name_hint}' enthält!")
        return None
    
    # Ersten Match nehmen
    course_id = int(matches.iloc[0]["courseID"])
    course_name = matches.iloc[0]["Live Courses"]
    
    logger.info(f"  ✓ Gefunden: {course_name} (ID: {course_id})")
    return course_id



def analyze_single_module(moodle: Moodle, module_id: int, course_id: int | None = None):
    """
    Analysiert ein einzelnes H5P-Modul.
    
    Args:
        moodle: Konfigurierte Moodle-Instanz
        module_id: Modul-ID (coursemodule ID)
        course_id: Course-ID (erforderlich!)
        
    Returns:
        Tuple von (activity, module_name, content) oder None
    """
    logger.info(f"🔍 Lade H5P-Modul {module_id} aus Kurs {course_id}...")
    
    if not course_id:
        logger.error("  ✗ course_id erforderlich!")
        return None
    
    # Hole Course Contents
    caller = APICaller(
        url=moodle.api_endpoint,
        params={
            **moodle.function_params,
            "wsfunction": "core_course_get_contents",
            "courseid": course_id
        }
    )
    
    try:
        caller.get()
        sections = caller.response.json()
    except Exception as e:
        logger.error(f"  ✗ Fehler beim Laden der Course Contents: {e}")
        return None
    
    # Finde das spezifische Modul
    target_module = None
    for section in sections:
        for module in section.get("modules", []):
            if module.get("id") == module_id:
                target_module = module
                break
        if target_module:
            break
    
    if not target_module:
        logger.error(f"  ✗ Modul {module_id} nicht in Kurs {course_id} gefunden!")
        return None
    
    # Prüfe ob es ein H5P-Modul ist
    if target_module.get("modname") != "h5pactivity":
        logger.error(f"  ✗ Modul {module_id} ist kein h5pactivity (sondern {target_module.get('modname')})!")
        return None
    
    logger.info(f"  ✓ Modul gefunden: {target_module.get('name')}")
    
    # Hole H5P-Activity Details (für fileurl)
    caller = APICaller(
        url=moodle.api_endpoint,
        params={
            **moodle.function_params,
            "wsfunction": "mod_h5pactivity_get_h5pactivities_by_courses",
            "courseids[0]": course_id
        }
    )
    
    try:
        caller.get()
        h5p_data = caller.response.json()
    except Exception as e:
        logger.error(f"  ✗ Fehler beim Laden der H5P-Activity-Daten: {e}")
        return None
    
    # Finde die Activity für unser Modul
    target_activity = None
    for activity in h5p_data.get("h5pactivities", []):
        if activity.get("coursemodule") == module_id:
            target_activity = activity
            break
    
    if not target_activity:
        logger.error(f"  ✗ H5P-Activity für Modul {module_id} nicht gefunden!")
        return None
    
    # Prüfe ob Package vorhanden
    if "package" not in target_activity or not target_activity["package"]:
        logger.error(f"  ✗ Modul hat kein H5P-Package!")
        return None
    
    package = target_activity["package"][0]
    if "fileurl" not in package:
        logger.error(f"  ✗ Keine fileurl im Package!")
        return None
    
    # Download H5P ZIP
    logger.info(f"  📦 Lade H5P-Datei herunter...")
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            h5pfile_call = APICaller(
                url=package["fileurl"],
                params=moodle.download_params
            )
            local_filename = h5pfile_call.getFile(package["filename"], tmp_dir)
            
            # Extrahiere content.json
            with zipfile.ZipFile(local_filename, "r") as zip_ref:
                zip_ref.extract("content/content.json", tmp_dir)
            
            content_json_path = Path(tmp_dir) / "content" / "content.json"
            with open(content_json_path, "r", encoding="utf-8") as f:
                content = json.load(f)
            
            logger.info(f"  ✓ content.json geladen")
            
            # Modulnamen aus Activity
            module_name = target_activity.get("name", f"Module_{module_id}")
            
            return (target_activity, module_name, content)
            
    except Exception as e:
        logger.error(f"  ✗ Fehler beim Download/Extraktion: {e}")
        return None


def find_interactive_videos(moodle: Moodle, course_id: int, limit: int = 5):
    """
    Findet Interactive Video H5P-Module in einem Kurs.
    
    Args:
        moodle: Konfigurierte Moodle-Instanz
        course_id: Kurs-ID
        limit: Max. Anzahl zu finden
        
    Returns:
        Liste von (activity_dict, module_name) Tuples
    """
    logger.info(f"🔍 Suche Interactive Videos in Kurs {course_id}...")
    
    # Hole Kursinhalt
    caller = APICaller(
        url=moodle.api_endpoint,
        params={
            **moodle.function_params,
            "wsfunction": "core_course_get_contents",
            "courseid": course_id
        }
    )
    caller.get()
    course_contents = caller.response.json()
    
    # Finde H5P-Module
    h5p_modules = []
    for section in course_contents:
        for module in section.get("modules", []):
            if module.get("modname") == "h5pactivity":
                h5p_modules.append(module)
    
    if not h5p_modules:
        logger.warning(f"  ⚠️  Keine H5P-Module in Kurs {course_id} gefunden")
        return []
    
    logger.info(f"  ✓ Gefunden: {len(h5p_modules)} H5P-Module")
    
    # Hole H5P-Details
    caller = APICaller(
        url=moodle.api_endpoint,
        params={
            **moodle.function_params,
            "wsfunction": "mod_h5pactivity_get_h5pactivities_by_courses",
            "courseids[]": course_id
        }
    )
    caller.get()
    h5p_data = caller.response.json()
    
    # Filtere Interactive Videos
    interactive_videos = []
    for activity in h5p_data.get("h5pactivities", []):
        if "package" not in activity or not activity["package"]:
            continue
        
        package = activity["package"][0]
        if "fileurl" not in package:
            continue
        
        # Download und prüfe ob interactiveVideo
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                h5pfile_call = APICaller(
                    url=package["fileurl"],
                    params=moodle.download_params
                )
                local_filename = h5pfile_call.getFile(package["filename"], tmp_dir)
                
                # Extrahiere content.json
                with zipfile.ZipFile(local_filename, "r") as zip_ref:
                    zip_ref.extract("content/content.json", tmp_dir)
                
                content_json_path = Path(tmp_dir) / "content" / "content.json"
                with open(content_json_path, "r", encoding="utf-8") as f:
                    content = json.load(f)
                
                # Prüfe ob interactiveVideo
                if "interactiveVideo" in content:
                    # Finde Modul-Namen
                    module_name = "Unknown"
                    for module in h5p_modules:
                        if module["id"] == activity["coursemodule"]:
                            module_name = module.get("name", "Unknown")
                            break
                    
                    interactive_videos.append((activity, module_name, content))
                    logger.info(f"  ✓ Interactive Video gefunden: {module_name}")
                    
                    if len(interactive_videos) >= limit:
                        break
        except Exception as e:
            logger.debug(f"  Fehler bei Activity {activity.get('id')}: {e}")
            continue
    
    logger.info(f"  ✓ {len(interactive_videos)} Interactive Videos identifiziert\n")
    return interactive_videos


def analyze_interactive_video(activity, module_name, content, output_dir: Path):
    """
    Analysiert ein Interactive Video und extrahiert alle Informationen.
    
    Args:
        activity: Activity Dict von Moodle API
        module_name: Name des Moduls
        content: content.json Dict
        output_dir: Ausgabeverzeichnis
    """
    video_id = activity["id"]
    safe_name = "".join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in module_name)
    video_dir = output_dir / f"{video_id}_{safe_name}"
    video_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"📹 Analysiere: {module_name} (ID: {video_id})")
    
    # 1. Speichere komplettes content.json
    with open(video_dir / "full_content.json", "w", encoding="utf-8") as f:
        json.dump(content, f, indent=2, ensure_ascii=False)
    logger.info(f"  ✓ Gespeichert: full_content.json")
    
    iv = content["interactiveVideo"]
    
    # 2. Video-Informationen
    video_info = {
        "module_name": module_name,
        "module_id": activity["coursemodule"],
        "h5p_activity_id": video_id,
        "video": {}
    }
    
    if "video" in iv and "files" in iv["video"]:
        video_info["video"] = {
            "url": iv["video"]["files"][0].get("path") if iv["video"]["files"] else None,
            "mime_type": iv["video"]["files"][0].get("mime") if iv["video"]["files"] else None
        }
    
    with open(video_dir / "video_info.json", "w", encoding="utf-8") as f:
        json.dump(video_info, f, indent=2, ensure_ascii=False)
    logger.info(f"  ✓ Video URL: {video_info['video'].get('url', 'N/A')}")
    
    # 3. Interaktionen (Fragen, Text, etc.)
    interactions = []
    if "assets" in iv and "interactions" in iv["assets"]:
        for idx, interaction in enumerate(iv["assets"]["interactions"]):
            interaction_data = {
                "index": idx,
                "duration": interaction.get("duration"),
                "library_name": interaction.get("action", {}).get("library", "Unknown"),
                "x_position": interaction.get("x"),
                "y_position": interaction.get("y"),
                "params": interaction.get("action", {}).get("params", {})
            }
            interactions.append(interaction_data)
        
        with open(video_dir / "interactions.json", "w", encoding="utf-8") as f:
            json.dump(interactions, f, indent=2, ensure_ascii=False)
        logger.info(f"  ✓ Interaktionen: {len(interactions)} gefunden")
        
        # Detailanalyse der Interaktionstypen
        interaction_types = {}
        for interaction in interactions:
            lib = interaction["library_name"]
            interaction_types[lib] = interaction_types.get(lib, 0) + 1
        
        logger.info(f"     Typen: {dict(interaction_types)}")
    else:
        logger.info(f"  ⚠️  Keine Interaktionen gefunden")
    
    # 4. Bookmarks (Kapitel)
    bookmarks = []
    if "bookmarks" in iv:
        for bookmark in iv["bookmarks"]:
            bookmark_data = {
                "time": bookmark.get("time"),
                "label": bookmark.get("label")
            }
            bookmarks.append(bookmark_data)
        
        with open(video_dir / "bookmarks.json", "w", encoding="utf-8") as f:
            json.dump(bookmarks, f, indent=2, ensure_ascii=False)
        logger.info(f"  ✓ Bookmarks: {len(bookmarks)} Kapitel")
    else:
        logger.info(f"  ⚠️  Keine Bookmarks gefunden")
    
    # 5. Zusammenfassung
    summary = {
        "module_name": module_name,
        "video_id": video_id,
        "has_video": bool(video_info["video"].get("url")),
        "interaction_count": len(interactions),
        "interaction_types": interaction_types if interactions else {},
        "bookmark_count": len(bookmarks),
        "has_summary": "summary" in iv
    }
    
    with open(video_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    logger.info(f"  ✓ Analyse abgeschlossen\n")
    
    return summary


if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("H5P Modul Struktur-Analyse")
    logger.info("=" * 80)
    logger.info(f"Modul-ID: {TARGET_MODULE_ID}\n")
    
    # Setup
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"📂 Ausgabe-Verzeichnis: {OUTPUT_DIR}\n")
    
    # Moodle Connection
    logger.info("🔐 Verbinde mit Moodle...")
    moodle = setup_production_moodle()
    logger.info("  ✓ Verbunden\n")
    
    # Finde Kurs (falls Name angegeben)
    course_id = None
    if COURSE_NAME_HINT:
        course_id = find_course_in_excel(COURSE_NAME_HINT, EXCEL_PATH)
        print()  # Leerzeile
    
    # Lade das spezifische Modul
    result = analyze_single_module(moodle, TARGET_MODULE_ID, course_id)
    
    if not result:
        logger.error("❌ Modul konnte nicht geladen werden!")
        exit(1)
    
    activity, module_name, content = result
    
    # Analysiere das Modul
    logger.info(f"📊 Analysiere Modul...\n")
    summary = analyze_interactive_video(activity, module_name, content, OUTPUT_DIR)
    
    # Gesamtzusammenfassung
    with open(OUTPUT_DIR / "analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump([summary], f, indent=2, ensure_ascii=False)
    
    logger.info("=" * 80)
    logger.info("✅ ANALYSE ABGESCHLOSSEN")
    logger.info("=" * 80)
    logger.info(f"\n📁 Ergebnisse in: {OUTPUT_DIR}")
    logger.info(f"📊 Modul-Typ: {list(content.keys())[0] if content else 'Unknown'}")
    logger.info(f"🔢 Interaktionen: {summary.get('interaction_count', 0)}")
    logger.info(f"🔖 Bookmarks: {summary.get('bookmark_count', 0)}")

