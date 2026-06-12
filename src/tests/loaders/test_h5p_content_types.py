"""
Analyse-Skript für H5P Content-Typen in Live-Kursen.

Durchsucht alle Kurse aus der Excel-Liste (Spalte "courseID") und analysiert,
welche H5P Content-Typen verwendet werden.

Ausgabe: Sortierte Tabelle mit absoluter und prozentualer Verteilung.
"""

import json
import logging
import os
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

from src.loaders.APICaller import APICaller
from src.loaders.moodle import Moodle

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


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


def load_live_courses(excel_path: str) -> list[int]:
    """
    Liest die Excel-Datei und extrahiert die Kurs-IDs aus der Spalte 'courseID'.
    
    Args:
        excel_path: Pfad zur Excel-Datei
        
    Returns:
        Liste von Kurs-IDs (integers)
    """
    df = pd.read_excel(excel_path)
    
    # Die Spalte heißt "courseID"
    column_name = "courseID"
    
    if column_name not in df.columns:
        raise ValueError(f"Spalte 'courseID' nicht gefunden. Verfügbare Spalten: {df.columns.tolist()}")
    
    # Extrahiere die Kurs-IDs (entferne NaN-Werte)
    course_ids = df[column_name].dropna().astype(int).tolist()
    
    print(f"📚 Gefunden: {len(course_ids)} Live-Kurse")
    return course_ids


def extract_h5p_content_type(h5p_zip_path: str) -> str | None:
    """
    Extrahiert den H5P Content-Typ aus einer H5P-ZIP-Datei.
    
    Args:
        h5p_zip_path: Pfad zur H5P-ZIP-Datei
        
    Returns:
        Content-Typ (z.B. "H5P.InteractiveVideo") oder None bei Fehler
    """
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            with zipfile.ZipFile(h5p_zip_path, "r") as zip_ref:
                # Extrahiere content.json
                zip_ref.extract("content/content.json", tmp_dir)
            
            content_json = Path(tmp_dir) / "content" / "content.json"
            with open(content_json, "r", encoding="utf-8") as f:
                content = json.load(f)
            
            # Versuche verschiedene Methoden, den Typ zu identifizieren
            
            # Methode 1: Top-Level Key (z.B. "interactiveVideo")
            # Diese Keys sind oft der Content-Typ
            top_keys = [k for k in content.keys() if not k.startswith("_")]
            if len(top_keys) == 1:
                return f"H5P.{top_keys[0]}"
            
            # Methode 2: Library-Field (wenn vorhanden)
            if "library" in content:
                return content["library"]
            
            # Methode 3: Mehrere Top-Keys → kombinierter Typ
            if len(top_keys) > 1:
                return f"H5P.Mixed[{','.join(top_keys)}]"
            
            # Fallback
            return "H5P.Unknown"
            
    except Exception as e:
        print(f"⚠️  Fehler beim Extrahieren des Content-Typs: {e}")
        return None


def analyze_h5p_content_types(course_ids: list[int], moodle: Moodle):
    """
    Analysiert alle H5P-Module in den angegebenen Kursen.
    
    Args:
        course_ids: Liste der Kurs-IDs
        moodle: Konfigurierte Moodle-Instanz
    """
    content_type_counter = Counter()
    total_h5p_modules = 0
    
    logger.info(f"\n🔍 Analysiere {len(course_ids)} Kurse...\n")
    
    for idx, course_id in enumerate(course_ids, 1):
        print(f"[{idx}/{len(course_ids)}] Kurs {course_id}...", end=" ", flush=True)
        
        try:
            # Hole alle Module des Kurses
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
            
            # Filtere H5P-Module
            h5p_count = 0
            h5p_module_ids = []
            
            for section in course_contents:
                for module in section.get("modules", []):
                    if module.get("modname") == "h5pactivity":
                        h5p_count += 1
                        total_h5p_modules += 1
                        h5p_module_ids.append(module["id"])
            
            # Wenn H5P-Module gefunden, hole ihre Details
            if h5p_module_ids:
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
                
                # Debug: Prüfe erste Activity-Struktur
                if idx == 1 and h5p_data.get("h5pactivities"):
                    first_activity = h5p_data["h5pactivities"][0]
                    logger.info(f"\n🔍 DEBUG - Erste Activity Struktur:")
                    logger.info(f"   Keys: {list(first_activity.keys())}")
                    if "package" in first_activity:
                        logger.info(f"   Package: {first_activity['package']}")
                    logger.info("")
                
                # Analysiere jedes H5P-Modul
                analyzed_count = 0
                skipped_no_file = 0
                
                for activity in h5p_data.get("h5pactivities", []):
                    # Die API gibt fileurl/filename verschachtelt in package[] zurück
                    if "package" not in activity or not activity["package"]:
                        skipped_no_file += 1
                        continue
                    
                    package = activity["package"][0]
                    if "fileurl" not in package or "filename" not in package:
                        skipped_no_file += 1
                        continue
                    
                    try:
                        # Download H5P ZIP
                        h5pfile_call = APICaller(
                            url=package["fileurl"],
                            params=moodle.download_params
                        )
                        
                        with tempfile.TemporaryDirectory() as tmp_dir:
                            local_filename = h5pfile_call.getFile(
                                package["filename"], tmp_dir
                            )
                            
                            # Extrahiere Content-Typ
                            content_type = extract_h5p_content_type(local_filename)
                            if content_type:
                                content_type_counter[content_type] += 1
                                analyzed_count += 1
                    except Exception as e:
                        logger.debug(f"Fehler bei Activity {activity.get('id', 'unknown')}: {e}")
                        continue
                
                status = f"✓ ({h5p_count} H5P-Module, {analyzed_count} analysiert"
                if skipped_no_file > 0:
                    status += f", {skipped_no_file} ohne Datei"
                status += ")"
                print(status)
            
        except Exception as e:
            print(f"✗ Fehler: {e}")
    
    print(f"\n{'='*70}")
    print(f"📊 ANALYSE ABGESCHLOSSEN")
    print(f"{'='*70}\n")
    print(f"Gesamt H5P-Module: {total_h5p_modules}")
    print(f"Verschiedene Content-Typen: {len(content_type_counter)}\n")
    
    # Erstelle sortierte Tabelle
    if content_type_counter:
        df = pd.DataFrame([
            {
                "Content-Typ": content_type,
                "Anzahl": count,
                "Prozent": f"{(count / total_h5p_modules * 100):.1f}%"
            }
            for content_type, count in content_type_counter.most_common()
        ])
        
        print(df.to_string(index=False))
        
        # Speichere Ergebnis als CSV
        output_path = Path("outputs") / "h5p_content_types_analysis.csv"
        output_path.parent.mkdir(exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\n💾 Ergebnis gespeichert: {output_path}")
    else:
        print("⚠️  Keine H5P-Module gefunden oder alle Analysen fehlgeschlagen.")


if __name__ == "__main__":
    # 1. Setup Moodle connection
    logger.info("🔐 Lade Credentials aus Azure Key Vault...")
    moodle = setup_production_moodle()
    logger.info("  ✓ Verbunden\n")
    
    # 2. Lade Live-Kurse aus Excel
    excel_path = Path(__file__).parent.parent.parent.parent / "documentation" / "Oct_Nov_KIC-course completion rate.xlsx"
    logger.info(f"📂 Lese Excel-Datei: {excel_path}")
    course_ids = load_live_courses(str(excel_path))
    
    # 3. Analysiere H5P Content-Typen
    analyze_h5p_content_types(course_ids, moodle)
