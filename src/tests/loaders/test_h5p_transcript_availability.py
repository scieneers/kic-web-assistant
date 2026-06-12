"""
Analyse-Skript: Prüft Transkript-Verfügbarkeit bei H5P Interactive Videos.

Dieses Skript:
1. Lädt alle Live-Kurse aus der Excel-Datei
2. Findet alle H5P-Module mit interactiveVideo
3. Prüft, ob Transkripte verfügbar sind (VTT-Datei im ZIP oder Vimeo)
4. Erstellt Statistik: Wie viele Videos haben Transkripte?
"""

import json
import logging
import os
import re
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import requests
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

# Logging Setup
logging.basicConfig(level=logging.WARNING, format='%(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Konfiguration
EXCEL_PATH = Path("documentation/Oct_Nov_KIC-course completion rate.xlsx")
OUTPUT_CSV = Path("outputs/h5p_transcript_availability.csv")


def setup_production_credentials():
    """
    Holt Production-Credentials aus Azure Key Vault.
    """
    key_vault_name = os.environ.get("KEY_VAULT_NAME", "kicwa-keyvault-lab")
    key_vault_uri = f"https://{key_vault_name}.vault.azure.net/"
    
    credential = DefaultAzureCredential()
    secret_client = SecretClient(vault_url=key_vault_uri, credential=credential)
    
    logger.info("🔐 Lade Production-Credentials aus Key Vault...")
    prod_url = secret_client.get_secret("DATA-SOURCE-PRODUCTION-MOODLE-URL").value
    prod_token = secret_client.get_secret("DATA-SOURCE-PRODUCTION-MOODLE-TOKEN").value
    vimeo_pat = secret_client.get_secret("VIMEO-PAT").value
    
    return {
        "moodle_url": prod_url,
        "moodle_token": prod_token,
        "vimeo_pat": vimeo_pat
    }


class SimpleMoodleAPI:
    """Vereinfachter Moodle API Client ohne env.py Abhängigkeit."""
    
    def __init__(self, base_url: str, token: str):
        self.base_url = base_url
        self.api_endpoint = f"{base_url}webservice/rest/server.php"
        self.token = token
        
    def call_function(self, function_name: str, **params):
        """Ruft eine Moodle API-Funktion auf."""
        payload = {
            "wstoken": self.token,
            "wsfunction": function_name,
            "moodlewsrestformat": "json",
            **params
        }
        response = requests.get(self.api_endpoint, params=payload)
        response.raise_for_status()
        return response.json()
    
    def download_file(self, file_url: str, local_path: str):
        """Lädt eine Datei von Moodle herunter."""
        response = requests.get(file_url, params={"token": self.token})
        response.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(response.content)
        return local_path


class SimpleVimeoAPI:
    """Vereinfachter Vimeo API Client."""
    
    def __init__(self, pat: str):
        self.pat = pat
        self.headers = {
            "Authorization": f"Bearer {pat}",
            "Accept": "application/vnd.vimeo.*+json;version=3.4"
        }
    
    def get_metadata(self, video_id: str):
        """Prüft ob Transkripte verfügbar sind."""
        url = f"https://api.vimeo.com/videos/{video_id}/texttracks"
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get("data") and len(data["data"]) > 0:
                    return data["data"][0]  # Erstes verfügbare Texttrack
            return None
        except Exception:
            return None


def load_live_courses(excel_path: Path) -> list[int]:
    """Lädt Course-IDs aus Excel."""
    df = pd.read_excel(excel_path)
    course_ids = df["courseID"].dropna().astype(int).tolist()
    return course_ids


def check_transcript_availability(content: dict, vimeo_api: SimpleVimeoAPI, tmp_dir: str) -> dict:
    """
    Prüft, ob für ein Interactive Video Transkripte verfügbar sind.
    
    Returns:
        dict mit: has_vtt_file, has_vimeo_video, transcript_available
    """
    result = {
        "has_vtt_file": False,
        "has_vimeo_video": False,
        "vimeo_video_id": None,
        "transcript_available": False,
        "transcript_source": None
    }
    
    # 1. Prüfe VTT-Datei im H5P-ZIP
    try:
        vtt_path = content['interactiveVideo']['video']['textTracks']['videoTrack'][0]['track']['path']
        result["has_vtt_file"] = True
        result["transcript_available"] = True
        result["transcript_source"] = "vtt_file"
    except (KeyError, IndexError):
        pass
    
    # 2. Prüfe Vimeo-Video
    try:
        videourl = content["interactiveVideo"]["video"]["files"][0]["path"]
        # Prüfe ob es eine Vimeo-URL ist (vimeo.com oder player.vimeo.com)
        if "vimeo.com" in videourl:
            # Extrahiere Video ID - nimm erste Ziffernfolge im path
            # Funktioniert für: vimeo.com/123456, player.vimeo.com/video/123456, player.vimeo.com/external/123456.hd.mp4
            match = re.search(r'/(\d+)', videourl)
            if match:
                video_id = match.group(1)
                result["has_vimeo_video"] = True
                result["vimeo_video_id"] = video_id
                
                # Prüfe ob Vimeo Transkripte hat
                texttrack = vimeo_api.get_metadata(video_id)
                if texttrack is not None:
                    if not result["transcript_available"]:
                        result["transcript_available"] = True
                        result["transcript_source"] = "vimeo"
                    
    except (KeyError, IndexError):
        pass
    
    return result


def analyze_transcript_availability(course_ids: list[int], moodle: SimpleMoodleAPI, vimeo: SimpleVimeoAPI) -> pd.DataFrame:
    """
    Analysiert Transkript-Verfügbarkeit für alle H5P-Interactive-Videos.
    """
    results = []
    total_courses = len(course_ids)
    
    logger.info(f"🔍 Analysiere {total_courses} Kurse...\n")
    
    for idx, course_id in enumerate(course_ids, 1):
        logger.info(f"[{idx}/{total_courses}] Kurs {course_id}...")
        
        try:
            # Hole H5P-Activities für diesen Kurs
            h5p_data = moodle.call_function(
                "mod_h5pactivity_get_h5pactivities_by_courses",
                **{"courseids[0]": course_id}
            )
            
            activities = h5p_data.get("h5pactivities", [])
            
            for activity in activities:
                if "package" not in activity or not activity["package"]:
                    continue
                
                package = activity["package"][0]
                if "fileurl" not in package:
                    continue
                
                # Download H5P ZIP
                try:
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        local_filename = Path(tmp_dir) / package["filename"]
                        moodle.download_file(package["fileurl"], str(local_filename))
                        
                        # Extrahiere content.json
                        with zipfile.ZipFile(local_filename, "r") as zip_ref:
                            zip_ref.extract("content/content.json", tmp_dir)
                        
                        content_json_path = Path(tmp_dir) / "content" / "content.json"
                        with open(content_json_path, "r", encoding="utf-8") as f:
                            content = json.load(f)
                        
                        # Nur Interactive Videos interessieren uns
                        if "interactiveVideo" not in content:
                            continue
                        
                        # Prüfe Transkript-Verfügbarkeit
                        transcript_info = check_transcript_availability(
                            content, vimeo, tmp_dir
                        )
                        
                        results.append({
                            "course_id": course_id,
                            "module_id": activity.get("coursemodule"),
                            "activity_name": activity.get("name"),
                            "has_vtt_file": transcript_info["has_vtt_file"],
                            "has_vimeo_video": transcript_info["has_vimeo_video"],
                            "vimeo_video_id": transcript_info["vimeo_video_id"],
                            "transcript_available": transcript_info["transcript_available"],
                            "transcript_source": transcript_info["transcript_source"]
                        })
                        
                except Exception as e:
                    logger.debug(f"  Fehler bei Activity {activity.get('name')}: {e}")
                    continue
                    
        except Exception as e:
            logger.warning(f"  ⚠️  Fehler bei Kurs {course_id}: {e}")
            continue
    
    df = pd.DataFrame(results)
    return df


def print_statistics(df: pd.DataFrame):
    """Gibt Statistiken aus."""
    total = len(df)
    
    if total == 0:
        logger.info("❌ Keine Interactive Videos gefunden!")
        return
    
    with_transcript = df["transcript_available"].sum()
    without_transcript = total - with_transcript
    
    vtt_count = df["has_vtt_file"].sum()
    vimeo_count = (df["transcript_source"] == "vimeo").sum()
    vimeo_available_total = df["has_vimeo_video"].sum()
    
    logger.info("\n" + "=" * 80)
    logger.info("📊 STATISTIK: H5P INTERACTIVE VIDEO TRANSKRIPTE")
    logger.info("=" * 80)
    logger.info(f"Gesamt analysierte Videos: {total}")
    logger.info(f"")
    logger.info(f"✓ Mit Transkript:          {with_transcript} ({with_transcript/total*100:.1f}%)")
    logger.info(f"  - VTT-Datei im H5P:      {vtt_count}")
    logger.info(f"  - Vimeo-Transkript:      {vimeo_count} (ohne VTT-File)")
    logger.info(f"  - Vimeo-Videos gesamt:   {vimeo_available_total}")
    logger.info(f"")
    logger.info(f"✗ Ohne Transkript:         {without_transcript} ({without_transcript/total*100:.1f}%)")
    logger.info("=" * 80)
    
    # Module ohne Transkript auflisten
    if without_transcript > 0:
        logger.info(f"\n📋 {without_transcript} Module OHNE Transkript:")
        no_transcript = df[~df["transcript_available"]]
        for idx, row in enumerate(no_transcript.iterrows(), 1):
            _, row = row
            logger.info(f"  - Modul {row['module_id']}: {row['activity_name'][:60]}")
            if idx >= 20:
                logger.info(f"  ... und {len(no_transcript) - 20} weitere")
                break


def main():
    logger.info("=" * 80)
    logger.info("H5P TRANSKRIPT-VERFÜGBARKEIT ANALYSE")
    logger.info("=" * 80 + "\n")
    
    # Setup mit Key Vault Credentials
    creds = setup_production_credentials()
    moodle = SimpleMoodleAPI(creds["moodle_url"], creds["moodle_token"])
    vimeo = SimpleVimeoAPI(creds["vimeo_pat"])
    logger.info("✓ APIs verbunden\n")
    
    # Lade Kurse
    course_ids = load_live_courses(EXCEL_PATH)
    logger.info(f"✓ {len(course_ids)} Live-Kurse geladen\n")
    
    # Analysiere
    df = analyze_transcript_availability(course_ids, moodle, vimeo)
    
    # Speichere Ergebnisse
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    logger.info(f"\n💾 Ergebnisse gespeichert: {OUTPUT_CSV}")
    
    # Statistiken
    print_statistics(df)
    
    logger.info("\n✅ Analyse abgeschlossen!")


if __name__ == "__main__":
    main()
