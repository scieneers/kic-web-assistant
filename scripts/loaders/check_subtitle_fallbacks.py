"""
Analysiert Videos ohne Vimeo-Transkript auf Fallback-Optionen (VTT-Dateien, etc.)
"""

import json
import logging
import os
import sys
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

# Ensure src is in path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from loaders.APICaller import APICaller
from loaders.moodle import Moodle

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def setup_production_moodle():
    """Setup Moodle mit Production Credentials aus Key Vault."""
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
    logger.info("FALLBACK-ANALYSE: Videos ohne Transkript")
    logger.info("="*80 + "\n")
    
    # Setup Production Moodle
    logger.info("🔐 Lade Production Credentials aus Key Vault...")
    moodle = setup_production_moodle()
    logger.info(f"✓ Moodle verbunden: {moodle.base_url}\n")
    
    # Test mit Modul 20984 aus Kurs 235 - YouTube Interactive Video
    test_cases = [
        (235, 20984, "1. Motivation For Deep Learning"),
    ]
    
    logger.info("🎯 Teste Modul 20984 (YouTube Interactive Video)\n")
    
    results = []
    for course_id, module_id, activity_name in test_cases:
        
        logger.info(f"{'='*80}")
        logger.info(f"📹 Modul {module_id} aus Kurs {course_id}")
        logger.info(f"   Name: {activity_name[:70]}")
        
        try:
            # Hole ALLE Module des Kurses über core_course_get_contents
            caller = APICaller(
                url=moodle.api_endpoint,
                params={
                    **moodle.function_params,
                    "wsfunction": "core_course_get_contents",
                    "courseid": course_id
                }
            )
            caller.get()
            
            logger.info(f"   📡 API Response Status: {caller.response.status_code}")
            
            sections = caller.response.json()
            
            if isinstance(sections, dict) and "exception" in sections:
                logger.error(f"   ❌ API Fehler: {sections.get('message', 'Unknown')}")
                continue
            
            logger.info(f"   📚 Kurs hat {len(sections)} Sections")
            
            # Durchsuche alle Sections nach dem Modul
            module = None
            for section in sections:
                for mod in section.get("modules", []):
                    if mod.get("id") == module_id:
                        module = mod
                        break
                if module:
                    break
            
            if not module:
                logger.warning(f"   ⚠️  Modul {module_id} nicht im Kurs gefunden!")
                logger.info(f"   💡 Moodle URL: {moodle.base_url}")
                continue
            
            logger.info(f"   ✓ Modul gefunden")
            logger.info(f"   📌 Typ: {module.get('modname', 'unknown')}")
            
            # Wenn es h5pactivity ist, hole die Activity-Details
            if module.get("modname") != "h5pactivity":
                logger.warning(f"   ⚠️  Modul ist kein h5pactivity, sondern {module.get('modname')}")
                continue
            
            # Hole H5P Activity Details
            h5p_caller = APICaller(
                url=moodle.api_endpoint,
                params={
                    **moodle.function_params,
                    "wsfunction": "mod_h5pactivity_get_h5pactivities_by_courses",
                    "courseids[0]": course_id
                }
            )
            h5p_caller.get()
            activities = h5p_caller.response.json().get("h5pactivities", [])
            
            activity = None
            for act in activities:
                if act.get("coursemodule") == module_id:
                    activity = act
                    break
            
            if not activity:
                logger.warning(f"   ⚠️  H5P Activity nicht gefunden (möglicherweise keine Berechtigung)")
                continue
            
            logger.info(f"   ✓ H5P Activity gefunden")
            
            if "package" not in activity or not activity["package"]:
                logger.warning(f"   ⚠️  Kein Package vorhanden!")
                continue
            
            package = activity["package"][0]
            
            # Download und analysiere H5P ZIP
            with tempfile.TemporaryDirectory() as tmp_dir:
                h5p_call = APICaller(url=package["fileurl"], params=moodle.download_params)
                local_file = h5p_call.getFile(package["filename"], tmp_dir)
                
                logger.info(f"\n   📦 Analysiere H5P-Package...")
                
                # Suche Untertitel-Dateien
                subtitle_files = []
                with zipfile.ZipFile(local_file, "r") as zf:
                    for name in zf.namelist():
                        if any(ext in name.lower() for ext in [".vtt", ".srt", "subtitle", "caption"]):
                            subtitle_files.append(name)
                    
                    zf.extract("content/content.json", tmp_dir)
                
                if subtitle_files:
                    logger.info(f"   ✓ {len(subtitle_files)} Untertitel-Dateien gefunden:")
                    for sf in subtitle_files[:5]:
                        logger.info(f"      - {sf}")
                    if len(subtitle_files) > 5:
                        logger.info(f"      ... und {len(subtitle_files) - 5} weitere")
                else:
                    logger.info(f"   ✗ Keine Untertitel-Dateien im ZIP")
                
                # Analysiere content.json
                content_path = Path(tmp_dir) / "content" / "content.json"
                with open(content_path, "r", encoding="utf-8") as f:
                    content = json.load(f)
                
                # Video-Plattform
                video_url = None
                platform = "unknown"
                try:
                    video_url = content["interactiveVideo"]["video"]["files"][0]["path"]
                    
                    if "youtube" in video_url.lower():
                        platform = "YouTube"
                    elif "vimeo" in video_url.lower():
                        platform = "Vimeo"
                    elif "ki-campus" in video_url.lower() or "moodle" in video_url.lower():
                        platform = "Self-hosted"
                    else:
                        platform = "Other"
                    
                    logger.info(f"\n   🎬 Video-Plattform: {platform}")
                    logger.info(f"   🔗 URL: {video_url[:70]}...")
                except (KeyError, IndexError) as e:
                    logger.info(f"\n   ✗ Keine Video-URL gefunden")
                
                # TextTracks
                text_track_count = 0
                try:
                    tracks = content["interactiveVideo"]["video"]["textTracks"]["videoTrack"]
                    text_track_count = len(tracks)
                    logger.info(f"\n   📝 {len(tracks)} TextTrack-Einträge in content.json:")
                    for i, track in enumerate(tracks[:5]):
                        label = track.get("label", "N/A")
                        lang = track.get("srcLang", "N/A")
                        has_path = "track" in track and "path" in track.get("track", {})
                        path_val = track.get("track", {}).get("path", "N/A") if has_path else "N/A"
                        logger.info(f"      Track {i+1}: {label} ({lang}) - Path: {path_val}")
                    if len(tracks) > 5:
                        logger.info(f"      ... und {len(tracks) - 5} weitere")
                except (KeyError, IndexError):
                    logger.info(f"\n   ✗ Keine TextTrack-Einträge")
                
                results.append({
                    "course_id": course_id,
                    "module_id": module_id,
                    "platform": platform,
                    "subtitle_files": len(subtitle_files),
                    "text_tracks": text_track_count,
                    "has_fallback": len(subtitle_files) > 0 or text_track_count > 0
                })
                
                logger.info("")
                
        except Exception as e:
            logger.error(f"   ❌ Fehler: {e}")
            import traceback
            traceback.print_exc()
    
    # Zusammenfassung
    if results:
        logger.info("="*80)
        logger.info("📊 ZUSAMMENFASSUNG")
        logger.info("="*80)
        
        df_results = pd.DataFrame(results)
        
        logger.info(f"\n✓ Erfolgreich analysiert: {len(results)} Videos")
        logger.info(f"✓ Mit Fallback-Optionen: {df_results['has_fallback'].sum()}/{len(results)}")
        
        platform_counts = df_results["platform"].value_counts()
        logger.info(f"\n🎬 Video-Plattformen:")
        for platform, count in platform_counts.items():
            logger.info(f"   {platform}: {count}")
        
        logger.info(f"\n📄 Details:")
        logger.info(f"   Ø Untertitel-Dateien: {df_results['subtitle_files'].mean():.1f}")
        logger.info(f"   Ø TextTrack-Einträge: {df_results['text_tracks'].mean():.1f}")
    
    logger.info("\n" + "="*80)
    logger.info("✅ Analyse abgeschlossen")


if __name__ == "__main__":
    main()
