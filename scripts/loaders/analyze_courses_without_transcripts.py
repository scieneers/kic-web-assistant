"""
Analysiert welche Kurse Interactive Videos ohne Transkripte haben.
Basiert auf dem vorhandenen CSV aus der Transkript-Verfügbarkeitsanalyse.
Testet YouTube-Videos auf verfügbare Transkripte.
"""

import json
import logging
import os
import re
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
from loaders.youtube import Youtube

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

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


def extract_youtube_id(url: str) -> str | None:
    """Extrahiert YouTube Video ID aus verschiedenen URL-Formaten."""
    if not url:
        return None
    
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([a-zA-Z0-9_-]{11})',
        r'youtube\.com\/v\/([a-zA-Z0-9_-]{11})',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    return None


def test_youtube_transcripts(course_id: int, moodle: Moodle):
    """Testet YouTube-Videos eines Kurses auf verfügbare Transkripte."""
    logger.info(f"\n{'='*80}")
    logger.info(f"🎬 TESTE YOUTUBE-VIDEOS IN KURS {course_id}")
    logger.info(f"{'='*80}\n")
    
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
    sections = caller.response.json()
    
    # Finde alle h5pactivity Module
    h5p_modules = []
    for section in sections:
        for mod in section.get("modules", []):
            if mod.get("modname") == "h5pactivity":
                h5p_modules.append(mod)
    
    logger.info(f"📚 Gefunden: {len(h5p_modules)} H5P Activities im Kurs\n")
    
    if not h5p_modules:
        logger.warning("⚠️  Keine H5P Activities gefunden!")
        return []
    
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
    
    youtube = Youtube()
    results = []
    
    for idx, module in enumerate(h5p_modules[:10], 1):  # Teste max 10 für den Anfang
        module_id = module.get("id")
        module_name = module.get("name", "Unbekannt")
        
        logger.info(f"{idx}. Modul {module_id}: {module_name[:60]}")
        
        # Finde entsprechende Activity
        activity = None
        for act in activities:
            if act.get("coursemodule") == module_id:
                activity = act
                break
        
        if not activity or "package" not in activity or not activity["package"]:
            logger.info(f"   ⚠️  Kein Package verfügbar\n")
            continue
        
        package = activity["package"][0]
        
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                # Download H5P Package
                h5p_call = APICaller(url=package["fileurl"], params=moodle.download_params)
                local_file = h5p_call.getFile(package["filename"], tmp_dir)
                
                # Extrahiere content.json
                with zipfile.ZipFile(local_file, "r") as zf:
                    zf.extract("content/content.json", tmp_dir)
                
                content_path = Path(tmp_dir) / "content" / "content.json"
                with open(content_path, "r", encoding="utf-8") as f:
                    content = json.load(f)
                
                # Extrahiere Video URL
                try:
                    video_url = content["interactiveVideo"]["video"]["files"][0]["path"]
                    youtube_id = extract_youtube_id(video_url)
                    
                    if youtube_id:
                        logger.info(f"   🎥 YouTube ID: {youtube_id}")
                        
                        # Teste YouTube Transcript API
                        texttrack, err_message = youtube.get_transcript(youtube_id)
                        
                        if texttrack and texttrack.transcript:
                            transcript_length = len(texttrack.transcript)
                            logger.info(f"   ✅ Transkript gefunden! ({transcript_length} Zeichen)")
                            logger.info(f"      Sprache: {texttrack.display_language}")
                            logger.info(f"      Preview: {texttrack.transcript[:100]}...")
                            
                            results.append({
                                "course_id": course_id,
                                "module_id": module_id,
                                "module_name": module_name,
                                "youtube_id": youtube_id,
                                "has_transcript": True,
                                "transcript_length": transcript_length,
                                "language": texttrack.display_language,
                                "error": None
                            })
                        else:
                            logger.info(f"   ❌ Kein Transkript: {err_message}")
                            
                            results.append({
                                "course_id": course_id,
                                "module_id": module_id,
                                "module_name": module_name,
                                "youtube_id": youtube_id,
                                "has_transcript": False,
                                "transcript_length": 0,
                                "language": None,
                                "error": err_message
                            })
                    else:
                        logger.info(f"   ℹ️  Keine YouTube URL: {video_url[:60]}...")
                        
                except (KeyError, IndexError) as e:
                    logger.info(f"   ⚠️  Video-Struktur nicht gefunden: {e}")
                
                logger.info("")
                
        except Exception as e:
            logger.error(f"   ❌ Fehler: {e}\n")
            import traceback
            traceback.print_exc()
    
    return results


def main():
    logger.info("="*80)
    logger.info("ANALYSE: KURSE OHNE TRANSKRIPTE + YOUTUBE FALLBACK TEST")
    logger.info("="*80)
    logger.info("")
    
    # Lade das CSV aus der vorherigen Analyse
    csv_path = Path(__file__).parent.parent.parent.parent / "outputs" / "h5p_transcript_availability.csv"
    
    if not csv_path.exists():
        logger.error(f"❌ CSV nicht gefunden: {csv_path}")
        logger.info("Bitte zuerst test_h5p_transcript_availability.py ausführen!")
        return
    
    df = pd.read_csv(csv_path)
    
    if not csv_path.exists():
        print(f"❌ CSV nicht gefunden: {csv_path}")
        print("Bitte zuerst test_h5p_transcript_availability.py ausführen!")
        return
    
    df = pd.read_csv(csv_path)
    
    print("="*80)
    print("KURSE MIT INTERACTIVE VIDEOS OHNE TRANSKRIPTE")
    print("="*80)
    print()
    
    # Filter: Videos ohne Transkript
    no_transcript = df[df['transcript_available'] == False].copy()
    
    print(f"📊 Gesamt: {len(no_transcript)} Videos ohne Transkript")
    print(f"📊 Total: {len(df)} Videos insgesamt")
    print(f"📊 Prozentsatz ohne Transkript: {len(no_transcript)/len(df)*100:.1f}%\n")
    
    # Gruppiere nach Kurs
    course_groups = no_transcript.groupby('course_id').agg({
        'module_id': 'count',
        'activity_name': 'first',  # Verwende activity_name statt course_name
        'transcript_source': lambda x: ', '.join([str(s) for s in x.unique()])
    }).rename(columns={'module_id': 'video_count', 'activity_name': 'example_activity'})
    
    # Sortiere nach Anzahl Videos (absteigend)
    course_groups = course_groups.sort_values('video_count', ascending=False)
    
    print(f"🎓 Betroffene Kurse: {len(course_groups)}\n")
    print("="*80)
    
    for idx, (course_id, row) in enumerate(course_groups.iterrows(), 1):
        print(f"\n{idx}. Kurs {course_id}")
        print(f"   📹 {row['video_count']} Videos ohne Transkript")
        
        # Zeige Details für die ersten 3 Kurse
        if idx <= 5:
            course_videos = no_transcript[no_transcript['course_id'] == course_id]
            print(f"   📋 Beispiel-Module:")
            for _, video in course_videos.head(3).iterrows():
                vimeo_info = f" (Vimeo: {video['vimeo_video_id']})" if pd.notna(video.get('vimeo_video_id')) else ""
                print(f"      - Modul {video['module_id']}: {video['activity_name'][:50]}{vimeo_info}")
            if len(course_videos) > 3:
                print(f"      ... und {len(course_videos) - 3} weitere")
    
    print("\n" + "="*80)
    print("\n📈 STATISTIK NACH VIMEO-STATUS:")
    print("="*80)
    
    vimeo_stats = no_transcript.groupby('has_vimeo_video').agg({
        'module_id': 'count'
    }).rename(columns={'module_id': 'count'})
    
    for has_vimeo, row in vimeo_stats.iterrows():
        status = "Vimeo Videos" if has_vimeo else "Nicht-Vimeo Videos"
        print(f"\n{status}:")
        print(f"   📹 {row['count']} Videos ohne Transkript")
    
    print("\n" + "="*80)
    print("\n💡 EMPFEHLUNG:")
    print("="*80)
    
    # Top 5 Kurse nach Video-Anzahl
    top5 = course_groups.head(5)
    print(f"\nDie folgenden {len(top5)} Kurse haben die meisten Videos ohne Transkript:")
    print()
    
    for idx, (course_id, row) in enumerate(top5.iterrows(), 1):
        print(f"{idx}. Kurs {course_id} - {row['video_count']} Videos")
    
    print("\nFokussiere die Fallback-Implementierung auf diese Kurse für maximalen Impact.")
    
    # Export: Detaillierte Liste pro Kurs
    output_path = Path(__file__).parent.parent.parent.parent / "outputs" / "courses_without_transcripts_detail.csv"
    
    course_detail = no_transcript[['course_id', 'module_id', 'activity_name', 'has_vimeo_video', 'vimeo_video_id']].copy()
    course_detail = course_detail.sort_values(['course_id', 'module_id'])
    course_detail.to_csv(output_path, index=False)
    
    logger.info(f"\n📄 Details exportiert nach: {output_path}")
    
    # YOUTUBE FALLBACK TEST für Kurs 235
    logger.info("\n")
    logger.info("🔍 Teste YouTube-Fallback für Kurs 235...")
    logger.info("🔐 Lade Production Credentials...")
    
    moodle = setup_production_moodle()
    youtube_results = test_youtube_transcripts(235, moodle)
    
    if youtube_results:
        logger.info("\n" + "="*80)
        logger.info("📊 YOUTUBE FALLBACK ERGEBNISSE")
        logger.info("="*80)
        
        df_yt = pd.DataFrame(youtube_results)
        
        success_count = df_yt['has_transcript'].sum()
        total_count = len(df_yt)
        
        logger.info(f"\n✅ Erfolgsquote: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
        
        if success_count > 0:
            logger.info(f"\n🎉 ERFOLG: {success_count} YouTube-Videos haben Transkripte!")
            logger.info("   Diese können als Fallback extrahiert werden.")
        
        # Export YouTube-Ergebnisse
        yt_output = Path(__file__).parent.parent.parent.parent / "outputs" / "youtube_fallback_test_results.csv"
        df_yt.to_csv(yt_output, index=False)
        logger.info(f"\n📄 YouTube-Test-Ergebnisse: {yt_output}")


if __name__ == "__main__":
    main()
