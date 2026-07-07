"""
Analysiert die wichtigen Kurse aus IMPORTANT_COURSES.txt:
1. Welche Modultypen existieren
2. Welche H5P Activity-Typen gibt es  
3. Wie viele Interactive Videos haben Transkripte
"""

import json
import logging
import os
import re
import sys
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path

import pandas as pd
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

# Ensure src is in path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from loaders.APICaller import APICaller
from loaders.moodle import Moodle
from loaders.vimeo import Vimeo

# Logging mit Stdout-Handler für sichtbare Ausgabe
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

EXCEL_PATH = Path("documentation/Oct_Nov_KIC-course completion rate.xlsx")
IMPORTANT_COURSES_FILE = Path("documentation/IMPORTANT_COURSES.txt")


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


def load_course_ids_from_file(important_courses_file: Path, excel_path: Path):
    """Lädt Course-IDs direkt aus IMPORTANT_COURSES.txt und holt Namen aus Excel."""
    
    # Lade Course-IDs aus Textdatei (Format: [99, 106, 313, ...])
    with open(important_courses_file, "r", encoding="utf-8") as f:
        content = f.read().strip()
        # Parse die Liste: [99, 106, 313, ...]
        course_ids = eval(content)  # Sicher, da wir die Datei selbst erstellt haben
    
    # Lade Excel für Namen
    df = pd.read_excel(excel_path)
    
    # Mapping: Course ID -> Course Name
    course_mapping = []
    
    for course_id in course_ids:
        # Suche in Excel nach Course-ID
        match = df[df['courseID'] == course_id]
        
        if not match.empty:
            course_name = match.iloc[0]['Live Courses']
            course_mapping.append({
                "id": course_id,
                "name": course_name
            })
            logger.info(f"✓ Kurs [{course_id}]: {course_name}")
        else:
            print(f"⚠️  Course-ID {course_id} nicht in Excel gefunden")
            sys.stdout.flush()
            course_mapping.append({
                "id": course_id,
                "name": f"Unbekannter Kurs {course_id}"
            })
    
    return course_mapping


def get_course_id_by_name(moodle: Moodle, course_name: str):
    """Sucht Kurs-ID anhand des Namens (exakte Übereinstimmung)."""
    caller = APICaller(
        url=moodle.api_endpoint,
        params={
            **moodle.function_params,
            "wsfunction": "core_course_search_courses",
            "criterianame": "search",
            "criteriavalue": course_name
        }
    )
    caller.get()
    courses = caller.response.json().get("courses", [])
    
    if not courses:
        return None, None
    
    # Exakte Übereinstimmung
    for course in courses:
        if course.get("fullname", "").strip() == course_name.strip():
            return course.get("id"), course.get("fullname")
    
    # Fallback: Erster Treffer
    return courses[0].get("id"), courses[0].get("fullname")


def analyze_course_modules(moodle: Moodle, course_id: int, course_name: str):
    """Analysiert alle Module eines Kurses."""
    print(f"\n{'='*80}")
    print(f"📚 Kurs: {course_name}")
    print(f"    ID: {course_id}")
    print(f"{'='*80}")
    sys.stdout.flush()
    
    # Hole alle Sections und Module
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
    
    # Sammle Statistiken
    module_types = defaultdict(int)
    all_modules = []
    
    for section in sections:
        for module in section.get("modules", []):
            mod_type = module.get("modname", "unknown")
            module_types[mod_type] += 1
            all_modules.append(module)
    
    # 1. Modultypen ausgeben
    print(f"\n📊 MODULTYPEN ({len(all_modules)} Module insgesamt):")
    for mod_type, count in sorted(module_types.items(), key=lambda x: x[1], reverse=True):
        percentage = count/len(all_modules)*100 if len(all_modules) > 0 else 0
        print(f"   {mod_type}: {count} ({percentage:.1f}%)")
    sys.stdout.flush()
    
    # 2. H5P Activities analysieren
    h5p_count = module_types.get("h5pactivity", 0)
    h5p_results = None
    
    if h5p_count > 0:
        print(f"\n🎯 H5P ACTIVITIES: {h5p_count} gefunden")
        sys.stdout.flush()
        h5p_results = analyze_h5p_activities(moodle, course_id, all_modules)
    else:
        print(f"\n⚠️  Keine H5P Activities in diesem Kurs")
        sys.stdout.flush()
    
    return {
        "course_id": course_id,
        "course_name": course_name,
        "total_modules": len(all_modules),
        "module_types": dict(module_types),
        "h5p_count": h5p_count,
        "h5p_results": h5p_results
    }


def analyze_h5p_activities(moodle: Moodle, course_id: int, modules: list):
    """Analysiert H5P Activities eines Kurses."""
    
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
    
    if not activities:
        print("   ⚠️  Keine H5P Activity-Details abrufbar")
        sys.stdout.flush()
        return None
    
    # Zähle H5P Content-Typen
    h5p_types = defaultdict(int)
    interactive_videos = []
    
    vimeo = Vimeo()
    
    print(f"   🔍 Verarbeite {len(activities)} H5P Activities...")
    sys.stdout.flush()
    
    activities_with_package = 0
    activities_without_package = 0
    
    for activity in activities:
        if "package" not in activity or not activity["package"]:
            activities_without_package += 1
            h5p_types["Unknown"] += 1
            continue
        
        activities_with_package += 1
        package = activity["package"][0]
        module_id = activity.get("coursemodule")
        activity_name = activity.get("name", "Unknown")
        
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
                
                # Bestimme H5P-Typ
                library = content.get("library", "Unknown")
                h5p_type = library.split(" ")[0]
                h5p_types[h5p_type] += 1
                
                # Wenn Interactive Video, prüfe Transkript
                if "H5P.InteractiveVideo" in library:
                    has_transcript = False
                    transcript_source = None
                    
                    # 1. Prüfe VTT-Datei im Package
                    try:
                        vtt_path = content["interactiveVideo"]["video"]["textTracks"]["videoTrack"][0]["track"]["path"]
                        has_transcript = True
                        transcript_source = "vtt_file"
                    except (KeyError, IndexError):
                        pass
                    
                    # 2. Prüfe Vimeo
                    if not has_transcript:
                        try:
                            video_url = content["interactiveVideo"]["video"]["files"][0]["path"]
                            if "vimeo" in video_url.lower():
                                match = re.search(r'/(\d+)', video_url)
                                if match:
                                    video_id = match.group(1)
                                    texttrack, _ = vimeo.get_transcript(video_id, fallback_transcript=None)
                                    if texttrack and texttrack.transcript:
                                        has_transcript = True
                                        transcript_source = "vimeo"
                        except (KeyError, IndexError):
                            pass
                    
                    interactive_videos.append({
                        "module_id": module_id,
                        "name": activity_name,
                        "has_transcript": has_transcript,
                        "transcript_source": transcript_source
                    })
                    
        except Exception as e:
            print(f"   ⚠️  Fehler bei Activity {activity.get('id', 'unknown')}: {type(e).__name__}: {str(e)}")
            sys.stdout.flush()
            # Zähle als "Unknown" wenn Fehler auftritt
            h5p_types["Unknown"] += 1
    
    print(f"   ✓ Activities mit Package: {activities_with_package}")
    print(f"   ⚠️  Activities ohne Package: {activities_without_package}")
    sys.stdout.flush()
    
    # 2. H5P Content-Typen ausgeben
    print(f"\n   📋 H5P CONTENT-TYPEN:")
    for h5p_type, count in sorted(h5p_types.items(), key=lambda x: x[1], reverse=True):
        print(f"      {h5p_type}: {count}")
    sys.stdout.flush()
    
    # 3. Interactive Video Transkripte
    iv_with_transcript = 0
    iv_total = len(interactive_videos)
    
    if interactive_videos:
        iv_with_transcript = sum(1 for v in interactive_videos if v["has_transcript"])
        print(f"\n   📹 INTERACTIVE VIDEOS: {iv_total} insgesamt")
        print(f"      ✅ Mit Transkript: {iv_with_transcript} ({iv_with_transcript/iv_total*100:.1f}%)")
        print(f"      ❌ Ohne Transkript: {iv_total - iv_with_transcript}")
        sys.stdout.flush()
        
        if iv_with_transcript > 0:
            print(f"\n      Transkript-Quellen:")
            sources = defaultdict(int)
            for v in interactive_videos:
                if v["has_transcript"] and v["transcript_source"]:
                    sources[v["transcript_source"]] += 1
            for source, count in sources.items():
                print(f"         {source}: {count}")
            sys.stdout.flush()
    
    return {
        "h5p_types": dict(h5p_types),
        "interactive_videos": {
            "total": iv_total,
            "with_transcript": iv_with_transcript,
            "without_transcript": iv_total - iv_with_transcript
        }
    }


def main():
    print("="*80)
    print("ANALYSE: IMPORTANT COURSES")
    print("="*80)
    sys.stdout.flush()
    
    logger.info("="*80)
    logger.info("ANALYSE: IMPORTANT COURSES")
    logger.info("="*80)
    
    # Pfade
    excel_path = Path(__file__).parent.parent.parent.parent / EXCEL_PATH
    courses_file = Path(__file__).parent.parent.parent.parent / IMPORTANT_COURSES_FILE
    
    if not excel_path.exists():
        logger.error(f"❌ Excel nicht gefunden: {excel_path}")
        return
    
    if not courses_file.exists():
        logger.error(f"❌ {courses_file} nicht gefunden!")
        return
    
    # Lade Course-IDs aus Textdatei
    print(f"\n📚 Lade Kurs-IDs aus Datei...")
    sys.stdout.flush()
    logger.info(f"\n📚 Lade Kurs-IDs aus Datei...")
    course_mapping = load_course_ids_from_file(courses_file, excel_path)
    
    valid_courses = [c for c in course_mapping if c["id"] is not None]
    print(f"\n✓ {len(valid_courses)} Kurse geladen\n")
    sys.stdout.flush()
    logger.info(f"\n✓ {len(valid_courses)} Kurse geladen\n")
    
    # Setup Production Moodle
    print(f"\n🔐 Lade Production Credentials...")
    sys.stdout.flush()
    logger.info(f"\n🔐 Lade Production Credentials...")
    moodle = setup_production_moodle()
    print(f"✓ Verbunden mit: {moodle.base_url}\n")
    sys.stdout.flush()
    logger.info(f"✓ Verbunden mit: {moodle.base_url}")
    
    # Analysiere jeden Kurs
    results = []
    
    for course in valid_courses:
        try:
            print(f"\n{'='*80}")
            print(f"Analysiere Kurs [{course['id']}]: {course['name']}")
            print(f"{'='*80}")
            sys.stdout.flush()
            
            result = analyze_course_modules(moodle, course["id"], course["name"])
            results.append(result)
            
            print(f"✓ Kurs [{course['id']}] abgeschlossen\n")
            sys.stdout.flush()
            
        except Exception as e:
            logger.error(f"\n❌ Fehler bei '{course['name']}': {e}")
            import traceback
            traceback.print_exc()
    
    # Zusammenfassung
    print(f"\n{'='*80}")
    print("📊 ZUSAMMENFASSUNG ÜBER ALLE KURSE")
    print(f"{'='*80}")
    sys.stdout.flush()
    
    total_modules = sum(r["total_modules"] for r in results)
    total_h5p = sum(r["h5p_count"] for r in results)
    
    print(f"\n✅ Analysierte Kurse: {len(results)}/{len(valid_courses)}")
    print(f"📦 Module insgesamt: {total_modules}")
    print(f"🎯 H5P Activities: {total_h5p}")
    sys.stdout.flush()
    
    # Aggregiere Modultypen
    all_module_types = defaultdict(int)
    for result in results:
        for mod_type, count in result["module_types"].items():
            all_module_types[mod_type] += count
    
    print(f"\n📊 MODULTYPEN AGGREGIERT:")
    for mod_type, count in sorted(all_module_types.items(), key=lambda x: x[1], reverse=True):
        percentage = count/total_modules*100 if total_modules > 0 else 0
        print(f"   {mod_type}: {count} ({percentage:.1f}%)")
    sys.stdout.flush()
    
    # H5P Statistik
    if total_h5p > 0:
        all_h5p_types = defaultdict(int)
        total_iv = 0
        total_iv_with_transcript = 0
        
        for result in results:
            if result["h5p_results"]:
                for h5p_type, count in result["h5p_results"]["h5p_types"].items():
                    all_h5p_types[h5p_type] += count
                
                iv_data = result["h5p_results"]["interactive_videos"]
                total_iv += iv_data["total"]
                total_iv_with_transcript += iv_data["with_transcript"]
        
        print(f"\n🎯 H5P CONTENT-TYPEN AGGREGIERT:")
        for h5p_type, count in sorted(all_h5p_types.items(), key=lambda x: x[1], reverse=True):
            print(f"   {h5p_type}: {count}")
        sys.stdout.flush()
        
        if total_iv > 0:
            print(f"\n📹 INTERACTIVE VIDEOS GESAMT:")
            print(f"   Total: {total_iv}")
            print(f"   ✅ Mit Transkript: {total_iv_with_transcript} ({total_iv_with_transcript/total_iv*100:.1f}%)")
            print(f"   ❌ Ohne Transkript: {total_iv - total_iv_with_transcript} ({(total_iv - total_iv_with_transcript)/total_iv*100:.1f}%)")
            sys.stdout.flush()
    
    print("\n" + "="*80)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
