"""
Debug-Skript: Testet Vimeo API für konkrete Video-IDs.

Prüft detailliert:
1. API-Authentifizierung
2. Texttracks-Endpoint Antwort
3. Verfügbare Sprachen
4. Download-Verfügbarkeit
"""

import json
import logging
import os

import requests
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Test Video-IDs (aus der CSV)
TEST_VIDEO_IDS = [
    "867856392",  # Modul 2195 - Introduction to Machine Learning
    "968600243",  # Modul 5550 - Vorstellung der Phase
    "558964357",  # Modul 9220 - Hat laut CSV VTT-File (zum Vergleich)
]


def get_vimeo_token():
    """Hole Vimeo PAT aus Key Vault."""
    key_vault_name = os.environ.get("KEY_VAULT_NAME", "kicwa-keyvault-lab")
    key_vault_uri = f"https://{key_vault_name}.vault.azure.net/"
    credential = DefaultAzureCredential()
    secret_client = SecretClient(vault_url=key_vault_uri, credential=credential)
    
    try:
        token = secret_client.get_secret("VIMEO-PAT").value
        logger.info("✓ Vimeo Token aus Key Vault geladen")
        return token
    except Exception as e:
        logger.error(f"✗ Fehler beim Laden des Vimeo Tokens: {e}")
        return None


def test_vimeo_api(video_id: str, token: str):
    """
    Testet Vimeo API für eine Video-ID.
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"TEST: Video ID {video_id}")
    logger.info(f"{'='*80}")
    
    # 1. Test: Video-Metadaten abrufen
    logger.info("\n1️⃣  Teste Video-Metadaten...")
    video_url = f"https://api.vimeo.com/videos/{video_id}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.vimeo.*+json;version=3.4",
    }
    
    try:
        response = requests.get(video_url, headers=headers)
        response.raise_for_status()
        video_data = response.json()
        logger.info(f"   ✓ Video gefunden: {video_data.get('name', 'N/A')}")
        logger.info(f"   - Dauer: {video_data.get('duration', 'N/A')}s")
        logger.info(f"   - Erstellt: {video_data.get('created_time', 'N/A')}")
    except requests.exceptions.HTTPError as e:
        logger.error(f"   ✗ HTTP-Fehler: {e.response.status_code} - {e.response.text}")
        return
    except Exception as e:
        logger.error(f"   ✗ Fehler: {e}")
        return
    
    # 2. Test: Texttracks abrufen
    logger.info("\n2️⃣  Teste Texttracks-Endpoint...")
    texttracks_url = f"https://api.vimeo.com/videos/{video_id}/texttracks"
    
    try:
        response = requests.get(texttracks_url, headers=headers)
        response.raise_for_status()
        texttracks_data = response.json()
        
        if "data" in texttracks_data and len(texttracks_data["data"]) > 0:
            logger.info(f"   ✓ {len(texttracks_data['data'])} Texttrack(s) gefunden:")
            for idx, track in enumerate(texttracks_data["data"]):
                logger.info(f"     [{idx}] {track.get('language', 'N/A')} - "
                           f"{track.get('name', 'N/A')} - "
                           f"Type: {track.get('type', 'N/A')}")
                logger.info(f"         Link: {track.get('link', 'N/A')}")
            
            # 3. Test: Ersten Track herunterladen
            logger.info("\n3️⃣  Teste Download des ersten Tracks...")
            first_track = texttracks_data["data"][0]
            track_url = first_track.get("link")
            
            if track_url:
                try:
                    track_response = requests.get(track_url, headers=headers)
                    track_response.raise_for_status()
                    content = track_response.text
                    logger.info(f"   ✓ Track heruntergeladen ({len(content)} Zeichen)")
                    logger.info(f"   Vorschau (erste 200 Zeichen):")
                    logger.info(f"   {content[:200]}")
                except requests.exceptions.HTTPError as e:
                    logger.error(f"   ✗ Download fehlgeschlagen: {e.response.status_code}")
                except Exception as e:
                    logger.error(f"   ✗ Fehler beim Download: {e}")
            else:
                logger.warning("   ⚠️  Kein Link zum Track gefunden")
                
        else:
            logger.warning("   ⚠️  Keine Texttracks gefunden")
            logger.info(f"   API-Antwort: {json.dumps(texttracks_data, indent=2)}")
            
    except requests.exceptions.HTTPError as e:
        logger.error(f"   ✗ HTTP-Fehler: {e.response.status_code}")
        logger.error(f"   Response: {e.response.text}")
    except Exception as e:
        logger.error(f"   ✗ Fehler: {e}")


def main():
    logger.info("="*80)
    logger.info("VIMEO API DEBUG")
    logger.info("="*80)
    
    # Token holen
    token = get_vimeo_token()
    if not token:
        logger.error("❌ Kein Vimeo Token verfügbar!")
        return
    
    # Test Videos
    for video_id in TEST_VIDEO_IDS:
        test_vimeo_api(video_id, token)
    
    logger.info("\n" + "="*80)
    logger.info("✅ Debug abgeschlossen")
    logger.info("="*80)


if __name__ == "__main__":
    main()
