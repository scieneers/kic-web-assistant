"""
Test script to validate HTML, Audio, and ZIP extraction in Resource modules.
"""

import logging
import os
import sys
from pathlib import Path

from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.loaders.moodle import Moodle

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_production_moodle():
    """Setup Moodle with Production Credentials from Azure Key Vault."""
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
    
    logger.info(f"✅ Connected to Production Moodle: {prod_url}")
    return moodle


def test_extraction():
    """Test HTML, Audio, and ZIP extraction."""
    
    print("\n" + "="*80)
    print("TEST: HTML, AUDIO, ZIP EXTRACTION")
    print("="*80 + "\n")
    
    moodle = setup_production_moodle()
    
    # Test cases from analysis
    test_cases = [
        {"course_id": 127, "module_id": 10759, "type": "MP3"},  # mp3_module_10759_course_127.json
        {"course_id": 191, "module_id": 22060, "type": "ZIP"},  # zip_module_22060_course_191.json
        {"course_id": 350, "module_id": 29185, "type": "HTML"},  # html_module_29185_course_350.json
    ]
    
    for test_case in test_cases:
        course_id = test_case["course_id"]
        module_id = test_case["module_id"]
        file_type = test_case["type"]
        
        print(f"\n{'='*80}")
        print(f"Testing {file_type}: Course {course_id}, Module {module_id}")
        print(f"{'='*80}\n")
        
        try:
            # Get course contents
            topics = moodle.get_course_contents(course_id)
            
            # Find module
            target_module = None
            for topic in topics:
                for module in topic.modules:
                    if module.id == module_id:
                        target_module = module
                        break
                if target_module:
                    break
            
            if not target_module:
                print(f"❌ Module {module_id} nicht gefunden!")
                continue
            
            print(f"✓ Modul gefunden: {target_module.name}")
            print(f"  Typ: {target_module.modname}")
            
            if target_module.contents:
                file_info = target_module.contents[0]
                print(f"  Datei: {file_info.filename}")
                print(f"  Type: {file_info.type}")
            
            # Extract
            print(f"\nStarte Extraktion...")
            err = moodle.extract_resource(target_module)
            
            if err:
                print(f"❌ Fehler: {err}")
            elif target_module.resource:
                resource = target_module.resource
                print(f"✓ Extraktion erfolgreich!")
                print(f"\n  Filename: {resource.filename}")
                print(f"  Type: {resource.mimetype}")
                print(f"  PDF: {resource.is_pdf}")
                print(f"  HTML: {resource.is_html}")
                print(f"  ZIP: {resource.is_zip}")
                print(f"  Audio: {resource.is_audio}")
                
                if resource.extracted_text:
                    preview = resource.extracted_text[:500]
                    print(f"\n  Extrahierter Text (erste 500 Zeichen):")
                    print(f"  {'-'*76}")
                    print(f"  {preview}")
                    print(f"  {'-'*76}")
                    print(f"\n  Gesamtlänge: {len(resource.extracted_text)} Zeichen")
                else:
                    print(f"\n  ⚠️  Kein Text extrahiert")
                
                # Test Document generation
                print(f"\n  Teste Document-Generierung...")
                doc = target_module.to_document(course_id)
                print(f"  ✓ Document erstellt: {len(doc.text)} Zeichen")
                
                if "--- Dokument" in doc.text or "ZIP-Archiv" in doc.text or "Audio-Datei" in doc.text:
                    print(f"  ✓ Resource-Inhalt in Document enthalten")
                else:
                    print(f"  ⚠️  Resource-Inhalt NICHT in Document gefunden!")
            else:
                print(f"⚠️  module.resource ist None")
        
        except Exception as e:
            print(f"❌ Fehler: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*80}")
    print("TESTS ABGESCHLOSSEN")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    test_extraction()
