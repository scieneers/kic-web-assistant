from pathlib import Path
import subprocess
import json
import yaml

IAC_PATH = Path(__file__).parent.parent.parent / 'kic-web-assistant-iac' 

def get_secrets(iac_path:Path=IAC_PATH):
    '''sops_file_path: path to the locally stored sops file'''

    sops_file_path = iac_path/'secrets.enc.json'

    if not sops_file_path.exists():
        raise ValueError(f'Error: {sops_file_path} does not exist')

    try:
        result = subprocess.run(['sops', '--decrypt', sops_file_path], capture_output=True, text=True, check=True)
        decrypted_data = json.loads(result.stdout)
        return decrypted_data
    except subprocess.CalledProcessError as e:
        print(f'Error decrypting SOPS file: {e}')
        raise e

def get_vectordb_api_key(iac_path:Path=IAC_PATH) -> str:
    qdrant_values_path = iac_path/'charts'/'qdrant'/'values.yaml'

    if not qdrant_values_path.exists():
        raise ValueError(f'Error: {qdrant_values_path} does not exist')
    
    try: 
        result = subprocess.run(['sops', '--decrypt', qdrant_values_path], capture_output=True, text=True, check=True)
        decrypted_data = yaml.safe_load(result.stdout)
        api_key = decrypted_data['apiKey']
    except subprocess.CalledProcessError as e:
        print(f'Error decrypting SOPS file: {e}')
        raise e
    
    return api_key

if __name__ == '__main__':
    print(get_vectordb_api_key())