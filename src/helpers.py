from pathlib import Path
import subprocess
import json

SOPS_FILE_PATH = Path(__file__).parent.parent.parent / 'kic-web-assistant-iac' / 'secrets.enc.json'

def get_secrets(sops_file_path:Path=SOPS_FILE_PATH):
    '''sops_file_path: path to the locally stored sops file'''

    if not sops_file_path.exists():
        raise ValueError(f'Error: {sops_file_path} does not exist')

    try:
        result = subprocess.run(['sops', '--decrypt', sops_file_path], capture_output=True, text=True, check=True)
        decrypted_data = json.loads(result.stdout)
        return decrypted_data
    except subprocess.CalledProcessError as e:
        print(f'Error decrypting SOPS file: {e}')
        raise e
        
if __name__ == '__main__':
    print(get_secrets())