from pathlib import Path

import requests
from pydantic import HttpUrl

from src.loaders.helper import TMP_DIR


class APICaller:
    def __init__(self, url: HttpUrl, params: dict = {}, headers: dict = {}, **kwargs) -> None:
        self.url = url
        self.params = {}
        self.params.update(params)
        self.headers = {}
        self.headers.update(headers)
        self.params.update(kwargs)
        self.response = requests.Response()

    def get(self, **kwargs):
        self.response = requests.get(url=self.url, params=self.params, headers=self.headers)
        self.response.raise_for_status()

    def getJSON(self, **kwargs) -> dict:
        self.get(**kwargs)
        response_json = self.response.json()
        if "exception" in response_json:
            raise Exception(f"{response_json['errorcode']}: {response_json['message']}")
        return response_json

    def getText(self, **kwargs) -> str:
        self.get(**kwargs)
        return self.response.text

    def getBuffer(self, **kwargs) -> str:
        self.get(**kwargs)
        return self.response.text

    def getFile(self, filename, tmp_dir):
        local_filename = Path(f"{tmp_dir}/{filename}")
        with requests.get(self.url, params=self.params, stream=True) as r:
            r.raise_for_status()
            with open(local_filename, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    # If you have chunk encoded response uncomment if
                    # and set chunk_size parameter to None.
                    # if chunk:
                    f.write(chunk)
        return local_filename
