import logging
from pathlib import Path

import requests
from pydantic import HttpUrl


class APICaller:
    def __init__(self, url: HttpUrl, params: dict = {}, headers: dict = {}, **kwargs) -> None:
        self.logger = logging.getLogger("loader")
        self.url = url
        self.params = {}
        self.params.update(params)
        self.headers = {}
        self.headers.update(headers)
        self.params.update(kwargs)
        self.response = requests.Response()
        # This fixes very slow requests, IPV6 is not properly supported by the ki-campus.org server
        # https://stackoverflow.com/questions/62599036/python-requests-is-slow-and-takes-very-long-to-complete-http-or-https-request
        requests.packages.urllib3.util.connection.HAS_IPV6 = False

    def get(self, **kwargs):
        self.response = requests.get(url=self.url, params=self.params, headers=self.headers)
        try:
            self.response.raise_for_status()
        except requests.exceptions.HTTPError as err:
            raise err

    def getJSON(self, **kwargs) -> dict:
        try:
            self.get(**kwargs)
        except requests.exceptions.HTTPError as err:
            self.logger.warn(f"Failed to retrieve {self.url}")
            raise err
        response_json = self.response.json()
        if "exception" in response_json:
            raise Exception(f"{response_json['errorcode']}: {response_json['message']}")
        return response_json

    def getText(self, **kwargs) -> str:
        try:
            self.get(**kwargs)
        except requests.exceptions.HTTPError as err:
            self.logger.warn(f"Failed to retrieve {self.url}")
            raise err
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
