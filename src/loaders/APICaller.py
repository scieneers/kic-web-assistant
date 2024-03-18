import requests
from pydantic import HttpUrl


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

    def getBuffeer(self, **kwargs) -> str:
        self.get(**kwargs)
        return self.response.text
