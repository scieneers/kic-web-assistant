from io import StringIO

from pydantic import HttpUrl

from src.env import EnvHelper
from src.loaders.APICaller import APICaller
from src.loaders.helper import convert_vtt_to_text


class Vimeo:
    def __init__(self) -> None:
        secrets = EnvHelper()
        self.api_endpoint = "https://api.vimeo.com/videos/"
        self.vimeo_token = secrets.VIMEO_PAT
        self.headers = {
            "Authorization": f"Bearer {self.vimeo_token}",
            "Accept": "application/vnd.vimeo.*+json;version=3.4",
        }

    def get_texttrack(self, video_id: int) -> dict:
        url = self.api_endpoint + video_id + "/texttracks"
        texttrack_caller = APICaller(url=url, headers=self.headers)
        response_json = texttrack_caller.getJSON()["data"]
        # TODO: always the first track in data list?
        return response_json[0] if response_json else None

    def get_transcript(self, transcript_url: HttpUrl) -> str:
        transcript_caller = APICaller(url=transcript_url, headers=self.headers)
        return convert_vtt_to_text(StringIO(transcript_caller.getText()))
