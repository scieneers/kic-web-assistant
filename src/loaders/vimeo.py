from io import StringIO
from typing import Optional

from src.env import EnvHelper
from src.loaders.APICaller import APICaller
from src.loaders.helper import convert_vtt_to_text
from src.loaders.models.texttrack import TextTrack


class Vimeo:
    def __init__(self) -> None:
        secrets = EnvHelper()
        self.api_endpoint = "https://api.vimeo.com/videos/"
        self.vimeo_token = secrets.VIMEO_PAT
        self.headers = {
            "Authorization": f"Bearer {self.vimeo_token}",
            "Accept": "application/vnd.vimeo.*+json;version=3.4",
        }

    def get_metadata(self, video_id: str) -> Optional[dict]:
        url = self.api_endpoint + video_id + "/texttracks"
        texttrack_caller = APICaller(url=url, headers=self.headers)
        response_json = texttrack_caller.getJSON()["data"]
        # TODO: always the first track in data list?
        return response_json[0] if response_json else None

    def get_transcript(self, video_id: str) -> TextTrack:
        texttrack_json = self.get_metadata(video_id)
        if texttrack_json:  # If Video has an transcript
            texttrack = TextTrack(**texttrack_json)

            transcript_caller = APICaller(url=texttrack.link, headers=self.headers)
            texttrack.transcript = convert_vtt_to_text(StringIO(transcript_caller.getText()))
        return texttrack
