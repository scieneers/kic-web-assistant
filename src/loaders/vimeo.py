import json
import re
from io import StringIO

import requests
import webvtt
from pydantic import BaseModel, HttpUrl

from src.env import EnvHelper


def convert_vtt_to_text(vtt_buffer: StringIO) -> str:
    vtt = webvtt.read_buffer(vtt_buffer)
    transcript = ""

    lines = []
    for line in vtt:
        # Strip the newlines from the end of the text.
        # Split the string if it has a newline in the middle
        # Add the lines to an array
        lines.extend(line.text.strip().splitlines())

    # Remove repeated lines
    previous = None
    for line in lines:
        if line == previous:
            continue
        transcript += " " + line
        previous = line

    return transcript


class TextTrack(BaseModel):
    """Vimeo TextTrack"""

    id: int
    display_language: str
    language: str
    link: HttpUrl | None = None


class Vimeo:
    """class for Vimeo API clients"""

    def __init__(self, environment: str = "STAGING") -> None:
        secrets = EnvHelper(production=True if environment == "PRODUCTION" else False)
        self.api_endpoint = "https://api.vimeo.com/videos/"
        self.vimeo_token = secrets.VIMEO_PAT

    def api_call(self, url: str):
        headers = {"Authorization": f"Bearer {self.vimeo_token}", "Accept": "application/vnd.vimeo.*+json;version=3.4"}
        response = requests.get(url=url, headers=headers)
        response.raise_for_status()
        return response

    def convertToJSON(self, response: requests.Response):
        response_json = response.json()
        if "exception" in response_json:
            raise Exception(f"{response_json['errorcode']}: {response_json['message']}")
        return response_json

    def get_videotime_content(self, cmid: int):
        # videotime_content = self.api_call('mod_videotime_get_videotime', cmid=id)
        videotime_content = self.patch_api_call()
        transcript = self.get_transcript(HttpUrl(videotime_content["vimeo_url"]))
        return transcript

    def extract_video_id(self, vimeo_url: HttpUrl):
        video_url_pattern = re.compile(r"\d+")
        links = re.findall(video_url_pattern, vimeo_url.path)
        return links[0]  # DEBUG: is it always the first url?

    def get_transcript(self, vimeo_url: HttpUrl) -> str:
        url = self.api_endpoint + "889859249" + "/texttracks"  # DEBUG: replace 1234 with extract_video_id
        response_texttrack = self.api_call(url)
        # DEBUG: always the first track in data list?
        current_texttrack = TextTrack(**self.convertToJSON(response_texttrack)["data"][0])
        response_vtt = self.api_call(current_texttrack.link)
        result = convert_vtt_to_text(StringIO(response_vtt.text))
        return result

    # DEBUG: remove as soon api_call 'mod_videotime_get_videotime' works
    def patch_api_call(self):
        with open("kic-web-assistant/src/loaders/test_mod_videotime_get_videotime.json", "r") as f:
            data = f.read()
        return json.loads(data)
