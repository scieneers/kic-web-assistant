from io import StringIO
from typing import Optional

import requests

from src.env import env
from src.loaders.APICaller import APICaller
from src.loaders.helper import convert_vtt_to_text
from src.loaders.models.texttrack import TextTrack


class Vimeo:
    def __init__(self) -> None:
        self.api_endpoint = "https://api.vimeo.com/videos/"
        self.vimeo_token = env.VIMEO_PAT
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

    def get_transcript(self, fallback_transcript: str, video_id: str) -> Optional[TextTrack]:
        texttrack_json = self.get_metadata(video_id)
        if texttrack_json:  # If Video has an transcript
            texttrack = TextTrack(**texttrack_json)
            transcript_caller = APICaller(url=texttrack.link, headers=self.headers)
            try:
                transcript_text = transcript_caller.getText()
            except requests.exceptions.HTTPError as err:
                if err.response.status_code == 404:
                    # Transcript URL was present, but no transcript on Vimeo,
                    # fallback to transcript file stored in h5p package
                    print("Falling back to reading file from H5P-Package")
                    transcript_text = self.get_transcript_from_file(fallback_transcript, video_id)
            try:
                texttrack.transcript = convert_vtt_to_text(StringIO(transcript_text))
                return texttrack
            except Exception as err:
                print(f"Reading Fallback Transcript failed: {fallback_transcript}")
                print(f"Error while converting VTT to text: {err}")
        return None

    # Fallback for retrieving transcript from h5p, if transcript on Vimeo is not available
    # Transcript is also stored in h5p package, but is often malformatted :(.
    def get_transcript_from_file(self, file_path: str, video_id: str) -> StringIO:
        with open(file_path, "r") as file:
            file_contents = file.read()
            return file_contents
