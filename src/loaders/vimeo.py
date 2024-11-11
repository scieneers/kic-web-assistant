import logging
from io import StringIO
from typing import Optional, Tuple

import requests

from src.env import env
from src.loaders.APICaller import APICaller
from src.loaders.helper import convert_vtt_to_text
from src.loaders.models.texttrack import TextTrack


class Vimeo:
    def __init__(self) -> None:
        self.logger = logging.getLogger("loader")
        self.api_endpoint = "https://api.vimeo.com/videos/"
        self.vimeo_token = env.VIMEO_PAT
        self.headers = {
            "Authorization": f"Bearer {self.vimeo_token}",
            "Accept": "application/vnd.vimeo.*+json;version=3.4",
        }

    def get_metadata(self, video_id: str) -> Tuple[Optional[dict], Optional[str]]:
        url = self.api_endpoint + video_id + "/texttracks"
        texttrack_caller = APICaller(url=url, headers=self.headers)
        err_message = None

        try:
            response_json = texttrack_caller.getJSON()["data"]
            de_index = None
            en_index = None
            de_autogen_index = None
            en_autogen_index = None

            # Loop through the data to find the required indices
            for index, item in enumerate(response_json):
                if item["language"].startswith("de"):
                    de_index = index
                elif item["language"].startswith("en"):
                    en_index = index
                elif item["language"] == "de-x-autogen":
                    de_autogen_index = index
                elif item["language"] == "en-x-autogen":
                    en_autogen_index = index

            # Select the appropriate index based on priority
            if de_index is not None:
                result_index = de_index
            elif en_index is not None:
                result_index = en_index
            elif de_autogen_index is not None:
                result_index = de_autogen_index
            elif en_autogen_index is not None:
                result_index = en_autogen_index
            else:
                result_index = None
                err_message = "Transcript liegt nicht (in Deutsch oder Englisch) vor"
        except requests.exceptions.HTTPError as err:
            if err.response.status_code == 404:
                return None, "Transcript konnte nicht abgerufen werden"
        return (response_json[result_index], None) if result_index is not None else (None, err_message)

    def get_transcript(self, video_id: str, fallback_transcript: str | None = None) -> Optional[TextTrack]:
        texttrack_json, err_message = self.get_metadata(video_id)
        if texttrack_json:  # If Video has an transcript
            texttrack = TextTrack(**texttrack_json)
            transcript_caller = APICaller(url=texttrack.link, headers=self.headers)
            try:
                transcript_text = transcript_caller.getText()
            except requests.exceptions.HTTPError as err:
                if err.response.status_code == 404:
                    # Transcript URL was present, but no transcript on Vimeo,
                    # fallback to transcript file stored in h5p package
                    if fallback_transcript is not None:
                        self.logger.warn("Falling back to reading file from H5P-Package")
                        transcript_text = self.get_transcript_from_file(fallback_transcript)
                    else:
                        return None, "Transcript liegt nicht (in Deutsch oder Englisch) vor"
            try:
                texttrack.transcript = convert_vtt_to_text(StringIO(transcript_text))
                return texttrack, None
            except Exception as err:
                self.logger.warn(f"Reading Fallback Transcript failed: {fallback_transcript}")
                self.logger.warn(f"Error while converting VTT to text: {err}")
                return None, "Transcript (VTT-Datei) nicht lesbar"
        return None, err_message

    # Fallback for retrieving transcript from h5p, if transcript on Vimeo is not available
    # Transcript is also stored in h5p package, but is often malformatted :(.
    def get_transcript_from_file(self, file_path: str) -> StringIO:
        with open(file_path, "r") as file:
            file_contents = file.read()
            return file_contents
