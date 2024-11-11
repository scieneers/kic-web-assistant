import time
from io import StringIO
from typing import Optional, Tuple
from xml.etree.ElementTree import ParseError

from retry import retry
from youtube_transcript_api import (
    NoTranscriptFound,
    TranscriptsDisabled,
    YouTubeTranscriptApi,
)
from youtube_transcript_api.formatters import WebVTTFormatter

from src.loaders.helper import convert_vtt_to_text
from src.loaders.models.texttrack import TextTrack


class Youtube:
    @retry(ParseError, tries=3, delay=2)
    def get_transcript(self, video_id: str) -> Tuple[Optional[TextTrack], str]:
        err_message = None

        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            transcript = transcript_list.find_manually_created_transcript(["de-DE", "de", "en"])
            transcript_json = transcript.fetch()
            # Mitigation of API rate limit? :(
            time.sleep(3)
        except (NoTranscriptFound, TranscriptsDisabled):
            # No transcript found, probably because the transcript is not available in the given language
            # or because the video is not available in the given language
            err_message = "Transcript liegt nicht (in Deutsch oder Englisch) vor"
            return None, err_message
        formatter = WebVTTFormatter()
        transcript = convert_vtt_to_text(StringIO(formatter.format_transcript(transcript_json)))
        texttrack = TextTrack(
            id=0,
            display_language="de",
            language="de",
            link=None,
            transcript=transcript,
        )
        return texttrack, err_message
