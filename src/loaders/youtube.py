import time
from io import StringIO
from typing import Optional

from youtube_transcript_api import NoTranscriptFound, YouTubeTranscriptApi
from youtube_transcript_api.formatters import WebVTTFormatter

from src.loaders.helper import convert_vtt_to_text
from src.loaders.models.texttrack import TextTrack


class Youtube:
    def get_transcript(self, video_id: str) -> Optional[TextTrack]:
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            transcript = transcript_list.find_manually_created_transcript(["de-DE", "de", "en"])
            transcript_json = transcript.fetch()
        except NoTranscriptFound:
            # No transcript found, probably because the transcript is not available in the given language
            return None
        formatter = WebVTTFormatter()
        transcript = convert_vtt_to_text(StringIO(formatter.format_transcript(transcript_json)))
        texttrack = TextTrack(
            id=0,
            display_language="de",
            language="de",
            link=None,
            transcript=transcript,
        )
        return texttrack
