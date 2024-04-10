from io import StringIO

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import WebVTTFormatter

from src.loaders.helper import convert_vtt_to_text
from src.loaders.models.texttrack import TextTrack


class Youtube:
    def get_transcript(self, video_id: str) -> TextTrack:
        transcript_json = YouTubeTranscriptApi.get_transcript(video_id, languages=["de-DE", "de", "en"])
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
