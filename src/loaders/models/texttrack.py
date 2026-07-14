from pydantic import BaseModel, HttpUrl


class TextTrack(BaseModel):
    """Video text track (Vimeo/YouTube) with flat transcript and timed segments."""

    id: int
    display_language: str
    language: str
    link: HttpUrl | None = None
    transcript: str | None = None
    # Timed cue segments [{"start_seconds": float, "text": str}] — preserved so
    # Transcript documents can carry per-chunk start times (citations never
    # link to the video itself, only mention the timestamp in text).
    segments: list[dict] | None = None
    # Watch-page URL of the video — used only as a stable identity anchor for
    # the Transcript document's source_doc_key (see Module.to_item_documents),
    # never as a citation link target.
    video_url: str | None = None

    def __str__(self) -> str:
        return self.transcript
