from pydantic import BaseModel, HttpUrl


class TextTrack(BaseModel):
    """Vimeo TextTrack"""

    id: int
    display_language: str
    language: str
    link: HttpUrl | None = None
    transcript: str | None = None

    def __str__(self) -> str:
        return self.transcript
