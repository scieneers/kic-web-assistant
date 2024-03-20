from enum import StrEnum
from typing import Optional

from pydantic import BaseModel, HttpUrl, computed_field


class VideoPlatforms(StrEnum):
    VIMEO = "vimeo"
    YOUTUBE = "youtube"
    SELF_HOSTED = "ki-campus"


class Transcript(BaseModel):
    video_url: HttpUrl
    type: VideoPlatforms
    transcript: str

    @computed_field  # type: ignore[misc]
    @property
    def language(self) -> Optional[str]:
        # TODO: implement language detection
        pass
