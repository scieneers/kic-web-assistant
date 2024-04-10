import re
from enum import StrEnum

from pydantic import BaseModel, Field, HttpUrl, computed_field


class VideoPlatforms(StrEnum):
    VIMEO = "vimeo"
    YOUTUBE = "youtube"
    SELF_HOSTED = "ki-campus"


class Video(BaseModel):
    id: int
    video_url: HttpUrl = Field(..., alias="vimeo_url")  # This Field can contain a Vimeo _or_ a Youtube URL

    @computed_field  # type: ignore[misc]
    @property
    def type(self) -> VideoPlatforms:
        match self.video_url.host:
            case "vimeo.com" | "player.vimeo.com":
                return VideoPlatforms.VIMEO
            case "www.youtube.com" | "youtu.be":
                return VideoPlatforms.YOUTUBE
            case "ki-campus-test.fernuni-hagen.de" | "moodle.ki-campus.org":
                return VideoPlatforms.SELF_HOSTED
            case _:
                raise NotImplementedError("Unknown VideoPlatform, implement me")

    @computed_field  # type: ignore[misc]
    @property
    def video_id(self) -> str:
        match self.type:
            case VideoPlatforms.VIMEO:
                vimeo_video_id_pattern = r"\d+"
                return re.findall(vimeo_video_id_pattern, str(self.video_url.path))[0]
            case VideoPlatforms.YOUTUBE:
                youtube_video_id_pattern = (
                    r"(?:youtube\.com\/(?:[^\/]+\/.+\/|(?:v|e(?:mbed)?)\/|.*[?&]v=)|youtu\.be\/)([^\"&?\/\s]{11})"
                )
                return re.findall(youtube_video_id_pattern, str(self.video_url))[0]
            case VideoPlatforms.SELF_HOSTED:
                return ""
