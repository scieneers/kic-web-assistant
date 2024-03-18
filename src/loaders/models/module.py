from enum import StrEnum
from typing import Optional

from pydantic import BaseModel, HttpUrl, computed_field

from src.loaders.models.downloadablecontent import DownloadableContent
from src.loaders.models.videotime import VideoTime


class ModuleTypes(StrEnum):
    VIDEOTIME = "videotime"
    PAGE = "page"
    H5P = "h5pactivity"


class Module(BaseModel):
    """Lowest level content block of a course. Can be a file, video, hp5, etc."""

    id: int
    visible: int
    name: str
    url: HttpUrl | None = None
    modname: str  # content type
    contents: list[DownloadableContent] | None = None
    videotime: VideoTime | None = None

    @computed_field  # type: ignore[misc]
    @property
    def type(self) -> Optional[ModuleTypes]:
        match self.modname:
            case "videotime":
                return ModuleTypes.VIDEOTIME
            case "page":
                return ModuleTypes.PAGE
            case "h5pactivity":
                return ModuleTypes.H5P
            case _:
                return None
