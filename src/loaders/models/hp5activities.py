from pathlib import Path

from pydantic import BaseModel, HttpUrl, root_validator


class H5PActivities(BaseModel):
    id: int
    coursemodule: int
    fileurl: HttpUrl
    filename: Path

    @root_validator(pre=True)
    def validate_fileurl(cls, values):
        values["fileurl"] = values["package"][0]["fileurl"]
        values["filename"] = values["package"][0]["filename"]
        return values
