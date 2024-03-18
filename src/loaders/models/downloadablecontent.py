from pydantic import BaseModel, field_validator


class DownloadableContent(BaseModel):
    filename: str | None = None
    fileurl: str | None = None
    type: str | None = None

    @field_validator("type", mode="before")
    def set_type(cls, type, values):
        fileurl = values.data.get("fileurl")

        file_extension = fileurl.split(".")[-1] if fileurl is not None else ""
        if file_extension not in ["html", "hp5"]:
            # raise ValueError(f'File extension {file_extension} is not supported.')
            return file_extension
