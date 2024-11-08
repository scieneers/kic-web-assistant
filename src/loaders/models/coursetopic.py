from pydantic import BaseModel, field_validator

from src.loaders.helper import process_html_summaries
from src.loaders.models.module import Module


class CourseTopic(BaseModel):
    """Representation of a course topic"""

    id: int
    name: str
    # TODO summary may contain video, file, etc. served from moodle server
    summary: str | None
    modules: list[Module | None]

    @field_validator("summary")
    @classmethod
    def remove_html(cls, summary: str) -> str:
        if not summary:
            return None
        return process_html_summaries(summary)

    def __str__(self) -> str:
        # modules = "\n ".join([str(module) for module in self.modules])
        text = f"Topic Summary: {self.summary}\n"  # "Topic Modules :\n" f"{modules}"
        return text
