from llama_index import Document
from pydantic import BaseModel, field_validator

from src.loaders.helper import process_html_summaries
from src.loaders.models.coursetopic import CourseTopic


class MoodleCourse(BaseModel):
    """Highest level representation of a Moodle course. Has 1 to many topics (called modules in ki-campus frontend)."""

    id: int
    shortname: str
    fullname: str
    displayname: str
    summary: str | None = None
    lang: str
    url: str
    topics: list[CourseTopic] = None

    @field_validator("summary")
    @classmethod
    def remove_html(cls, summary: str) -> str:
        if not summary:
            return None
        return process_html_summaries(summary)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.url = f"{self.url}{self.id}"

    def __str__(self) -> str:
        topics = "\n ".join([str(topic) for topic in self.topics])
        text = f"Course Summary: {self.summary}\n" "Course Topics:\n" f"{topics}"
        return text

    def to_document(self) -> Document:
        text = str(self)
        metadata = {
            "id": self.id,
            "shortname": self.shortname,
            "fullname": self.fullname,
            "type": "course",
            "url": self.url,
        }
        if self.lang:
            metadata.update({"language": self.lang})
        return Document(text=text, metadata=metadata)
