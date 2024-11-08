from llama_index.core import Document
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
        text = f"Course Title: {self.fullname}\n Course Summary: {self.summary}\n"
        if self.topics:
            topics = "\n ".join([str(topic) for topic in self.topics])
            text += "Course Topics:\n" f"{topics}"
        return text

    def to_document(self) -> Document:
        text = str(self)
        metadata = {
            "course_id": self.id,
            "shortname": self.shortname,
            "fullname": self.fullname,
            "type": "Kurs",
            "source": "Moodle",
            "url": self.url,
        }
        if self.lang:
            metadata.update({"language": self.lang})

        course_document = Document(text=text, metadata=metadata)
        docs = [course_document]

        if self.topics:
            module_documents = []
            for topic in self.topics:
                for module in topic.modules:
                    module_documents.append(module.to_document(self.id))
            docs += module_documents

        return docs
