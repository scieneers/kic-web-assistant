from typing import Optional

import requests
from bs4 import BeautifulSoup
from llama_index.core import Document
from pydantic import AliasChoices, BaseModel, Field, field_validator


class CourseAttributes(BaseModel):
    name: str
    description: Optional[str]
    # TODO tell cornelia where apis differ
    languages: str | None = Field(
        alias=AliasChoices("inLanguage", "languages")
    )  # inLanguage"  Field(validation_alias=AliasChoices('first_name', 'fname'))
    url: str = Field(validation_alias=AliasChoices("url", "uniformResourceLocator"))

    # moochup does not always provide a language, learn.ki-campus.org does
    @field_validator("description")
    @classmethod
    def remove_html_tags(cls, description: str) -> str:
        if "<" not in description:
            return description

        soup = BeautifulSoup(description, "html.parser")
        return soup.get_text()

    @field_validator("languages", mode="before")
    @classmethod
    def single_language(cls, languages: list[str]) -> str:
        if len(languages) > 1:
            raise ValueError("Only one language is expected.")
        elif len(languages) == 0:
            return None
        return languages[0]


class CourseInfo(BaseModel):
    id: str
    type: str
    attributes: CourseAttributes

    @field_validator("type")
    @classmethod
    def name_be_course(cls, type: str) -> str:
        if type.lower() == "courses" or type == "Course":
            type = "course"

        # print(type.lower())
        if type != "course":
            raise ValueError("Only type 'course' is expected.")
        return type.lower()

    def to_document(self) -> Document:
        text = f"Kursname: {self.attributes.name}\n Kursbeschreibung: {self.attributes.description}"
        metadata = {"source": "Moochup", "type": "Kurs", "url": self.attributes.url, "course_id": self.id}
        if self.attributes.languages:
            metadata["lang"] = (self.attributes.languages,)
        return Document(text=text, metadata=metadata)


class Moochup:
    """Moochup is an api for course overviews."""

    def __init__(self, api_url) -> None:
        self.api_url = api_url

    def fetch_data(self) -> list[CourseInfo]:
        """Course information currently is available from two moochub endpoints.
        One for for courses in moodle, one for courses in hpi.
        Course infromation is distribuded over multiple pages."""

        def fetch_pages(api_url) -> dict:
            response = requests.get(
                api_url, headers={"Accept": "application/vnd.api+json; moochub-version=3.0, application/problem+json"}
            )
            response.raise_for_status()
            return response.json()

        courses = []
        course_infos_page = fetch_pages(self.api_url)
        courses.extend(course_infos_page["data"])
        while "next" in course_infos_page["links"] and course_infos_page["links"]["next"]:
            course_infos_page = fetch_pages(course_infos_page["links"]["next"])
            courses.extend(course_infos_page["data"])

        courses = [CourseInfo(**course) for course in courses]
        return courses

    def get_course_documents(self) -> list[Document]:
        """Returns a list of all course payloads."""
        course_data = self.fetch_data()
        return [course.to_document() for course in course_data]
