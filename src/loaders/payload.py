from pydantic import BaseModel
from enum import Enum


class Types(str, Enum):
    course = "course"
    faq = "faq"

class Payload(BaseModel):
    type: Types
    vector_content: str
