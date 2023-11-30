from pydantic import BaseModel, ConfigDict
from enum import Enum


class Types(Enum):
    course = 'course'
    faq = 'faq'

class Payload(BaseModel):
    type: Types
    vector_content: str

    model_config = ConfigDict(use_enum_values=True)
