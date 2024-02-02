import requests
from pydantic import BaseModel, Field, field_validator 
from bs4 import BeautifulSoup
from pydantic import BaseModel, AliasPath, AliasChoices
from llama_index import Document
from src.env import EnvHelper

class CourseAttributes(BaseModel):
    name: str
    abstract: str
    # TODO tell cornelia where apis differ
    language: str|None = Field(alias='languages')
    url: str = Field(validation_alias=AliasChoices("url", 'uniformResourceLocator'))
    
    # moochup does not always provide a language, learn.ki-campus.org does
    @field_validator('abstract')
    @classmethod
    def remove_html_tags(cls, abstract: str) -> str:
        if '<' not in abstract:
            return abstract
        
        soup = BeautifulSoup(abstract, 'html.parser')
        return soup.get_text()
    
    @field_validator('language', mode='before')
    @classmethod
    def single_language(cls, languages: list[str|None]) -> str:
        if len(languages) > 1:
            raise ValueError('Only one language is expected.')
        elif len(languages) == 0:
            return None
        return languages[0]
    
class CourseInfo(BaseModel):
    id: str
    type: str
    attributes: CourseAttributes

    @field_validator('type')
    @classmethod
    def name_be_course(cls, type: str) -> str:
        if type != 'courses':
            raise ValueError('Only courses are expected.')
        return type
    
    def to_document(self) -> Document:
        text = f'Kursname: {self.attributes.name}\n Kursbeschreibung: {self.attributes.abstract}'
        metadata = {'type': 'course',
                    'url': self.attributes.url}
        if self.attributes.language:
            metadata['lang'] = self.attributes.language, 
        return Document(text=text, metadata=metadata)

class Moochup():
    '''Moochup is an api for course overviews.'''
    def __init__(self, api_url) -> None:
        self.api_url = api_url

    def fetch_data(self) -> list[CourseInfo]:
        '''Course information currently is available from two moochub endpoints. 
        One for for courses in moodle, one for courses in hpi.  
        Course infromation is distribuded over multiple pages.'''

        def fetch_pages(api_url) -> dict:
            response = requests.get(api_url, headers={'Accept': 'application/vnd.api+json; moochub-version=2.1, application/problem+json'})
            response.raise_for_status()
            return response.json()

        courses = []
        course_infos_page = fetch_pages(self.api_url)
        courses.extend(course_infos_page['data'])
        while 'next' in course_infos_page['links'] and course_infos_page['links']['next']:
            course_infos_page = fetch_pages(course_infos_page['links']['next'])
            courses.extend(course_infos_page['data'])

        courses = [CourseInfo(**course) for course in courses]
        return courses

    def get_course_documents(self) -> list[Document]:
        '''Returns a list of all course payloads.'''
        course_data = self.fetch_data()
        return [course.to_document() for course in course_data]


if __name__ == '__main__':
    moochup_client = Moochup(EnvHelper().DATA_SOURCE_MOOCHUP_MOODLE_URL)
    course_data = moochup_client.fetch_data()
    print(course_data)
    print(course_data[0].to_document())
