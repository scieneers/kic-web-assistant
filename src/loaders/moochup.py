import requests
from pydantic import BaseModel, field_validator
from src.loaders.payload import Payload, Types
from bs4 import BeautifulSoup


class CourseAttributes(BaseModel):
    name: str
    abstract: str
    
    @field_validator('abstract')
    @classmethod
    def remove_html_tags(cls, abstract: str) -> str:
        if '<' not in abstract:
            return abstract
        
        soup = BeautifulSoup(abstract, 'html.parser')
        return soup.get_text()

class CourseInfo(BaseModel):
    type: str
    attributes: CourseAttributes

    @field_validator('type')
    @classmethod
    def name_be_course(cls, type: str) -> str:
        if type != 'courses':
            raise ValueError('Only courses are expected.')
        return type


def fetch_data(api_url:str='https://learn.ki-campus.org/bridges/moochub/courses') -> list[CourseInfo]:
    '''Course information currently is available from two moochub endpoints (currently only one available). All courses are distribuded over multiple pages.'''

    def fetch_pages(api_url) -> dict:
        response = requests.get(api_url, headers={'Accept': 'application/vnd.api+json; moochub-version=2.1, application/problem+json'})
        response.raise_for_status()
        return response.json()

    courses = []
    course_infos_page = fetch_pages(api_url)
    courses.extend(course_infos_page['data'])
    while 'next' in course_infos_page['links']:
        course_infos_page = fetch_pages(course_infos_page['links']['next'])
        courses.extend(course_infos_page['data'])

    courses = [CourseInfo(**course) for course in courses]
    return courses

def create_payload(course_info:CourseInfo) -> Payload:
    '''Extracts relevant information from course_info and returns a document dictionary.'''
    content = f'Kursname: {course_info.attributes.name}\n Kursbeschreibung: {course_info.attributes.abstract}'
    return Payload(type=Types.course, vector_content=content)

def get_course_payloads() -> list[Payload]:
    '''Returns a list of all course payloads.'''
    course_data = fetch_data()
    return [create_payload(course) for course in course_data]

if __name__ == '__main__':
    course_data = fetch_data()
    print(course_data)

    print(create_payload(course_data[0]))
