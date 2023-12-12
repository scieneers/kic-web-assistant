import requests
from pydantic import BaseModel
from abc import abstractmethod
from typing import Dict, Iterator, List
from pydantic import BaseModel, field_validator
from src.helpers import get_secrets
from bs4 import BeautifulSoup

def remove_html_tags(text:str) -> str:
    if '<' not in text:
        return text
    soup = BeautifulSoup(text, 'html.parser')
    return soup.get_text()

class CourseTopic(BaseModel):
    id: int
    name: str
    summary: str
    @field_validator('summary')
    @classmethod
    def remove_html(cls, summary:str) -> str:
        return remove_html_tags(summary)

class MoodleCourse(BaseModel):
    id: int
    shortname: str
    fullname: str
    displayname: str
    summary: str
    topics: list[CourseTopic]=None

    @field_validator('summary')
    @classmethod
    def remove_html(cls, summary:str) -> str:
        return remove_html_tags(summary)
    
class Moodle():
    """class for Moodle API clients
        base_url: base url of moodle instance
        token: token for moodle api
    """
    def __init__(self, base_url:str=None, token:str=None) -> None:
        secrets = get_secrets()
        
        if base_url is None:
            moodle_url = secrets['DATA_SOURCES']['STAGING']['MOODLE_URL']
            self.base_url = f"https://{moodle_url}/webservice/rest/server.php"
        else:
            self.base_url = base_url

        if token is None:
            self.token = secrets['DATA_SOURCES']['STAGING']['MOODLE_TOKEN']
        else:
            self.token = token
    
    def api_call(self, function:str, **kwargs) -> list[dict]:
        params = {
                "wstoken": self.token,
                "wsfunction": function,
                "moodlewsrestformat": "json",
            } 
        params.update(kwargs)
        response = requests.get(url=self.base_url, params=params)
        return response.json()

    def get_courses(self) -> list[MoodleCourse]:
        '''get all courses that are set to visible on moodle'''
        courses = self.api_call('core_course_get_courses')
        courses = [MoodleCourse(**course) for course in courses if course['visible']]
        return courses
        
    def get_course_contents(self, course_id:int) -> list[CourseTopic]:
        """get all contents of a course"""
        course_contents = self.api_call('core_course_get_contents', courseid=course_id)
        course_contents = [CourseTopic(**topic) for topic in course_contents]
        return course_contents

    def extract(self) -> list[MoodleCourse]:
        courses = self.get_courses()
        for course in courses:
            course.topics = self.get_course_contents(course.id)
        return courses

if __name__ == '__main__':
    moodle = Moodle()
    courses = moodle.extract()
    print(courses)