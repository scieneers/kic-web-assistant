import requests
from pydantic import BaseModel, field_validator
from src.helpers import get_secrets
from bs4 import BeautifulSoup
from llama_index import Document

ENVIRONMENT = 'STAGING'
if ENVIRONMENT == 'PRODUCTION':
    print("Danger Zone: You are using PRODUCTION environment")

def remove_html_tags(text:str) -> str:
    if '<' not in text:
        return text.strip()
    soup = BeautifulSoup(text, 'html.parser')
    return soup.get_text().strip()

class CourseTopic(BaseModel):
    id: int
    name: str
    summary: str|None
    
    @field_validator('summary')
    @classmethod
    def remove_html(cls, summary:str) -> str:
        if not summary:
            return None
        return remove_html_tags(summary)
    
    def __str__(self) -> str:
        text = f'Topic name: {self.name}'
        if self.summary is not None:
            text += f'\nTopic summary: {self.summary}'
        return text
    

class MoodleCourse(BaseModel):
    id: int
    shortname: str
    fullname: str
    displayname: str
    summary: str
    lang: str
    url: str
    topics: list[CourseTopic]=None

    @field_validator('summary')
    @classmethod
    def remove_html(cls, summary:str) -> str:
        if not summary:
            raise ValueError('Summary should not be empty')
        return remove_html_tags(summary)
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.url = f"{self.url}{self.id}"

    def __str__(self) -> str:
        topics = '\n'.join([str(topic) for topic in self.topics])
        text = f"""
        Course Summary: {self.summary}
        Course Topics: 
        {topics}"""
        return text

    def to_document(self) -> Document:
        text = str(self)
        metadata = {"id": self.id, 
                    "shortname": self.shortname,
                    "fullname": self.fullname, 
                    "type": "course",
                    "url": self.url}
        if self.lang:
            metadata['lang'] = self.lang
        return Document(text=text, metadata=metadata)
        

class Moodle():
    """class for Moodle API clients
        base_url: base url of moodle instance
        token: token for moodle api
    """
    def __init__(self, environment:str=ENVIRONMENT, base_url:str=None, token:str=None) -> None:
        if environment not in ['PRODUCTION', 'STAGING']:
            raise ValueError('Environment must be either PRODUCTION or STAGING.')
        
        secrets = get_secrets()
        self.base_url = secrets['DATA_SOURCES'][environment]['MOODLE_URL'] if base_url is None else base_url
        self.api_endpoint = f"{self.base_url}webservice/rest/server.php"
        self.token = secrets['DATA_SOURCES'][environment]['MOODLE_TOKEN'] if token is None else token
    
    def api_call(self, function:str, **kwargs) -> list[dict]:
        params = {
                "wstoken": self.token,
                "wsfunction": function,
                "moodlewsrestformat": "json",
            } 
        params.update(kwargs)
        response = requests.get(url=self.api_endpoint, params=params)
        response.raise_for_status()

        response = response.json()
        if 'exception' in response:
            raise Exception(f"{response['errorcode']}: {response['message']}")
        return response

    def get_courses(self) -> list[MoodleCourse]:
        '''get all courses that are set to visible on moodle'''
        courses = self.api_call('core_course_get_courses')
        course_url = self.base_url + 'course/view.php?id='
        courses = [MoodleCourse(url=course_url, **course) for course in courses if course['visible']]
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
        course_documents = [course.to_document() for course in courses]
        return course_documents

if __name__ == '__main__':
    moodle = Moodle(environment="PRODUCTION")
    courses = moodle.extract()
    print(courses)