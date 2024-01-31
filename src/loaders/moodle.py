import requests
from pydantic import BaseModel, field_validator
from env import EnvHelper
from bs4 import BeautifulSoup
from llama_index import Document
import re

def process_html_summaries(text:str) -> str:
    '''remove html tags from summaries, beautify poorly formatted texts'''
    if '<' not in text:
        new_text = text.strip()
    else:
        soup = BeautifulSoup(text, 'html.parser')
        for br_tag in soup.find_all('br'):
            br_tag.replace_with(' ')
        new_text = soup.get_text().strip()
    new_text = new_text.replace('\n', ' ').replace('\r', '')
    #\r = carriage return character 
    new_text = re.sub(r'\s{3,}', '  ', new_text)
    return new_text

class CourseTopic(BaseModel):
    id: int
    name: str
    summary: str|None
    
    @field_validator('summary')
    @classmethod
    def remove_html(cls, summary:str) -> str:
        if not summary:
            return None
        return process_html_summaries(summary)
    
    def __str__(self) -> str:
        return f'{self.name}: {self.summary}'
    
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
        return process_html_summaries(summary)
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.url = f'{self.url}{self.id}'

    def __str__(self) -> str:
        topics = '\n '.join([str(topic) for topic in self.topics])
        text = (f'Course Summary: {self.summary}\n'
                'Course Topics:\n'
                f'{topics}')
        return text

    def to_document(self) -> Document:
        text = str(self)
        metadata = {'id': self.id, 
                    'shortname': self.shortname,
                    'fullname': self.fullname, 
                    'type': 'course',
                    'url': self.url}
        if self.lang:
            metadata['lang'] = self.lang
        return Document(text=text, metadata=metadata)
        

class Moodle():
    '''class for Moodle API clients
       base_url: base url of moodle instance
       token: token for moodle api'''
    def __init__(self, base_url:str=None, token:str=None) -> None:
        secrets = EnvHelper()
        self.base_url = secrets.DATA_SOURCE_MOODLE_URL if base_url is None else base_url
        self.api_endpoint = f'{self.base_url}webservice/rest/server.php'
        self.token = secrets.DATA_SOURCE_MOODLE_TOKEN if token is None else token
    
    def api_call(self, function:str, **kwargs) -> list[dict]:
        params = {
                'wstoken': self.token,
                'wsfunction': function,
                'moodlewsrestformat': 'json',
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
        '''get all contents of a course'''
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
    moodle = Moodle()
    courses = moodle.extract()
    print(courses)