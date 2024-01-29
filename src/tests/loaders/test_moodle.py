
from src.loaders.moodle import Moodle, CourseTopic
import pytest
import json

def test_get_topics(course_id:int=16):
    #course_contents = Moodle("STAGING").get_course_contents(course_id)
    with open(f'src/tests/data/core_course_get_contents.json', 'r') as f:
        course_contents = json.load(f)

    course_contents = [CourseTopic(**topic) for topic in course_contents]
    
    course_contents_dicts = [course_content.model_dump() for course_content in course_contents]

    with open(f'src/tests/data/proccessed_course_contents.json', 'w') as f:
        json.dump(course_contents_dicts, f, indent=4, ensure_ascii=False)