
from src.loaders.moodle import Moodle, CourseTopic
import pytest
import json

def test_get_topics(course_id:int=16):
    # https://ki-campus-test.fernuni-hagen.de/course/view.php?id=16
    # course_contents = Moodle().get_course_contents(course_id)
    with open(f'src/tests/data/core_course_get_contents.json', 'r') as f:
        course_contents = json.load(f)

    courses = []
    for course_content in course_contents:
        course = CourseTopic(**course_content)
        courses.append(course)
    
    course_contents_dicts = [course.model_dump() for course in courses]
    with open(f'outputs/proccessed_course_contents.json', 'w') as f:
        json.dump(course_contents_dicts, f, indent=4, ensure_ascii=False)