import pytest

from src.loaders.moochup import CourseAttributes, CourseInfo


@pytest.mark.parametrize(
    "input, expected",
    [
        (
            "<h2>KI-Lectures: Lernen und Bildung mit KI</h2>\n\n<p>eine Veranstaltungsreihe vom <em>AI.EDU Research Lab</em> des Forschungsschwerpunkts D²L² der FernUniversität in Hagen und dem Projekt <em>tech4comp</em> mit der Universität Leipzig und der TU Dresden, zusammen mit dem DFKI, die im Sommersemester 2021 auf dem KI-Campus stattfindet.</p>\n\n<p><strong>Inhalt</strong>  </p>\n\n",
            "KI-Lectures: Lernen und Bildung mit KI\neine Veranstaltungsreihe vom AI.EDU Research Lab des Forschungsschwerpunkts D²L² der FernUniversität in Hagen und dem Projekt tech4comp mit der Universität Leipzig und der TU Dresden, zusammen mit dem DFKI, die im Sommersemester 2021 auf dem KI-Campus stattfindet.\nInhalt \n",
        ),
        ("This has no valid <> html tags.", "This has no valid <> html tags."),
    ],
)
def test_moochup_course_attributes(input: str, expected: str):
    course_attributes = CourseAttributes(name="Test Title", description=input, languages=["de"])
    assert course_attributes.description == expected


def test_moochup_payload_creation():
    course_example = {
        "id": "0cf46a4b-9022-4cd5-9815-92312b3a84dcc",
        "type": "courses",
        "attributes": {
            "name": "KI und Ethik II",
            "courseCode": "KI_Ethik_II",
            "courseMode": "MOOC",
            "abstract": "Die Entwicklung Künstlicher Intelligenz erfolgt eingebettet in den größeren Kontext einer [...]",
            "languages": ["de"],
            "startDate": "2021-10-18T17:10:00Z",
            "endDate": None,
            "availableUntil": None,
            "image": {
                "url": "https://kicampus-public.s3.openhpicloud.de/courses/orBzqgERBJsLyvTtwv0Oo/visual_v1.png",
                "licenses": [
                    {
                        "id": "CC0-1.0",
                        "url": "https://creativecommons.org/publicdomain/zero/1.0",
                        "name": "Creative Commons Zero v1.0 Universal",
                        "author": None,
                    }
                ],
            },
            "video": None,
            "instructors": [],
            "learningObjectives": [],
            "duration": None,
            "workload": None,
            "partnerInstitute": [],
            "moocProvider": {
                "name": "KI-Campus",
                "url": "https://learn.ki-campus.org/",
                "logo": "https://imgproxy.services.openhpi.de/AcQdY-UlLcD3gY7NcMYcFeVk4-kbZWFH0r31JuyXdI4/fit/0/0/ce/false/aHR0cHM6Ly9sZWFy/bi5raS1jYW1wdXMu/b3JnL2Fzc2V0cy9s/b2dvLTQ3NTE5NjQz/NGEwNjlhNjY1M2Y4/ZDI0MmI2YTU5NDE2/MmEyYTU3MzkxZDky/ZjlkMTFlN2Y0NmMw/M2EyOWU3NWQucG5n.png",
            },
            "courseLicenses": [
                {
                    "id": "CC-BY-SA-4.0",
                    "url": "https://creativecommons.org/licenses/by-sa/4.0",
                    "name": "Creative Commons Attribution Share Alike 4.0 International",
                    "author": "KI-Campus",
                }
            ],
            "access": ["free"],
            "url": "https://ki-campus.org/courses/KI_Ethik_II",
        },
    }

    course = CourseInfo(**course_example)
    course.to_document()
