
from src.loaders.moochup import CourseAttributes

def test_moochup_payload_creation():
    input_text = '<h2>KI-Lectures: Lernen und Bildung mit KI</h2>\n\n<p>eine Veranstaltungsreihe vom <em>AI.EDU Research Lab</em> des Forschungsschwerpunkts D²L² der FernUniversität in Hagen und dem Projekt <em>tech4comp</em> mit der Universität Leipzig und der TU Dresden, zusammen mit dem DFKI, die im Sommersemester 2021 auf dem KI-Campus stattfindet.</p>\n\n<p><strong>Inhalt</strong>  </p>\n\n'
    expected = 'KI-Lectures: Lernen und Bildung mit KI\neine Veranstaltungsreihe vom AI.EDU Research Lab des Forschungsschwerpunkts D²L² der FernUniversität in Hagen und dem Projekt tech4comp mit der Universität Leipzig und der TU Dresden, zusammen mit dem DFKI, die im Sommersemester 2021 auf dem KI-Campus stattfindet.\nInhalt \n'

    course_attributes = CourseAttributes(name="Test Title", abstract=input_text)
    assert course_attributes.abstract == expected
