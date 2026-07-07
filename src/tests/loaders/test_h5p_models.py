"""Unit tests for H5P activity data models.

Tests the core extraction logic (from_h5p_params) and text serialization (to_text)
for the most commonly used H5P content types, using minimal fixture dicts — no API calls.

Covered types:
- QuizQuestion (MultiChoice, SingleChoiceSet)
- TrueFalseQuestion
- FillInBlanksQuestion
- Summary
"""

import pytest

from src.loaders.models.h5pactivities.h5p_quiz_questions import QuizQuestion, TrueFalseQuestion
from src.loaders.models.h5pactivities.h5p_blanks import FillInBlanksQuestion
from src.loaders.models.h5pactivities.h5p_summary import Summary


class TestQuizQuestionMultiChoice:
    LIBRARY = "H5P.MultiChoice"

    def _params(self, question="Was ist ML?", answers=None):
        if answers is None:
            answers = [
                {"text": "Ein KI-Teilgebiet", "correct": True},
                {"text": "Eine Programmiersprache", "correct": False},
                {"text": "Ein Betriebssystem", "correct": False},
            ]
        return {"question": question, "answers": answers}

    def test_from_h5p_params_returns_instance(self):
        result = QuizQuestion.from_h5p_params(self.LIBRARY, self._params())
        assert result is not None
        assert isinstance(result, QuizQuestion)

    def test_extracts_question_text(self):
        result = QuizQuestion.from_h5p_params(self.LIBRARY, self._params(question="Definiere Overfitting!"))
        assert result.question == "Definiere Overfitting!"

    def test_extracts_correct_answers(self):
        result = QuizQuestion.from_h5p_params(self.LIBRARY, self._params())
        assert "Ein KI-Teilgebiet" in result.correct_answers
        assert len(result.correct_answers) == 1

    def test_extracts_incorrect_answers(self):
        result = QuizQuestion.from_h5p_params(self.LIBRARY, self._params())
        assert "Eine Programmiersprache" in result.incorrect_answers
        assert "Ein Betriebssystem" in result.incorrect_answers

    def test_to_text_contains_question(self):
        q = QuizQuestion.from_h5p_params(self.LIBRARY, self._params())
        text = q.to_text()
        assert "Was ist ML?" in text

    def test_to_text_contains_correct_answer(self):
        q = QuizQuestion.from_h5p_params(self.LIBRARY, self._params())
        text = q.to_text()
        assert "Ein KI-Teilgebiet" in text
        assert "Korrekte Antwort" in text

    def test_to_text_contains_incorrect_answers(self):
        q = QuizQuestion.from_h5p_params(self.LIBRARY, self._params())
        text = q.to_text()
        assert "Inkorrekte Antwort" in text

    def test_to_text_has_quiz_prefix(self):
        q = QuizQuestion.from_h5p_params(self.LIBRARY, self._params())
        assert q.to_text().startswith("[Quiz]")

    def test_returns_none_when_no_correct_answers(self):
        params = {"question": "Test?", "answers": [{"text": "Antwort", "correct": False}]}
        result = QuizQuestion.from_h5p_params(self.LIBRARY, params)
        assert result is None

    def test_returns_none_when_empty_question(self):
        params = {"question": "", "answers": [{"text": "Antwort", "correct": True}]}
        result = QuizQuestion.from_h5p_params(self.LIBRARY, params)
        assert result is None


class TestQuizQuestionSingleChoiceSet:
    LIBRARY = "H5P.SingleChoiceSet"

    def _params(self):
        return {
            "choices": [
                {
                    "question": "Was ist Overfitting?",
                    "answers": ["Überanpassung an Trainingsdaten", "Zu wenig Training", "Gutes Modell"],
                }
            ]
        }

    def test_from_h5p_params_returns_instance(self):
        result = QuizQuestion.from_h5p_params(self.LIBRARY, self._params())
        assert result is not None

    def test_first_answer_is_correct(self):
        result = QuizQuestion.from_h5p_params(self.LIBRARY, self._params())
        assert "Überanpassung an Trainingsdaten" in result.correct_answers

    def test_to_text_non_empty(self):
        q = QuizQuestion.from_h5p_params(self.LIBRARY, self._params())
        assert len(q.to_text()) > 0

    def test_returns_none_on_empty_choices(self):
        result = QuizQuestion.from_h5p_params(self.LIBRARY, {"choices": []})
        assert result is None


class TestTrueFalseQuestion:
    LIBRARY = "H5P.TrueFalse"

    def test_from_h5p_params_true_answer(self):
        params = {"question": "KI ist dasselbe wie ML?", "correct": "true"}
        result = TrueFalseQuestion.from_h5p_params(self.LIBRARY, params)
        assert result is not None
        assert result.correct_answer is True

    def test_from_h5p_params_false_answer(self):
        params = {"question": "Ist KI dasselbe wie ML?", "correct": "false"}
        result = TrueFalseQuestion.from_h5p_params(self.LIBRARY, params)
        assert result is not None
        assert result.correct_answer is False

    def test_to_text_starts_with_marker(self):
        params = {"question": "Test Frage?", "correct": "true"}
        q = TrueFalseQuestion.from_h5p_params(self.LIBRARY, params)
        assert q.to_text().startswith("[Wahr/Falsch]")

    def test_to_text_contains_wahr_for_true(self):
        params = {"question": "Test?", "correct": "true"}
        q = TrueFalseQuestion.from_h5p_params(self.LIBRARY, params)
        assert "Wahr" in q.to_text()

    def test_to_text_contains_falsch_for_false(self):
        params = {"question": "Test?", "correct": "false"}
        q = TrueFalseQuestion.from_h5p_params(self.LIBRARY, params)
        assert "Falsch" in q.to_text()

    def test_returns_none_when_question_missing(self):
        params = {"question": "", "correct": "true"}
        result = TrueFalseQuestion.from_h5p_params(self.LIBRARY, params)
        assert result is None

    def test_returns_none_when_correct_invalid(self):
        params = {"question": "Test?", "correct": "maybe"}
        result = TrueFalseQuestion.from_h5p_params(self.LIBRARY, params)
        assert result is None


class TestFillInBlanksQuestion:
    LIBRARY = "H5P.Blanks"

    def _params(self):
        return {
            "text": "Erkläre den Begriff:",
            "questions": ["Machine *Learning* ist ein Teilgebiet der **KI**."],
        }

    def test_from_h5p_params_returns_instance(self):
        result = FillInBlanksQuestion.from_h5p_params(self.LIBRARY, self._params())
        assert result is not None
        assert isinstance(result, FillInBlanksQuestion)

    def test_extracts_intro_text_as_question(self):
        result = FillInBlanksQuestion.from_h5p_params(self.LIBRARY, self._params())
        assert result.question == "Erkläre den Begriff:"

    def test_extracts_text_with_blanks(self):
        result = FillInBlanksQuestion.from_h5p_params(self.LIBRARY, self._params())
        assert "Machine" in result.text_with_blanks

    def test_to_text_starts_with_marker(self):
        q = FillInBlanksQuestion.from_h5p_params(self.LIBRARY, self._params())
        assert q.to_text().startswith("[Lückentext]")

    def test_to_text_contains_question(self):
        q = FillInBlanksQuestion.from_h5p_params(self.LIBRARY, self._params())
        assert "Erkläre den Begriff:" in q.to_text()

    def test_returns_none_on_empty_questions(self):
        params = {"text": "Einleitung", "questions": []}
        result = FillInBlanksQuestion.from_h5p_params(self.LIBRARY, params)
        assert result is None

    def test_fallback_question_when_no_intro(self):
        params = {"text": "", "questions": ["*Lückentext* hier."]}
        result = FillInBlanksQuestion.from_h5p_params(self.LIBRARY, params)
        assert result is not None
        assert result.question == "Lückentext"


class TestSummary:
    LIBRARY = "H5P.Summary"

    def _params(self):
        return {
            "intro": "Wähle die richtige Aussage:",
            "summaries": [
                {
                    "summary": [
                        "Machine Learning ist ein Teilgebiet der KI.",
                        "ML hat nichts mit Daten zu tun.",
                        "ML ist eine Programmiersprache.",
                    ]
                }
            ],
        }

    def test_from_h5p_params_returns_instance(self):
        result = Summary.from_h5p_params(self.LIBRARY, self._params())
        assert result is not None
        assert isinstance(result, Summary)

    def test_extracts_intro(self):
        result = Summary.from_h5p_params(self.LIBRARY, self._params())
        assert result.intro == "Wähle die richtige Aussage:"

    def test_extracts_statement_groups(self):
        result = Summary.from_h5p_params(self.LIBRARY, self._params())
        assert len(result.statement_groups) == 1
        assert len(result.statement_groups[0]) == 3

    def test_to_text_contains_intro(self):
        s = Summary.from_h5p_params(self.LIBRARY, self._params())
        text = s.to_text()
        assert "Wähle die richtige Aussage:" in text

    def test_to_text_marks_correct_statement(self):
        s = Summary.from_h5p_params(self.LIBRARY, self._params())
        text = s.to_text()
        assert "Korrekt:" in text
        assert "Machine Learning ist ein Teilgebiet der KI." in text

    def test_to_text_marks_incorrect_statements(self):
        s = Summary.from_h5p_params(self.LIBRARY, self._params())
        text = s.to_text()
        assert "Falsch:" in text

    def test_returns_none_on_empty_summaries(self):
        params = {"intro": "Test", "summaries": []}
        result = Summary.from_h5p_params(self.LIBRARY, params)
        assert result is None

    def test_from_h5p_summary_data(self):
        summary_data = {
            "task": {
                "params": {
                    "intro": "Frage:",
                    "summaries": [{"summary": ["Richtig.", "Falsch."]}],
                }
            }
        }
        result = Summary.from_h5p_summary_data(summary_data)
        assert result is not None
        assert result.intro == "Frage:"

    def test_from_h5p_summary_data_returns_none_without_task(self):
        result = Summary.from_h5p_summary_data({})
        assert result is None
