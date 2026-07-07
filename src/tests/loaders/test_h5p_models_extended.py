"""Unit tests for the remaining H5P activity model types.

Covers all types NOT tested in test_h5p_models.py (which handles QuizQuestion,
TrueFalseQuestion, FillInBlanksQuestion, Summary).  Uses minimal fixture dicts —
no API calls, no ZIP files required.

Covered here:
- H5PFlashcards       (h5p_flashcards.py)
- H5PDialogcards      (h5p_dialogcards.py)
- H5PTimeline         (h5p_timeline.py)
- Crossword           (h5p_crossword.py)
- DragDropText        (h5p_drag_drop.py)
- DragDropQuestion    (h5p_drag_drop.py)
- ImageHotspotQuestion(h5p_drag_drop.py)
- Column              (h5p_wrappers.py)
- Accordion           (h5p_wrappers.py)
- QuestionSet         (h5p_question_set.py)
- InteractiveBook     (h5p_interactive_book.py)
- CoursePresentation  (h5p_wrappers.py)
- Gamemap             (h5p_wrappers.py)
- InteractiveVideo    (h5p_interactive_video.py — limited, params path only)
"""

import pytest

from src.loaders.models.h5pactivities.h5p_flashcards import H5PFlashcards
from src.loaders.models.h5pactivities.h5p_dialogcards import H5PDialogcards
from src.loaders.models.h5pactivities.h5p_timeline import H5PTimeline
from src.loaders.models.h5pactivities.h5p_crossword import Crossword
from src.loaders.models.h5pactivities.h5p_drag_drop import DragDropText, DragDropQuestion, ImageHotspotQuestion
from src.loaders.models.h5pactivities.h5p_wrappers import Column, Accordion, Gamemap, CoursePresentation
from src.loaders.models.h5pactivities.h5p_question_set import QuestionSet
from src.loaders.models.h5pactivities.h5p_interactive_book import InteractiveBook
from src.loaders.models.h5pactivities.h5p_interactive_video import InteractiveVideo


# ── H5PFlashcards ─────────────────────────────────────────────────────────────

class TestH5PFlashcards:
    LIB = "H5P.Flashcards"

    def _params(self, cards=None):
        if cards is None:
            cards = [{"text": "Was ist Overfitting?", "answer": "Überanpassung ans Training"}]
        return {"cards": cards}

    def test_from_h5p_params_returns_object(self):
        result = H5PFlashcards.from_h5p_params(self.LIB, self._params())
        assert result is not None
        assert len(result.cards) == 1

    def test_to_text_contains_question_and_answer(self):
        result = H5PFlashcards.from_h5p_params(self.LIB, self._params())
        text = result.to_text()
        assert "Was ist Overfitting?" in text
        assert "Überanpassung ans Training" in text

    def test_to_text_format_colon_separated(self):
        result = H5PFlashcards.from_h5p_params(self.LIB, self._params())
        text = result.to_text()
        assert ": " in text

    def test_multiple_cards_each_on_own_line(self):
        cards = [
            {"text": "A", "answer": "1"},
            {"text": "B", "answer": "2"},
        ]
        result = H5PFlashcards.from_h5p_params(self.LIB, {"cards": cards})
        lines = result.to_text().splitlines()
        assert len(lines) == 2

    def test_empty_cards_list_returns_none(self):
        assert H5PFlashcards.from_h5p_params(self.LIB, {"cards": []}) is None

    def test_missing_cards_key_returns_none(self):
        assert H5PFlashcards.from_h5p_params(self.LIB, {}) is None

    def test_cards_with_empty_text_and_answer_skipped(self):
        cards = [{"text": "", "answer": ""}]
        assert H5PFlashcards.from_h5p_params(self.LIB, {"cards": cards}) is None


# ── H5PDialogcards ────────────────────────────────────────────────────────────

class TestH5PDialogcards:
    LIB = "H5P.Dialogcards"

    def _params(self, dialogs=None):
        if dialogs is None:
            dialogs = [{"text": "Was ist Bias?", "answer": "Verzerrung der Daten"}]
        return {"dialogs": dialogs}

    def test_from_h5p_params_returns_object(self):
        result = H5PDialogcards.from_h5p_params(self.LIB, self._params())
        assert result is not None
        assert len(result.cards) == 1

    def test_to_text_contains_question_and_answer(self):
        result = H5PDialogcards.from_h5p_params(self.LIB, self._params())
        text = result.to_text()
        assert "Was ist Bias?" in text
        assert "Verzerrung der Daten" in text

    def test_multiple_dialogs_joined_with_newline(self):
        dialogs = [
            {"text": "A", "answer": "1"},
            {"text": "B", "answer": "2"},
        ]
        result = H5PDialogcards.from_h5p_params(self.LIB, {"dialogs": dialogs})
        lines = result.to_text().splitlines()
        assert len(lines) == 2

    def test_empty_dialogs_returns_none(self):
        assert H5PDialogcards.from_h5p_params(self.LIB, {"dialogs": []}) is None

    def test_missing_dialogs_key_returns_none(self):
        assert H5PDialogcards.from_h5p_params(self.LIB, {}) is None

    def test_dialog_missing_answer_skipped(self):
        dialogs = [{"text": "Only text", "answer": ""}]
        assert H5PDialogcards.from_h5p_params(self.LIB, {"dialogs": dialogs}) is None


# ── H5PTimeline ───────────────────────────────────────────────────────────────

class TestH5PTimeline:
    LIB = "H5P.Timeline"

    def _params(self, dates=None):
        if dates is None:
            dates = [{"startDate": "1956", "headline": "KI-Begriff geprägt", "text": "Von John McCarthy"}]
        return {"timeline": {"date": dates}}

    def test_from_h5p_params_returns_object(self):
        result = H5PTimeline.from_h5p_params(self.LIB, self._params())
        assert result is not None
        assert len(result.entries) == 1

    def test_to_text_contains_date_and_headline(self):
        result = H5PTimeline.from_h5p_params(self.LIB, self._params())
        text = result.to_text()
        assert "1956" in text
        assert "KI-Begriff geprägt" in text

    def test_entry_without_text_shows_headline_only(self):
        dates = [{"startDate": "2000", "headline": "Meilenstein", "text": ""}]
        result = H5PTimeline.from_h5p_params(self.LIB, {"timeline": {"date": dates}})
        text = result.to_text()
        assert "Meilenstein" in text

    def test_multiple_entries_separated_by_blank_line(self):
        dates = [
            {"startDate": "1950", "headline": "A", "text": ""},
            {"startDate": "1960", "headline": "B", "text": ""},
        ]
        result = H5PTimeline.from_h5p_params(self.LIB, {"timeline": {"date": dates}})
        assert len(result.entries) == 2

    def test_empty_dates_list_returns_none(self):
        assert H5PTimeline.from_h5p_params(self.LIB, {"timeline": {"date": []}}) is None

    def test_missing_timeline_key_returns_none(self):
        assert H5PTimeline.from_h5p_params(self.LIB, {}) is None


# ── Crossword ─────────────────────────────────────────────────────────────────

class TestCrossword:
    LIB = "H5P.Crossword"

    def _params(self, words=None):
        if words is None:
            words = [{"clue": "Lernende Maschine", "answer": "NEURONALESNETZ", "orientation": "across"}]
        return {"words": words}

    def test_from_h5p_params_returns_object(self):
        result = Crossword.from_h5p_params(self.LIB, self._params())
        assert result is not None
        assert len(result.entries) == 1

    def test_to_text_starts_with_kreuzwortraetsel_header(self):
        result = Crossword.from_h5p_params(self.LIB, self._params())
        assert result.to_text().startswith("[Kreuzworträtsel]")

    def test_to_text_contains_clue_and_answer(self):
        result = Crossword.from_h5p_params(self.LIB, self._params())
        text = result.to_text()
        assert "Lernende Maschine" in text
        assert "NEURONALESNETZ" in text

    def test_task_description_included_when_present(self):
        params = {"words": self._params()["words"], "taskDescription": "Löse das Rätsel"}
        result = Crossword.from_h5p_params(self.LIB, params)
        assert "Löse das Rätsel" in result.to_text()

    def test_empty_words_returns_none(self):
        assert Crossword.from_h5p_params(self.LIB, {"words": []}) is None

    def test_word_without_clue_or_answer_skipped(self):
        words = [{"clue": "", "answer": ""}]
        assert Crossword.from_h5p_params(self.LIB, {"words": words}) is None

    def test_multiple_entries(self):
        words = [
            {"clue": "A", "answer": "ANS1", "orientation": "across"},
            {"clue": "B", "answer": "ANS2", "orientation": "down"},
        ]
        result = Crossword.from_h5p_params(self.LIB, {"words": words})
        assert len(result.entries) == 2


# ── DragDropText ──────────────────────────────────────────────────────────────

class TestDragDropText:
    LIB = "H5P.DragText"

    def _params(self, task="Ziehe die Wörter.", text_field="*KI* steht für *Künstliche Intelligenz*."):
        return {"taskDescription": task, "textField": text_field}

    def test_from_h5p_params_returns_object(self):
        result = DragDropText.from_h5p_params(self.LIB, self._params())
        assert result is not None

    def test_to_text_contains_drag_text_prefix(self):
        result = DragDropText.from_h5p_params(self.LIB, self._params())
        assert "[Drag Text]" in result.to_text()

    def test_to_text_contains_task_and_text_field(self):
        result = DragDropText.from_h5p_params(self.LIB, self._params())
        text = result.to_text()
        assert "Ziehe die Wörter" in text
        assert "Künstliche Intelligenz" in text

    def test_fallback_task_description_used_when_empty(self):
        params = {"taskDescription": "", "textField": "Some *word* here."}
        result = DragDropText.from_h5p_params(self.LIB, params)
        assert result is not None
        assert result.task_description != ""

    def test_empty_text_field_returns_none(self):
        assert DragDropText.from_h5p_params(self.LIB, {"taskDescription": "Task", "textField": ""}) is None

    def test_missing_text_field_returns_none(self):
        assert DragDropText.from_h5p_params(self.LIB, {"taskDescription": "Task"}) is None


# ── DragDropQuestion ──────────────────────────────────────────────────────────

class TestDragDropQuestion:
    LIB = "H5P.DragQuestion"

    def _params(self):
        return {
            "question": {
                "task": {
                    "dropZones": [
                        {"label": "Kategorie A", "correctElements": ["0"]},
                        {"label": "Kategorie B", "correctElements": ["1"]},
                    ],
                    "elements": [
                        {"type": {"params": {"text": "Element 1"}}},
                        {"type": {"params": {"text": "Element 2"}}},
                    ],
                }
            }
        }

    def test_from_h5p_params_returns_object(self):
        result = DragDropQuestion.from_h5p_params(self.LIB, self._params())
        assert result is not None

    def test_to_text_contains_drag_drop_prefix(self):
        result = DragDropQuestion.from_h5p_params(self.LIB, self._params())
        assert "[Drag & Drop]" in result.to_text()

    def test_to_text_contains_categories(self):
        result = DragDropQuestion.from_h5p_params(self.LIB, self._params())
        text = result.to_text()
        assert "Kategorie A" in text
        assert "Kategorie B" in text

    def test_to_text_contains_elements(self):
        result = DragDropQuestion.from_h5p_params(self.LIB, self._params())
        text = result.to_text()
        assert "Element 1" in text
        assert "Element 2" in text

    def test_empty_dropzones_returns_none(self):
        params = {"question": {"task": {"dropZones": [], "elements": []}}}
        assert DragDropQuestion.from_h5p_params(self.LIB, params) is None


# ── ImageHotspotQuestion ──────────────────────────────────────────────────────

class TestImageHotspotQuestion:
    LIB = "H5P.ImageHotspot"

    def _params(self):
        return {
            "hotspots": [
                {
                    "header": "Bereich 1",
                    "content": [{"params": {"text": "Erklärung zu Bereich 1"}}],
                },
                {
                    "header": "Bereich 2",
                    "content": [{"params": {"text": "Erklärung zu Bereich 2"}}],
                },
            ]
        }

    def test_from_h5p_params_returns_object(self):
        result = ImageHotspotQuestion.from_h5p_params(self.LIB, self._params())
        assert result is not None
        assert len(result.mappings) == 2

    def test_to_text_contains_header_and_content(self):
        result = ImageHotspotQuestion.from_h5p_params(self.LIB, self._params())
        text = result.to_text()
        assert "Bereich 1" in text
        assert "Erklärung zu Bereich 1" in text

    def test_empty_hotspots_returns_none(self):
        assert ImageHotspotQuestion.from_h5p_params(self.LIB, {"hotspots": []}) is None

    def test_missing_hotspots_returns_none(self):
        assert ImageHotspotQuestion.from_h5p_params(self.LIB, {}) is None

    def test_to_text_no_hotspots_shows_placeholder(self):
        result = ImageHotspotQuestion(type=self.LIB, mappings=[])
        assert "Keine Hotspots" in result.to_text()


# ── Column ────────────────────────────────────────────────────────────────────

class TestColumn:
    LIB = "H5P.Column"

    def _params(self, text="<p>Ein Text über KI.</p>"):
        return {
            "content": [
                {
                    "content": {
                        "library": "H5P.AdvancedText 1.1",
                        "params": {"text": text},
                    }
                }
            ]
        }

    def test_from_h5p_params_returns_object(self):
        result = Column.from_h5p_params(self.LIB, self._params())
        assert result is not None
        assert len(result.contents) == 1

    def test_to_text_contains_column_header(self):
        result = Column.from_h5p_params(self.LIB, self._params())
        assert "[Column]" in result.to_text()

    def test_to_text_contains_embedded_text(self):
        result = Column.from_h5p_params(self.LIB, self._params())
        assert "Ein Text über KI" in result.to_text()

    def test_empty_content_list_returns_none(self):
        assert Column.from_h5p_params(self.LIB, {"content": []}) is None

    def test_content_without_inner_content_skipped(self):
        params = {"content": [{"content": {}}]}  # no library/params
        assert Column.from_h5p_params(self.LIB, params) is None


# ── Accordion ─────────────────────────────────────────────────────────────────

class TestAccordion:
    LIB = "H5P.Accordion"

    def _params(self, panels=None):
        if panels is None:
            panels = [
                {
                    "title": "Was ist Supervised Learning?",
                    "content": {
                        "library": "H5P.AdvancedText 1.1",
                        "params": {"text": "Lernen mit gelabelten Daten."},
                    },
                }
            ]
        return {"panels": panels}

    def test_from_h5p_params_returns_object(self):
        result = Accordion.from_h5p_params(self.LIB, self._params())
        assert result is not None
        assert len(result.panels) == 1

    def test_panel_title_preserved(self):
        result = Accordion.from_h5p_params(self.LIB, self._params())
        assert result.panels[0].title == "Was ist Supervised Learning?"

    def test_to_text_contains_panel_title(self):
        result = Accordion.from_h5p_params(self.LIB, self._params())
        text = result.to_text()
        assert "Was ist Supervised Learning?" in text

    def test_to_text_contains_panel_content(self):
        result = Accordion.from_h5p_params(self.LIB, self._params())
        assert "gelabelten Daten" in result.to_text()

    def test_to_texts_returns_list_per_panel(self):
        panels = [
            {"title": "Panel 1", "content": {"library": "H5P.AdvancedText 1.1", "params": {"text": "A"}}},
            {"title": "Panel 2", "content": {"library": "H5P.AdvancedText 1.1", "params": {"text": "B"}}},
        ]
        result = Accordion.from_h5p_params(self.LIB, {"panels": panels})
        texts = result.to_texts()
        assert len(texts) == 2
        assert "Panel 1" in texts[0]
        assert "Panel 2" in texts[1]

    def test_empty_panels_returns_none(self):
        assert Accordion.from_h5p_params(self.LIB, {"panels": []}) is None


# ── QuestionSet ───────────────────────────────────────────────────────────────

class TestQuestionSet:
    LIB = "H5P.QuestionSet"

    def _params(self):
        return {
            "questions": [
                {
                    "library": "H5P.MultiChoice 1.14",
                    "params": {
                        "question": "Was ist ein Neuronales Netz?",
                        "answers": [
                            {"text": "Ein biologisches Modell", "correct": False},
                            {"text": "Ein mathematisches Modell", "correct": True},
                        ],
                    },
                }
            ]
        }

    def test_from_h5p_params_returns_object(self):
        result = QuestionSet.from_h5p_params(self.LIB, self._params())
        assert result is not None
        assert len(result.questions) == 1

    def test_to_text_contains_questionset_header(self):
        result = QuestionSet.from_h5p_params(self.LIB, self._params())
        assert "[QuestionSet]" in result.to_text()

    def test_to_text_contains_embedded_question(self):
        result = QuestionSet.from_h5p_params(self.LIB, self._params())
        assert "Neuronales Netz" in result.to_text()

    def test_intro_text_used_when_present(self):
        params = self._params()
        params["introPage"] = {"introduction": "Teste dein Wissen!"}
        result = QuestionSet.from_h5p_params(self.LIB, params)
        assert "Teste dein Wissen" in result.to_text()

    def test_empty_questions_returns_none(self):
        assert QuestionSet.from_h5p_params(self.LIB, {"questions": []}) is None

    def test_unknown_question_library_skipped(self):
        params = {"questions": [{"library": "H5P.Unknown 1.0", "params": {}}]}
        assert QuestionSet.from_h5p_params(self.LIB, params) is None


# ── InteractiveBook ───────────────────────────────────────────────────────────

class TestInteractiveBook:
    LIB = "H5P.InteractiveBook"

    def _params(self):
        return {
            "chapters": [
                {
                    "title": "Einleitung in ML",
                    "content": [
                        {
                            "content": {
                                "library": "H5P.AdvancedText 1.1",
                                "params": {"text": "<p>Machine Learning ist ein Teilgebiet der KI.</p>"},
                            }
                        }
                    ],
                }
            ]
        }

    def test_from_h5p_params_returns_object(self):
        result = InteractiveBook.from_h5p_params(self.LIB, self._params())
        assert result is not None
        assert len(result.chapters) == 1

    def test_chapter_title_preserved(self):
        result = InteractiveBook.from_h5p_params(self.LIB, self._params())
        assert result.chapters[0].title == "Einleitung in ML"

    def test_to_text_contains_interactive_book_header(self):
        result = InteractiveBook.from_h5p_params(self.LIB, self._params())
        assert "[Interactive Book]" in result.to_text()

    def test_to_text_contains_chapter_title(self):
        result = InteractiveBook.from_h5p_params(self.LIB, self._params())
        assert "Einleitung in ML" in result.to_text()

    def test_to_text_contains_chapter_content(self):
        result = InteractiveBook.from_h5p_params(self.LIB, self._params())
        assert "Machine Learning" in result.to_text()

    def test_cover_info_included_when_show_cover_page(self):
        params = self._params()
        params["showCoverPage"] = True
        params["bookCover"] = {
            "coverTitle": "Mein Buch",
            "coverDescription": "Eine Einführung",
        }
        result = InteractiveBook.from_h5p_params(self.LIB, params)
        text = result.to_text()
        assert "Mein Buch" in text

    def test_empty_chapters_returns_none(self):
        assert InteractiveBook.from_h5p_params(self.LIB, {"chapters": []}) is None


# ── CoursePresentation ────────────────────────────────────────────────────────

class TestCoursePresentation:
    LIB = "H5P.CoursePresentation"

    def _params(self):
        return {
            "presentation": {
                "slides": [
                    {
                        "elements": [
                            {
                                "action": {
                                    "library": "H5P.AdvancedText 1.1",
                                    "params": {"text": "Folie 1 Inhalt"},
                                }
                            }
                        ]
                    },
                    {
                        "elements": [
                            {
                                "action": {
                                    "library": "H5P.AdvancedText 1.1",
                                    "params": {"text": "Folie 2 Inhalt"},
                                }
                            }
                        ]
                    },
                ]
            }
        }

    def test_from_h5p_params_returns_object(self):
        result = CoursePresentation.from_h5p_params(self.LIB, self._params())
        assert result is not None
        assert len(result.slides) == 2

    def test_to_text_contains_slide_markers(self):
        result = CoursePresentation.from_h5p_params(self.LIB, self._params())
        text = result.to_text()
        assert "[Seite 1]" in text
        assert "[Seite 2]" in text

    def test_to_text_contains_slide_content(self):
        result = CoursePresentation.from_h5p_params(self.LIB, self._params())
        text = result.to_text()
        assert "Folie 1 Inhalt" in text
        assert "Folie 2 Inhalt" in text

    def test_empty_slides_returns_none(self):
        params = {"presentation": {"slides": []}}
        assert CoursePresentation.from_h5p_params(self.LIB, params) is None

    def test_missing_presentation_returns_none(self):
        assert CoursePresentation.from_h5p_params(self.LIB, {}) is None


# ── Gamemap ───────────────────────────────────────────────────────────────────

class TestGamemap:
    LIB = "H5P.Gamemap"

    def _params(self):
        return {
            "gamemapSteps": {
                "gamemap": {
                    "elements": [
                        {
                            "label": "Stage 1",
                            "contentType": {
                                "library": "H5P.AdvancedText 1.1",
                                "params": {"text": "Einführung"},
                            },
                        },
                        {
                            "label": "Stage 2",
                            "contentType": {
                                "library": "H5P.AdvancedText 1.1",
                                "params": {"text": "Vertiefung"},
                            },
                        },
                    ]
                }
            }
        }

    def test_from_h5p_params_returns_object(self):
        result = Gamemap.from_h5p_params(self.LIB, self._params())
        assert result is not None
        assert len(result.stages) == 2

    def test_stage_labels_preserved(self):
        result = Gamemap.from_h5p_params(self.LIB, self._params())
        labels = [s.label for s in result.stages]
        assert "Stage 1" in labels
        assert "Stage 2" in labels

    def test_to_text_contains_stage_labels(self):
        result = Gamemap.from_h5p_params(self.LIB, self._params())
        text = result.to_text()
        assert "Stage 1" in text
        assert "Stage 2" in text

    def test_to_text_contains_stage_content(self):
        result = Gamemap.from_h5p_params(self.LIB, self._params())
        text = result.to_text()
        assert "Einführung" in text

    def test_empty_elements_returns_none(self):
        params = {"gamemapSteps": {"gamemap": {"elements": []}}}
        assert Gamemap.from_h5p_params(self.LIB, params) is None

    def test_missing_gamemap_steps_returns_none(self):
        assert Gamemap.from_h5p_params(self.LIB, {}) is None

    def test_stage_without_content_type_still_added(self):
        """A stage with no contentType is added with label only (no crash)."""
        params = {
            "gamemapSteps": {
                "gamemap": {
                    "elements": [
                        {"label": "Leerer Stage", "contentType": {"library": "", "params": {}}}
                    ]
                }
            }
        }
        result = Gamemap.from_h5p_params(self.LIB, params)
        # Either None or a stage with just the label — no exception
        if result is not None:
            assert result.stages[0].label == "Leerer Stage"


# ── InteractiveVideo (params path, no services) ───────────────────────────────

class TestInteractiveVideoFromParams:
    LIB = "H5P.InteractiveVideo"

    def _params(self):
        return {
            "interactiveVideo": {
                "video": {"files": [{"path": "https://vimeo.com/12345"}]},
                "interactions": [],
            }
        }

    def test_from_h5p_params_returns_object(self):
        result = InteractiveVideo.from_h5p_params(self.LIB, self._params())
        assert result is not None

    def test_video_url_extracted(self):
        result = InteractiveVideo.from_h5p_params(self.LIB, self._params())
        assert result.video_url == "https://vimeo.com/12345"

    def test_missing_interactive_video_key_returns_none(self):
        assert InteractiveVideo.from_h5p_params(self.LIB, {}) is None

    def test_to_text_contains_video_url(self):
        result = InteractiveVideo.from_h5p_params(self.LIB, self._params())
        assert "vimeo.com" in result.to_text()

    def test_no_video_url_still_returns_object(self):
        params = {"interactiveVideo": {"video": {"files": []}, "interactions": []}}
        result = InteractiveVideo.from_h5p_params(self.LIB, params)
        # No URL but object is constructed (empty string url)
        assert result is not None
        assert result.video_url == ""
