"""Structured item-payload extraction from H5P content.json.

Walks the raw content.json recursively and collects quiz-question and
flashcard payloads WITH their solutions — independent of the text-rendering
handler chain (which flattens everything to answer-free module text, see the
to_text() methods). The payloads feed Module.to_item_documents(), which turns
them into standalone QuizItem/Flashcard index documents.

Deliberately a generic tree walk instead of per-container logic: H5P nests
content as {"library": ..., "params": ..., "subContentId": ...} nodes in
container-specific shapes (QuestionSet.questions, InteractiveVideo
assets.interactions[].action, Column content[].content, summary.task, ...) —
the walk finds them all without knowing each container's schema.
"""

import logging
from typing import Any, Optional

from src.loaders.models.hp5activities import strip_html
from src.loaders.models.h5pactivities.h5p_quiz_questions import QuizQuestion, TrueFalseQuestion
from src.loaders.models.h5pactivities.h5p_summary import Summary
from src.loaders.models.h5pactivities.h5p_blanks import FillInBlanksQuestion

logger = logging.getLogger(__name__)


def _clean(text: str) -> str:
    return strip_html(text or "").strip()


def _quiz_payload(question: str, correct: list[str], incorrect: list[str], kind: str,
                  sub_content_id: Optional[str], start_seconds: Optional[float]) -> dict:
    payload = {
        "kind": kind,
        "question": _clean(question),
        "correct_answers": [_clean(a) for a in correct if _clean(a)],
        "incorrect_answers": [_clean(a) for a in incorrect if _clean(a)],
    }
    if sub_content_id:
        payload["sub_content_id"] = sub_content_id
    if start_seconds is not None:
        payload["start_seconds"] = start_seconds
    return payload


def _multichoice_payloads(library: str, params: dict, sub_content_id, start_seconds) -> list[dict]:
    payloads = []
    if "H5P.MultiChoice" in library:
        quiz = QuizQuestion.from_h5p_params(library, params)
        if quiz:
            payloads.append(
                _quiz_payload(quiz.question, quiz.correct_answers, quiz.incorrect_answers,
                              "multichoice", sub_content_id, start_seconds)
            )
    else:  # SingleChoiceSet: parse ALL choices (from_h5p_params keeps only the first)
        for i, choice in enumerate(params.get("choices", [])):
            question_text = (choice.get("question") or "").strip()
            answers = [a.strip() for a in choice.get("answers", []) if a and a.strip()]
            if not question_text or not answers:
                continue
            # First answer is always correct in SingleChoiceSet.
            per_question_id = f"{sub_content_id}-{i}" if sub_content_id else None
            payloads.append(
                _quiz_payload(question_text, answers[:1], answers[1:],
                              "singlechoice", per_question_id, start_seconds)
            )
    return payloads


def _process_node(library: str, params: dict, sub_content_id: Optional[str],
                  start_seconds: Optional[float], quiz_items: list[dict], flashcards: list[dict]) -> None:
    """Turn one {library, params} node into payloads, or recurse into it."""
    if "H5P.MultiChoice" in library or "H5P.SingleChoiceSet" in library:
        quiz_items.extend(_multichoice_payloads(library, params, sub_content_id, start_seconds))
        return
    if "H5P.TrueFalse" in library:
        question = TrueFalseQuestion.from_h5p_params(library, params)
        if question:
            correct = "Wahr" if question.correct_answer else "Falsch"
            incorrect = "Falsch" if question.correct_answer else "Wahr"
            quiz_items.append(
                _quiz_payload(question.question, [correct], [incorrect], "truefalse", sub_content_id, start_seconds)
            )
        return
    if "H5P.Summary" in library:
        summary = Summary.from_h5p_params(library, params)
        if summary:
            payload = _quiz_payload(summary.intro, [], [], "summary", sub_content_id, start_seconds)
            payload["statement_groups"] = [
                [_clean(s) for s in group] for group in summary.statement_groups
            ]  # first statement of each group is the correct one
            quiz_items.append(payload)
        return
    if "H5P.Blanks" in library:
        blanks = FillInBlanksQuestion.from_h5p_params(library, params)
        if blanks:
            payload = _quiz_payload(blanks.question, [], [], "blanks", sub_content_id, start_seconds)
            # Solutions stay inline as H5P *solution* markup — the consumer
            # (tutor) parses them; module text renders neutral ___ placeholders.
            payload["text_with_blanks"] = _clean(blanks.text_with_blanks)
            quiz_items.append(payload)
        return
    if "H5P.Dialogcards" in library or "H5P.Flashcards" in library:
        cards = params.get("dialogs") if "Dialogcards" in library else params.get("cards")
        for card in cards or []:
            if not isinstance(card, dict):
                continue
            front = _clean(card.get("text", ""))
            back = _clean(card.get("answer", ""))
            if front and back:
                flashcards.append({"front": front, "back": back})
        return
    # Unknown/container type: recurse into its params.
    _walk(params, quiz_items, flashcards, start_seconds)


def _walk(obj: Any, quiz_items: list[dict], flashcards: list[dict],
          start_seconds: Optional[float] = None) -> None:
    if isinstance(obj, dict):
        # InteractiveVideo interaction wrapper: carries the video timestamp of
        # its "action" node — pass it down so quiz items know their position.
        duration = obj.get("duration")
        if isinstance(duration, dict) and isinstance(obj.get("action"), dict):
            start_seconds = duration.get("from", start_seconds)

        library = obj.get("library")
        params = obj.get("params")
        if isinstance(library, str) and isinstance(params, dict):
            _process_node(library, params, obj.get("subContentId"), start_seconds, quiz_items, flashcards)
            return
        for value in obj.values():
            _walk(value, quiz_items, flashcards, start_seconds)
    elif isinstance(obj, list):
        for value in obj:
            _walk(value, quiz_items, flashcards, start_seconds)


def collect_item_payloads(library: str, content: dict) -> tuple[list[dict], list[dict]]:
    """Extract (quiz_items, flashcards) payloads from a package's content.json.

    ``library`` is the package's main library (from h5p.json); ``content`` is
    the parsed content/content.json, i.e. the root library's params.
    """
    quiz_items: list[dict] = []
    flashcards: list[dict] = []
    try:
        _process_node(library, content, content.get("subContentId"), None, quiz_items, flashcards)
    except Exception:
        # Payload collection is additive on top of the text pipeline — a
        # malformed package must not fail the module extraction.
        logger.warning("H5P item-payload collection failed (library=%s)", library, exc_info=True)
        return [], []
    return quiz_items, flashcards
