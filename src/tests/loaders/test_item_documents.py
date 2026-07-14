"""Unit tests for structured item documents (QuizItem/GlossaryEntry/Flashcard).

Covers:
- h5p_payloads.collect_item_payloads: recursive content.json walk per H5P type
- Module.to_item_documents: document shape, key schemes, dedup, deep links
- Module.to_document: slim module text (inventory line, no answer leaks)
- get_data key/hash extensions: source_doc_key override, payload_fingerprint
  and EmptyModule marker in the content hash
"""

from llama_index.core import Document

from src.loaders.get_data import _content_hash, _source_doc_key
from src.loaders.models.glossary import Glossary, GlossaryEntry
from src.loaders.models.h5pactivities.h5p_payloads import collect_item_payloads
from src.loaders.models.module import Module
from src.vectordb.doc_types import FLASHCARD, GLOSSARY_ENTRY, QUIZ_ITEM


def _module(**overrides) -> Module:
    defaults = dict(
        id=456,
        visible=1,
        name="Testmodul Überwachtes Lernen",
        modname="h5pactivity",
        url="https://moodle.ki-campus.org/mod/h5pactivity/view.php?id=456",
    )
    defaults.update(overrides)
    return Module(**defaults)


def _multichoice_params(question="<p>Was ist ML?</p>"):
    return {
        "question": question,
        "answers": [
            {"text": "Ein KI-Teilgebiet", "correct": True},
            {"text": "Ein Betriebssystem", "correct": False},
        ],
    }


# ---------------------------------------------------------------------------
# collect_item_payloads
# ---------------------------------------------------------------------------


class TestCollectItemPayloads:
    def test_standalone_multichoice(self):
        quiz, cards = collect_item_payloads("H5P.MultiChoice 1.16", _multichoice_params())
        assert cards == []
        assert len(quiz) == 1
        assert quiz[0]["kind"] == "multichoice"
        assert quiz[0]["question"] == "Was ist ML?"  # HTML stripped
        assert quiz[0]["correct_answers"] == ["Ein KI-Teilgebiet"]
        assert quiz[0]["incorrect_answers"] == ["Ein Betriebssystem"]

    def test_truefalse_normalized_to_wahr_falsch(self):
        quiz, _ = collect_item_payloads("H5P.TrueFalse 1.8", {"question": "ML ist KI?", "correct": "true"})
        assert quiz[0]["kind"] == "truefalse"
        assert quiz[0]["correct_answers"] == ["Wahr"]
        assert quiz[0]["incorrect_answers"] == ["Falsch"]

    def test_singlechoiceset_extracts_all_questions(self):
        params = {
            "choices": [
                {"question": "Frage 1?", "answers": ["Richtig 1", "Falsch 1"]},
                {"question": "Frage 2?", "answers": ["Richtig 2", "Falsch 2", "Falsch 3"]},
            ]
        }
        quiz, _ = collect_item_payloads("H5P.SingleChoiceSet 1.11", params)
        assert len(quiz) == 2  # from_h5p_params would keep only the first
        assert quiz[1]["correct_answers"] == ["Richtig 2"]
        assert quiz[1]["incorrect_answers"] == ["Falsch 2", "Falsch 3"]

    def test_questionset_nested_with_sub_content_id(self):
        content = {
            "introPage": {"introduction": "Quiz zum Modul"},
            "questions": [
                {
                    "library": "H5P.MultiChoice 1.16",
                    "params": _multichoice_params(),
                    "subContentId": "abc-123",
                }
            ],
        }
        quiz, _ = collect_item_payloads("H5P.QuestionSet 1.20", content)
        assert len(quiz) == 1
        assert quiz[0]["sub_content_id"] == "abc-123"

    def test_interactive_video_carries_timestamp_and_summary(self):
        content = {
            "interactiveVideo": {
                "assets": {
                    "interactions": [
                        {
                            "duration": {"from": 42, "to": 50},
                            "action": {
                                "library": "H5P.TrueFalse 1.8",
                                "params": {"question": "Stimmt das?", "correct": "false"},
                                "subContentId": "s1",
                            },
                        }
                    ]
                },
                "summary": {
                    "task": {
                        "library": "H5P.Summary 1.10",
                        "params": {"intro": "Check", "summaries": [{"summary": ["Richtig.", "Falsch."]}]},
                    }
                },
            }
        }
        quiz, _ = collect_item_payloads("H5P.InteractiveVideo 1.27", content)
        kinds = {q["kind"] for q in quiz}
        assert kinds == {"truefalse", "summary"}
        truefalse = next(q for q in quiz if q["kind"] == "truefalse")
        assert truefalse["start_seconds"] == 42
        assert truefalse["sub_content_id"] == "s1"
        summary = next(q for q in quiz if q["kind"] == "summary")
        assert summary["statement_groups"] == [["Richtig.", "Falsch."]]

    def test_blanks_keeps_solutions_in_payload(self):
        params = {"text": "Fülle aus", "questions": ["Berlin ist die *Hauptstadt* von Deutschland."]}
        quiz, _ = collect_item_payloads("H5P.Blanks 1.14", params)
        assert quiz[0]["kind"] == "blanks"
        assert "*Hauptstadt*" in quiz[0]["text_with_blanks"]

    def test_dialogcards_become_flashcards(self):
        params = {"dialogs": [{"text": "<p>Neuron</p>", "answer": "Grundeinheit eines neuronalen Netzes"}]}
        quiz, cards = collect_item_payloads("H5P.Dialogcards 1.9", params)
        assert quiz == []
        assert cards == [{"front": "Neuron", "back": "Grundeinheit eines neuronalen Netzes"}]

    def test_malformed_content_returns_empty_instead_of_raising(self):
        quiz, cards = collect_item_payloads("H5P.MultiChoice 1.16", {"question": "x", "answers": "kaputt"})
        assert quiz == [] and cards == []


# ---------------------------------------------------------------------------
# Module.to_item_documents
# ---------------------------------------------------------------------------


class TestToItemDocuments:
    def test_quiz_item_document_shape(self):
        module = _module()
        module.quiz_items = [
            {
                "kind": "multichoice",
                "question": "Was ist Overfitting?",
                "correct_answers": ["Auswendiglernen der Trainingsdaten"],
                "incorrect_answers": ["Zu kleines Modell"],
                "sub_content_id": "abc-123",
            }
        ]
        docs = module.to_item_documents(course_id=79)
        assert len(docs) == 1
        md = docs[0].metadata
        assert md["type"] == QUIZ_ITEM
        assert md["source_doc_key"] == "moodle:79:456:quiz:abc-123"
        assert md["fullname"] == "Testmodul Überwachtes Lernen"  # module name, not the question
        assert md["title"].startswith("Was ist Overfitting?")
        assert md["payload"]["correct_answers"] == ["Auswendiglernen der Trainingsdaten"]
        assert md["payload_fingerprint"]
        assert md["item_order"] == 0
        # Embedding text is answer-free (defense in depth)
        assert "Auswendiglernen der Trainingsdaten" in docs[0].text  # as one of the options
        assert "Korrekt" not in docs[0].text

    def test_quiz_key_falls_back_to_content_hash(self):
        module = _module()
        module.quiz_items = [
            {"kind": "truefalse", "question": "F?", "correct_answers": ["Wahr"], "incorrect_answers": ["Falsch"]}
        ]
        key = module.to_item_documents(79)[0].metadata["source_doc_key"]
        assert key.startswith("moodle:79:456:quiz:")
        assert len(key.rsplit(":", 1)[1]) == 12  # sha fallback id

    def test_duplicate_questions_get_ordinal_suffix(self):
        payload = {"kind": "truefalse", "question": "F?", "correct_answers": ["Wahr"], "incorrect_answers": ["Falsch"]}
        module = _module()
        module.quiz_items = [dict(payload), dict(payload)]
        keys = [d.metadata["source_doc_key"] for d in module.to_item_documents(79)]
        assert len(set(keys)) == 2
        assert keys[1] == f"{keys[0]}-2"

    def test_glossary_entries_with_deep_link(self):
        module = _module(modname="glossary", url="https://moodle.ki-campus.org/mod/glossary/view.php?id=456")
        module.glossary = Glossary(
            glossary_id=1,
            module_id=456,
            entries=[GlossaryEntry(id=7, concept="Neuron", definition="Grundeinheit")],
        )
        docs = module.to_item_documents(79)
        md = docs[0].metadata
        assert md["type"] == GLOSSARY_ENTRY
        assert md["source_doc_key"] == "moodle:79:456:glossary:7"
        assert md["title"] == "Neuron"
        assert md["url"].endswith("&mode=entry&hook=7")
        assert docs[0].text == "Neuron: Grundeinheit"

    def test_flashcard_documents(self):
        module = _module()
        module.flashcards = [{"front": "Neuron", "back": "Grundeinheit"}]
        docs = module.to_item_documents(79)
        md = docs[0].metadata
        assert md["type"] == FLASHCARD
        assert md["source_doc_key"].startswith("moodle:79:456:card:")
        assert docs[0].text == "Neuron: Grundeinheit"

    def test_payload_fingerprint_changes_with_solution(self):
        base = {"kind": "truefalse", "question": "F?", "correct_answers": ["Wahr"], "incorrect_answers": ["Falsch"]}
        flipped = {**base, "correct_answers": ["Falsch"], "incorrect_answers": ["Wahr"]}
        m1, m2 = _module(), _module()
        m1.quiz_items, m2.quiz_items = [base], [flipped]
        fp1 = m1.to_item_documents(79)[0].metadata["payload_fingerprint"]
        fp2 = m2.to_item_documents(79)[0].metadata["payload_fingerprint"]
        assert fp1 != fp2


# ---------------------------------------------------------------------------
# Slim Module.to_document
# ---------------------------------------------------------------------------


class TestSlimToDocument:
    def test_glossary_body_replaced_by_inventory(self):
        module = _module(modname="glossary")
        module.glossary = Glossary(
            glossary_id=1,
            module_id=456,
            entries=[
                GlossaryEntry(id=1, concept="Neuron", definition="Grundeinheit eines Netzes"),
                GlossaryEntry(id=2, concept="Gradient", definition="Ableitung der Verlustfunktion"),
            ],
        )
        doc = module.to_document(79)
        assert "Grundeinheit eines Netzes" not in doc.text  # definitions live in item docs
        assert "Glossar mit 2 Einträgen" in doc.text
        assert "Neuron" in doc.text and "Gradient" in doc.text  # concepts stay discoverable
        assert doc.metadata["type"] == "module"  # inventory prevents EmptyModule

    def test_pure_flashcard_module_keeps_only_inventory(self):
        module = _module(h5p_content_type="H5P.Flashcards 1.7")
        module.interactive_video = {"video_url": "", "vimeo_id": None, "interactions": ["Neuron: Grundeinheit"]}
        module.flashcards = [{"front": "Neuron", "back": "Grundeinheit"}]
        doc = module.to_document(79)
        assert "Neuron: Grundeinheit" not in doc.text  # cards live in item docs
        assert "1 Karteikarte(n)" in doc.text

    def test_quiz_inventory_counts(self):
        module = _module()
        module.quiz_items = [{"kind": "truefalse", "question": "F?", "correct_answers": ["Wahr"]}] * 3
        doc = module.to_document(79)
        assert "3 Quizfrage(n)" in doc.text

    def test_truly_empty_module_stays_empty_marker(self):
        doc = _module(modname="page").to_document(79)
        assert doc.metadata["type"] == "EmptyModule"
        assert "unsupported_label" not in doc.metadata


class TestUnsupportedModname:
    """Content types with no extraction handler (module.type is None, e.g.
    native Moodle Quiz) are tagged distinctly from a genuinely empty module —
    see UNSUPPORTED_MODNAME_LABELS and the chat fallback in answer.py."""

    def test_known_unsupported_modname_gets_friendly_label(self):
        doc = _module(modname="quiz", name="Lernziel-Check II").to_document(79)
        assert doc.metadata["type"] == "EmptyModule"
        assert doc.metadata["unsupported_modname"] == "quiz"
        assert doc.metadata["unsupported_label"] == "natives Moodle-Quiz"

    def test_unknown_modname_gets_generic_label_naming_the_raw_type(self):
        doc = _module(modname="forum", name="Diskussion").to_document(79)
        assert doc.metadata["unsupported_modname"] == "forum"
        assert "forum" in doc.metadata["unsupported_label"]

    def test_supported_modname_never_gets_unsupported_label_even_if_empty(self):
        # "page" IS in ModuleTypes — an empty page is genuinely-empty, not
        # "unsupported"; module.type is not None here.
        doc = _module(modname="page", text=None, intro=None).to_document(79)
        assert doc.metadata["type"] == "EmptyModule"
        assert "unsupported_label" not in doc.metadata


# ---------------------------------------------------------------------------
# get_data key/hash extensions
# ---------------------------------------------------------------------------


class TestKeyOverrideAndHash:
    def test_source_doc_key_honors_preset_key(self):
        doc = Document(
            text="x",
            metadata={"source": "Moodle", "course_id": 79, "module_id": 456, "source_doc_key": "moodle:79:456:quiz:abc"},
        )
        assert _source_doc_key(doc) == "moodle:79:456:quiz:abc"

    def test_without_preset_key_module_scheme_unchanged(self):
        doc = Document(text="x", metadata={"source": "Moodle", "course_id": 79, "module_id": 456})
        assert _source_doc_key(doc) == "moodle:79:456"

    def test_payload_fingerprint_changes_hash(self):
        md = {"source": "Moodle", "url": "https://x.com", "title": "T"}
        doc_plain = Document(text="same", metadata=md)
        doc_payload = Document(text="same", metadata={**md, "payload_fingerprint": '{"a": 1}'})
        doc_payload2 = Document(text="same", metadata={**md, "payload_fingerprint": '{"a": 2}'})
        assert _content_hash(doc_plain, 500, 83) != _content_hash(doc_payload, 500, 83)
        assert _content_hash(doc_payload, 500, 83) != _content_hash(doc_payload2, 500, 83)

    def test_empty_module_flip_changes_hash(self):
        md = {"source": "Moodle", "url": "https://x.com", "title": "T"}
        doc_module = Document(text="Module Name: X", metadata={**md, "type": "module"})
        doc_empty = Document(text="Module Name: X", metadata={**md, "type": "EmptyModule"})
        assert _content_hash(doc_module, 500, 83) != _content_hash(doc_empty, 500, 83)

    def test_regular_type_does_not_change_hash(self):
        # Deliberate: folding the full type into the hash would invalidate every
        # existing hash and force a fleet-wide re-embed. Only the EmptyModule
        # marker participates.
        md = {"source": "Drupal", "url": "https://x.com", "title": "T"}
        doc_untyped = Document(text="same", metadata=md)
        doc_typed = Document(text="same", metadata={**md, "type": "page"})
        assert _content_hash(doc_untyped, 500, 83) == _content_hash(doc_typed, 500, 83)


# ---------------------------------------------------------------------------
# Transcripts: VTT segments, Transcript documents, cue-aware chunking
# ---------------------------------------------------------------------------

VTT_SAMPLE = """WEBVTT

00:00:01.000 --> 00:00:04.000
Willkommen zum Kurs.

00:00:04.000 --> 00:00:08.000
Willkommen zum Kurs.
Heute geht es um Overfitting.

00:01:30.500 --> 00:01:35.000
Overfitting bedeutet Auswendiglernen.
"""


class TestTranscriptSegments:
    def test_convert_vtt_to_segments_keeps_start_times_and_dedups(self):
        from io import StringIO

        from src.loaders.helper import convert_vtt_to_segments

        segments = convert_vtt_to_segments(StringIO(VTT_SAMPLE))
        assert segments[0] == {"start_seconds": 1.0, "text": "Willkommen zum Kurs."}
        # duplicate caption line dropped, remaining text kept with its cue time
        assert segments[1]["text"] == "Heute geht es um Overfitting."
        assert segments[2]["start_seconds"] == 90.5

    def test_transcript_document_chunks_carry_start_seconds(self):
        from llama_index.core import Document

        from src.loaders.helper import iter_nodes_from_document_hierarchical

        doc = Document(
            text="ignored — segments drive the chunking",
            metadata={
                "type": "Transcript",
                "source": "Moodle",
                "url": "https://vimeo.com/123",
                "segments": [
                    {"start_seconds": 1.0, "text": "Willkommen zum Kurs."},
                    {"start_seconds": 90.5, "text": "Overfitting bedeutet Auswendiglernen."},
                ],
            },
        )
        nodes = list(iter_nodes_from_document_hierarchical(doc, chunk_size_tokens=500, chunk_overlap_tokens=83))
        assert len(nodes) == 1  # small transcript packs into one chunk
        md = nodes[0].metadata
        assert md["chunk_method"] == "transcript_cues_v1"
        assert md["start_seconds"] == 1.0
        assert "segments" not in md  # bulky list never reaches the index

    def test_transcript_chunks_split_on_token_budget(self):
        from llama_index.core import Document

        from src.loaders.helper import iter_nodes_from_document_hierarchical

        segments = [{"start_seconds": float(i * 10), "text": "wort " * 40} for i in range(10)]
        doc = Document(text="x", metadata={"type": "Transcript", "segments": segments})
        nodes = list(iter_nodes_from_document_hierarchical(doc, chunk_size_tokens=100, chunk_overlap_tokens=0))
        assert len(nodes) > 1
        assert nodes[1].metadata["start_seconds"] > nodes[0].metadata["start_seconds"]

    def test_module_emits_transcript_document_with_stable_key(self):
        from src.loaders.models.texttrack import TextTrack
        from src.vectordb.doc_types import TRANSCRIPT

        module = _module(modname="videotime")
        module.transcripts = [
            TextTrack(
                id=0,
                display_language="de",
                language="de",
                transcript="Willkommen zum Kurs.",
                segments=[{"start_seconds": 1.0, "text": "Willkommen zum Kurs."}],
                video_url="https://www.youtube.com/watch?v=abc",
            )
        ]
        docs = [d for d in module.to_item_documents(79) if d.metadata["type"] == TRANSCRIPT]
        assert len(docs) == 1
        md = docs[0].metadata
        assert md["source_doc_key"].startswith("moodle:79:456:transcript:")
        # Citation link stays on the KI-Campus module page — never the
        # external video platform. video_url is only used to derive the
        # stable key above, not stored in the payload (nothing reads it).
        assert md["url"] == str(module.url)
        assert md["payload"] == {}
        assert md["segments"]  # ride along for the cue-aware chunker
        # key derived from video_url, not position
        module.transcripts.insert(0, None)
        docs_after = [d for d in module.to_item_documents(79) if d.metadata["type"] == TRANSCRIPT]
        assert docs_after[0].metadata["source_doc_key"] == md["source_doc_key"]


# ---------------------------------------------------------------------------
# Books: chapter order + BookChapter documents
# ---------------------------------------------------------------------------


class TestBookChapters:
    def test_chapter_order_from_structure_walks_hierarchy(self):
        from src.loaders.models.book import chapter_order_from_structure

        structure = [
            {"title": "Intro", "href": "2/index.html", "subitems": [{"title": "Sub", "href": "10/index.html"}]},
            {"title": "Ende", "href": "3/index.html"},
        ]
        order = chapter_order_from_structure(structure)
        assert order == {"2": 0, "10": 1, "3": 2}

    def test_chapter_order_handles_none_and_dict(self):
        from src.loaders.models.book import chapter_order_from_structure

        assert chapter_order_from_structure(None) == {}
        assert chapter_order_from_structure({"href": "7/index.html"}) == {"7": 0}

    def test_module_emits_chapter_documents_with_deep_links(self):
        from src.loaders.models.book import Book, BookChapter
        from src.vectordb.doc_types import BOOK_CHAPTER

        module = _module(modname="book", url="https://moodle.ki-campus.org/mod/book/view.php?id=456")
        module.book = Book(
            book_id=1,
            module_id=456,
            chapters=[
                BookChapter(chapter_id="2", title="Einführung", html_text="Inhalt A"),
                BookChapter(chapter_id="10", title="Vertiefung", html_text="Inhalt B"),
            ],
        )
        docs = [d for d in module.to_item_documents(79) if d.metadata["type"] == BOOK_CHAPTER]
        assert len(docs) == 2
        assert docs[0].metadata["url"].endswith("&chapterid=2")
        assert docs[0].metadata["title"] == "Einführung"
        assert docs[0].metadata["item_order"] == 0
        assert "Inhalt A" in docs[0].text

    def test_book_body_replaced_by_inventory_in_module_doc(self):
        from src.loaders.models.book import Book, BookChapter

        module = _module(modname="book")
        module.book = Book(
            book_id=1,
            module_id=456,
            intro="Worum es geht.",
            chapters=[BookChapter(chapter_id="2", title="Einführung", html_text="Langer Kapiteltext")],
        )
        doc = module.to_document(79)
        assert "Langer Kapiteltext" not in doc.text  # chapters live in item docs
        assert "Worum es geht." in doc.text  # intro stays
        assert "Book mit 1 Kapiteln: Einführung" in doc.text
