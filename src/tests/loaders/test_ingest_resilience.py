"""Regression tests: a single broken source item must not abort the ingest run.

Background: `get_data.extract()` wraps Moochup, Moodle and Drupal in one
try/except that logs and *re-raises*. Anything escaping an extractor therefore
kills the whole run — and because Moodle runs before Drupal, a single unparseable
Moodle module also prevents the Drupal stage from ever starting. Worse, the
extraction happens *before* the content-hash comparison, so a deterministic
failure (broken H5P package, unknown Drupal course format) repeats on every
subsequent run: the index silently freezes until someone reads the logs.

Two gaps existed:
  * `get_module_contents` called `extract_h5p` without any error handling, while
    every other module type catches its own errors and returns a message.
  * `Drupal.get_course_type` raised on any unknown `field_format`, and the node
    loop in `get_page_type` has no error handling at all.
"""

import logging
import types

import pytest
import requests

from src.loaders.APICaller import MoodleMaintenanceError
from src.loaders.drupal import Drupal
from src.loaders.models.coursetopic import CourseTopic
from src.loaders.models.module import Module, ModuleTypes
from src.loaders.models.moodlecourse import MoodleCourse
from src.loaders.moodle import Moodle


def _module(module_id: int, modname: str = "h5pactivity") -> Module:
    return Module(
        id=module_id,
        visible=1,
        name=f"Modul {module_id}",
        url=f"https://moodle.example.org/mod/h5pactivity/view.php?id={module_id}",
        modname=modname,
    )


def _topic(*modules: Module) -> types.SimpleNamespace:
    return types.SimpleNamespace(id=1, modules=list(modules))


def _h5p_activity(module_id: int) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        coursemodule=module_id,
        intro=None,
        fileurl="https://moodle.example.org/pluginfile.php/1/mod_h5pactivity/package/broken.h5p",
        filename="broken.h5p",
    )


class TestModuleExtractionIsIsolated:
    """One poisonous module is reported as failed; the run keeps going."""

    @pytest.mark.parametrize(
        "exc",
        [
            # Missing "content/content.json" member — only BadZipFile is handled
            # inside _process_h5p_package.
            KeyError("There is no item named 'content/content.json' in the archive"),
            # Malformed content.json / h5p.json.
            ValueError("Expecting value: line 1 column 1 (char 0)"),
            # Handler hitting an unexpected H5P field type, e.g.
            # params.get("taskDescription", "").strip() on a dict.
            AttributeError("'dict' object has no attribute 'strip'"),
            # Package download answering 403/404/500 — raise_for_status() in
            # APICaller.getFile. Caught for embedded H5P in pages, but the
            # standalone H5P module path had no handler.
            requests.exceptions.HTTPError("403 Client Error: Forbidden"),
        ],
        ids=["missing_zip_member", "broken_json", "handler_type_error", "download_http_error"],
    )
    def test_h5p_failure_marks_module_failed_without_aborting(self, monkeypatch, exc):
        moodle = Moodle()

        def boom(module, activity):
            raise exc

        monkeypatch.setattr(moodle, "extract_h5p", boom)

        failed = moodle.get_module_contents(_topic(_module(4711)), [_h5p_activity(4711)])

        assert [fm.modul.id for fm in failed] == [4711]
        assert type(exc).__name__ in failed[0].err_message

    def test_later_modules_are_still_processed_after_a_failure(self, monkeypatch):
        """The failure must not truncate the rest of the topic."""
        moodle = Moodle()
        processed: list[int] = []

        def flaky(module, activity):
            processed.append(module.id)
            if module.id == 1:
                raise KeyError("content/content.json")
            return None

        monkeypatch.setattr(moodle, "extract_h5p", flaky)

        topic = _topic(_module(1), _module(2), _module(3))
        activities = [_h5p_activity(1), _h5p_activity(2), _h5p_activity(3)]
        failed = moodle.get_module_contents(topic, activities)

        assert processed == [1, 2, 3]
        assert [fm.modul.id for fm in failed] == [1]

    def test_site_maintenance_still_aborts_the_run(self, monkeypatch):
        """A site-wide outage is NOT a per-module failure.

        Swallowing it would let the run continue against a dead API, see far
        fewer documents than are indexed and — once past the stale-deletion
        threshold — delete existing content (see APICaller.MoodleMaintenanceError).
        """
        moodle = Moodle()

        def maintenance(module, activity):
            raise MoodleMaintenanceError("sitemaintenance")

        monkeypatch.setattr(moodle, "extract_h5p", maintenance)

        with pytest.raises(MoodleMaintenanceError):
            moodle.get_module_contents(_topic(_module(4711)), [_h5p_activity(4711)])

    def test_other_module_types_keep_their_own_error_contract(self, monkeypatch):
        """Extractors that return a message keep working unchanged."""
        moodle = Moodle()
        monkeypatch.setattr(moodle, "extract_glossary", lambda module: "Fehler beim Laden")

        failed = moodle.get_module_contents(_topic(_module(99, modname="glossary")), [])

        assert failed[0].err_message == "Fehler beim Laden"

    def test_invisible_modules_are_skipped(self, monkeypatch):
        moodle = Moodle()
        monkeypatch.setattr(
            moodle, "extract_h5p", lambda module, activity: pytest.fail("should not be called")
        )
        hidden = Module(
            id=5, visible=0, name="Versteckt", modname="h5pactivity",
            url="https://moodle.example.org/mod/h5pactivity/view.php?id=5",
        )

        assert moodle.get_module_contents(_topic(hidden), [_h5p_activity(5)]) == []


def _moodle_with_one_h5p_module(monkeypatch, extract_h5p, *, module_id: int = 4711, course_id: int = 7):
    """Real Moodle loader with the API layer stubbed: one course, one H5P module."""
    moodle = Moodle()
    course = MoodleCourse(
        id=course_id, shortname="k7", fullname="Kurs 7", displayname="Kurs 7",
        summary="Zusammenfassung", lang="de", url="https://moodle.example.org/course/view.php?id=",
    )
    topic = CourseTopic(id=1, name="Thema", summary=None, modules=[_module(module_id)])

    monkeypatch.setattr(moodle, "get_courses", lambda: [course])
    monkeypatch.setattr(moodle, "get_course_contents", lambda course_id: [topic])
    monkeypatch.setattr(moodle, "_load_module_intros_for_course", lambda course_id: None)
    monkeypatch.setattr(moodle, "get_h5p_module_ids", lambda course_id: [_h5p_activity(module_id)])
    monkeypatch.setattr(moodle, "extract_h5p", extract_h5p)
    # Best-effort Excel report at the end of the stream — irrelevant here.
    monkeypatch.setattr("src.loaders.moodle.save_failed_transcripts_to_excel", lambda **kwargs: None)
    return moodle


def _boom(module, activity):
    raise KeyError("content/content.json")


def _intro_then_boom(module, activity):
    # Mirrors extract_h5p: the intro is copied onto the module BEFORE the
    # package download, so a failing download leaves name + intro behind.
    module.intro = "<p>In diesem Modul lernst du Gradient Descent kennen.</p>"
    raise requests.exceptions.ReadTimeout("read timeout")


def _partial(module, activity):
    module.text = "Ein Absatz mit echtem Inhalt, der indexiert werden soll."
    return "Transkript konnte nicht geladen werden"


def _ok_but_empty(module, activity):
    # Successful extraction of a genuinely empty module — not a failure.
    return None


class TestDegradedModuleMarking:
    """Moodle marks failed modules whose document holds nothing beyond name/intro.

    The document is still yielded: whether an existing index version is kept is
    decided in get_data.py, which knows the index (see TestKeepExistingOnFailure).
    """

    def _module_docs(self, moodle):
        return [doc for _, doc in moodle.iter_course_documents_stream() if doc.metadata.get("module_id")]

    def test_failed_module_without_content_is_marked_and_still_yielded(self, monkeypatch):
        moodle = _moodle_with_one_h5p_module(monkeypatch, _boom)

        docs = self._module_docs(moodle)

        assert [d.metadata["module_id"] for d in docs] == [4711]
        assert docs[0].metadata["type"] == "EmptyModule"
        assert (7, 4711) in moodle.failed_module_keys
        assert (7, 4711) in moodle.degraded_module_keys

    def test_failed_module_with_only_intro_is_marked(self, monkeypatch):
        """Regression: the old length threshold missed modules with an intro."""
        moodle = _moodle_with_one_h5p_module(monkeypatch, _intro_then_boom)

        docs = self._module_docs(moodle)

        assert "Gradient Descent" in docs[0].text
        assert (7, 4711) in moodle.degraded_module_keys

    def test_partial_success_is_not_marked(self, monkeypatch):
        moodle = _moodle_with_one_h5p_module(monkeypatch, _partial)

        docs = self._module_docs(moodle)

        assert "echtem Inhalt" in docs[0].text
        assert (7, 4711) in moodle.failed_module_keys
        assert (7, 4711) not in moodle.degraded_module_keys

    def test_successful_empty_module_is_not_marked(self, monkeypatch):
        moodle = _moodle_with_one_h5p_module(monkeypatch, _ok_but_empty)

        self._module_docs(moodle)

        assert moodle.failed_module_keys == set()
        assert moodle.degraded_module_keys == set()


class TestHasExtractedContent:
    def test_name_only(self):
        m = _module(1)
        assert m.has_extracted_content(m.to_document(7)) is False

    def test_name_and_intro(self):
        m = _module(1)
        m.intro = "<p>Beschreibung des Moduls</p>"
        assert m.has_extracted_content(m.to_document(7)) is False

    def test_html_only_intro_counts_as_no_intro(self):
        m = _module(1)
        m.intro = "<p> </p>"
        assert m.has_extracted_content(m.to_document(7)) is False

    def test_text_content(self):
        m = _module(1)
        m.text = "Inhalt"
        assert m.has_extracted_content(m.to_document(7)) is True

    def test_intro_plus_text(self):
        m = _module(1)
        m.intro = "<p>Beschreibung</p>"
        m.text = "Inhalt"
        assert m.has_extracted_content(m.to_document(7)) is True


class TestKeepExistingOnFailure:
    """End-to-end through Fetch_Data.extract() with a fake index.

    Only the Moodle stage runs (RUN_SOURCES=MOODLE); Moodle itself is the real
    loader with its API layer stubbed, so the marking in moodle.py and the
    decision in get_data.py are exercised together.
    """

    MODULE_KEY = "moodle:7:4711"
    TRANSCRIPT_KEY = "moodle:7:4711:transcript:abc"
    SUMMARY_KEY = "moodle:7:summary"

    def _run(self, monkeypatch, extract_h5p, existing: dict[str, str]):
        from unittest.mock import MagicMock

        import src.loaders.get_data as get_data

        moodle = _moodle_with_one_h5p_module(monkeypatch, extract_h5p)

        store = MagicMock()
        store.load_content_hashes.side_effect = lambda source, index: dict(existing) if source == "Moodle" else {}
        store.any_match.return_value = False
        store.upload_documents.side_effect = lambda index, docs: len(docs)

        embedder = MagicMock()
        embedder.get_text_embedding.return_value = [0.0, 0.0, 0.0]
        embedder.get_text_embedding_batch.side_effect = lambda texts: [[0.0, 0.0, 0.0] for _ in texts]
        llm = MagicMock()
        llm.return_value.get_embedder.return_value = embedder

        monkeypatch.setenv("RUN_SOURCES", "MOODLE")
        # The stale guard refuses deletion when the run saw far fewer docs than
        # the index holds (protects against partial fetches) — disable it so the
        # test observes what the sweep WOULD delete.
        monkeypatch.setenv("STALE_DELETE_MIN_SEEN_RATIO", "0")
        monkeypatch.setattr(get_data, "LLM", llm)
        monkeypatch.setattr(get_data, "VectorDBAzureSearch", lambda: store)
        monkeypatch.setattr(get_data, "Moodle", lambda run_ctx=None: moodle)
        monkeypatch.setattr(get_data.RunLogger, "attach_append_blob_handler", lambda self: None)
        monkeypatch.setattr(get_data.RunLogger, "get_readonly_sas_url", lambda self, expiry_hours=24: None)
        # Fetch_Data reconfigures the shared "loader" logger (propagate=False,
        # own handler, level) — restore it afterwards so caplog-based tests
        # elsewhere keep working regardless of test order.
        loader_logger = logging.getLogger("loader")
        monkeypatch.setattr(loader_logger, "propagate", loader_logger.propagate)
        monkeypatch.setattr(loader_logger, "handlers", list(loader_logger.handlers))
        monkeypatch.setattr(loader_logger, "level", loader_logger.level)

        result = get_data.Fetch_Data(run_id="test-run", index_name="test-index").extract()

        deleted = [c.args[1] for c in store.delete_by_filter.call_args_list]
        uploaded = [d for c in store.upload_documents.call_args_list for d in c.args[1]]
        return result, deleted, uploaded

    @staticmethod
    def _deleted_keys(deleted):
        return {f.split("'")[1] for f in deleted if f.startswith("source_doc_key eq ")}

    def test_existing_module_is_kept_when_extraction_fails(self, monkeypatch):
        existing = {self.SUMMARY_KEY: "x", self.MODULE_KEY: "old-hash", self.TRANSCRIPT_KEY: "t-hash"}

        result, deleted, uploaded = self._run(monkeypatch, _boom, existing)

        assert self.MODULE_KEY not in self._deleted_keys(deleted)
        assert self.TRANSCRIPT_KEY not in self._deleted_keys(deleted)
        assert not [d for d in uploaded if d["source_doc_key"] == self.MODULE_KEY]
        counts = result["counts"]
        assert counts["moodle_documents"] == counts["moodle_ingested"] + counts["moodle_skipped"]

    def test_existing_module_with_intro_is_kept_when_download_times_out(self, monkeypatch):
        existing = {self.SUMMARY_KEY: "x", self.MODULE_KEY: "old-hash"}

        _, deleted, uploaded = self._run(monkeypatch, _intro_then_boom, existing)

        assert self.MODULE_KEY not in self._deleted_keys(deleted)
        assert not [d for d in uploaded if d["source_doc_key"] == self.MODULE_KEY]

    def test_new_module_gets_placeholder_when_extraction_fails(self, monkeypatch):
        """Regression: a failed NEW module must still be indexed, otherwise
        the API rejects chats in that module with "no module found"."""
        _, deleted, uploaded = self._run(monkeypatch, _boom, existing={self.SUMMARY_KEY: "x"})

        module_chunks = [d for d in uploaded if d["source_doc_key"] == self.MODULE_KEY]
        assert module_chunks
        assert {d["type"] for d in module_chunks} == {"EmptyModule"}
        assert module_chunks[0]["module_id"] == 4711

    def test_partial_success_still_updates_existing_module(self, monkeypatch):
        existing = {self.SUMMARY_KEY: "x", self.MODULE_KEY: "old-hash", self.TRANSCRIPT_KEY: "t-hash"}

        _, deleted, uploaded = self._run(monkeypatch, _partial, existing)

        assert f"source_doc_key eq '{self.MODULE_KEY}'" in deleted
        assert [d for d in uploaded if d["source_doc_key"] == self.MODULE_KEY]
        # Item documents of the failed module stay protected from the sweep.
        assert self.TRANSCRIPT_KEY not in self._deleted_keys(deleted)

    def test_module_removed_from_moodle_is_still_swept(self, monkeypatch):
        """The protection is scoped to failed modules — real deletions still happen."""
        gone = "moodle:7:9999"
        existing = {self.SUMMARY_KEY: "x", self.MODULE_KEY: "old-hash", gone: "g-hash"}

        _, deleted, _ = self._run(monkeypatch, _boom, existing)

        assert gone in self._deleted_keys(deleted)
        assert self.MODULE_KEY not in self._deleted_keys(deleted)


class TestDrupalCourseType:
    @pytest.mark.parametrize(
        "short_type,expected",
        [
            ("mooc", "Online-Kurse & MOOCs"),
            ("blended", "Blended Learning"),
            ("micro", "Micro Content"),
            ("podcast", "Podcasts"),
            ("video", "Lernvideos"),
            ("paths", "Lernpfade"),
        ],
    )
    def test_known_formats_still_map_to_labels(self, short_type, expected):
        assert Drupal().get_course_type(short_type) == expected

    def test_unknown_format_falls_back_instead_of_raising(self, caplog):
        """A new course format in the Drupal CMS is an editorial change.

        It must not take down the Drupal stage — get_page_type() iterates nodes
        without any error handling, so a raise here fails the entire run.
        """
        with caplog.at_level("WARNING", logger="loader"):
            assert Drupal().get_course_type("selfpaced") == "selfpaced"

        assert "UNKNOWN_COURSE_TYPE" in caplog.text
