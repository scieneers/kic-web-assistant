import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import time
import unicodedata
import zipfile
from pathlib import Path
from typing import Iterable, Optional

import requests

from bs4 import BeautifulSoup, ParserRejectedMarkup
from llama_index.core import Document
from pydantic import ValidationError
from tqdm import tqdm

from src.env import env
from src.loaders.APICaller import APICaller, MoodleMaintenanceError
from src.loaders.failed_transcripts import (
    FailedCourse,
    FailedModule,
    FailedTranscripts,
    save_failed_transcripts_to_excel,
)
from src.loaders.models.book import Book, BookChapter
from src.loaders.models.coursetopic import CourseTopic
from src.loaders.models.folder import Folder
from src.loaders.models.glossary import Glossary, GlossaryEntry
from src.loaders.models.hp5activities import H5PActivities
from src.loaders.models.module import ModuleTypes
from src.loaders.models.h5pactivities.h5p_base import get_handler_for_library, initialize_registry
from src.loaders.models.moodlecourse import MoodleCourse
from src.loaders.models.resource import Resource
from src.loaders.models.url import UrlModule
from src.loaders.models.videotime import Video, VideoPlatforms
from src.loaders.audio import Audio
from src.loaders.pdf import PDF
from src.loaders.vimeo import Vimeo
from src.loaders.youtube import Youtube
from src.loaders.run_logger import RunContext, format_kv
from src.loaders.helper import normalize_text_for_rag


class Moodle:
    """class for Moodle API clients
    base_url: base url of moodle instance
    token: token for moodle api"""
    
    def __init__(self, run_ctx: Optional[RunContext] = None) -> None:
        self.logger = logging.getLogger("loader")
        self.run_ctx = run_ctx
        self._extract_started_ts = time.time()
        self.base_url = env.DATA_SOURCE_MOODLE_URL
        self.api_endpoint = f"{self.base_url}webservice/rest/server.php"
        self.token = env.DATA_SOURCE_MOODLE_TOKEN
        self.function_params = {
            "wstoken": self.token,
            "moodlewsrestformat": "json",
        }
        self.download_params = {
            "token": self.token,
            "forcedownload": 1,
        }
        # Cache für Module-Intros: {coursemodule_id: intro_html}
        # Verschiedene Modultypen können intro haben, wird pro Kurs geladen
        self.module_intros_cache = {}

    def get_courses(self) -> list[MoodleCourse]:
        """get all courses that are set to visible on moodle"""
        caller = APICaller(
            url=self.api_endpoint,
            params=self.function_params,
            wsfunction="core_course_get_courses",
        )
        courses = caller.getJSON()
        course_url = self.base_url + "course/view.php?id="
        courses = [MoodleCourse(url=course_url, **course) for course in courses if course["visible"] == 1]

        # Testing-only knob: cap the number of courses ingested. Unset (<=0) means all.
        limit = int(os.getenv("MOODLE_COURSE_LIMIT", "0"))
        if limit > 0:
            self.logger.warning(
                "MOODLE_COURSE_LIMIT=%s set: ingesting only the first %s of %s visible courses",
                limit,
                limit,
                len(courses),
            )
            courses = courses[:limit]

        return courses

    def get_course_contents(self, course_id: int) -> list[CourseTopic]:
        """get all contents/topics/modules of a course"""
        course_contents_caller = APICaller(
            url=self.api_endpoint,
            params=self.function_params,
            wsfunction="core_course_get_contents",
            courseid=course_id,
        )
        course_contents = course_contents_caller.getJSON()
        course_contents = [CourseTopic(**topic) for topic in course_contents]
        return course_contents

    def get_h5p_module_ids(self, course_id: int) -> list[H5PActivities]:
        h5p_activities: list[H5PActivities] = []
        h5p_module_ids_caller = APICaller(
            url=self.api_endpoint,
            params={**self.function_params, "courseids[0]": course_id},
            wsfunction="mod_h5pactivity_get_h5pactivities_by_courses",
        )
        try:
            ids_json = h5p_module_ids_caller.getJSON()
        except MoodleMaintenanceError:
            # Whole site is down — abort the run, don't treat as "no H5P activities".
            raise
        except Exception as e:
            # Moodle core bug: when an H5P activity has a missing/orphaned backing
            # file, the webservice raises a server-side PHP error
            # ("Call to a member function get_pathnamehash() on bool") for the
            # *whole* course. We can't fix Moodle, so treat this course as having
            # no H5P activities instead of aborting the entire ingestion run.
            self.logger.warning(
                "Moodle: get_h5p_module_ids failed for course_id=%s, skipping H5P activities: %s",
                course_id,
                e,
            )
            return []
        h5p_activities = [H5PActivities(**activity) for activity in ids_json["h5pactivities"]]
        return h5p_activities

    def get_videotime_content(self, cmid: int):
        caller = APICaller(
            url=self.api_endpoint,
            params=self.function_params,
            wsfunction="mod_videotime_get_videotime",
            cmid=cmid,
        )
        videotime_content = caller.getJSON()
        return videotime_content

    def extract(self) -> list[Document]:
        """Extracts all Moodle courses and returns a full list of documents.

        NOTE: This is memory heavy. Prefer `iter_course_documents()` for ingestion.
        """
        docs: list[Document] = []
        courses: list[MoodleCourse] = []
        for course, course_docs in self.iter_course_documents():
            courses.append(course)
            docs.extend(course_docs)
        docs.append(self.get_toc_document(courses))
        return docs


    def iter_course_documents_stream(self) -> Iterable[tuple[MoodleCourse, Document]]:
        """Yield Moodle documents as a stream: (course_obj, doc).

        This is a bounded-memory iterator designed for ingestion:
        - yields the course-level Document first
        - then yields one Document per module

        The caller can delete old documents once per course and then upsert
        documents incrementally.
        """

        failedTranscripts: FailedTranscripts = FailedTranscripts(courses=[])

        courses = self.get_courses()
        self.logger.info("Moodle: fetched %s visible courses", len(courses))
        if self.run_ctx:
            self.run_ctx.set_counter("moodle_courses_total", len(courses))
            self.run_ctx.set_counter("moodle_courses_done", 0)
            self.run_ctx.set_counter("moodle_topics_done", 0)
            self.run_ctx.set_counter("moodle_modules_done", 0)
            self.run_ctx.set_counter("moodle_modules_failed", 0)

        progress_every = int(os.getenv("RUN_MOODLE_PROGRESS_EVERY", "50"))

        # Fortschrittsbalken über alle Kurse. Zählt genau einmal pro Kurs weiter
        # (nicht pro yield), da pro Kurs mehrere Dokumente erzeugt werden.
        course_progress = tqdm(courses, desc="Moodle: Kurse", unit="Kurs")
        for i, course in enumerate(course_progress):
            course_progress.set_postfix_str(f"id={course.id} ({i + 1}/{len(courses)})")
            self.logger.debug("Processing course id=%s (%s/%s)", course.id, i + 1, len(courses))

            # Load the lightweight course structure first.
            course.topics = self.get_course_contents(course.id)
            self.logger.info("Moodle: course_id=%s topics=%s", course.id, len(course.topics))

            if self.run_ctx:
                self.run_ctx.set_last(course_id=course.id)
                self.run_ctx.inc("moodle_courses_done")
                self.run_ctx.checkpoint()
                self.logger.info(
                    "CHECKPOINT %s",
                    format_kv(
                        RUN_ID=self.run_ctx.run_id,
                        STAGE="MOODLE",
                        EVENT="COURSE_START",
                        COURSE_ID=course.id,
                        COURSE_INDEX=f"{i+1}/{len(courses)}",
                    ),
                )

            # Refresh per-course intro cache (avoid unbounded growth)
            self.module_intros_cache = {}
            self._load_module_intros_for_course(course.id)

            h5p_activity_ids = self.get_h5p_module_ids(course.id)
            self.logger.info("Moodle: course_id=%s h5p_activities=%s", course.id, len(h5p_activity_ids))

            # Yield course document (summary) first.
            # NOTE: MoodleCourse.__str__ can include topics, but CourseTopic.__str__ does NOT
            # include modules, so this stays reasonably small.
            course_doc_md = {
                "course_id": course.id,
                "shortname": course.shortname,
                "fullname": course.fullname,
                "type": "Kurs",
                "source": "Moodle",
                "url": course.url,
                **({"language": course.lang} if getattr(course, "lang", None) else {}),
            }
            yield course, Document(text=str(course), metadata=course_doc_md)

            # Process each topic and module (this fills module objects with extracted content)
            for topic in course.topics:
                if self.run_ctx:
                    self.run_ctx.set_last(topic_id=getattr(topic, "id", None))
                    self.run_ctx.inc("moodle_topics_done")
                    self.run_ctx.checkpoint()
                failed_modules = self.get_module_contents(topic, h5p_activity_ids)
                if failed_modules:
                    failedTranscripts.courses.append(FailedCourse(course=course, modules=failed_modules))

                # Periodic progress log
                if self.run_ctx:
                    modules_done = int(self.run_ctx.snapshot().get("counters", {}).get("moodle_modules_done", 0))
                    if progress_every > 0 and modules_done > 0 and modules_done % progress_every == 0:
                        elapsed_s = max(1, int(time.time() - self._extract_started_ts))
                        rate = round(modules_done / (elapsed_s / 60), 2)
                        self.logger.info(
                            "PROGRESS %s",
                            format_kv(
                                RUN_ID=self.run_ctx.run_id,
                                STAGE="MOODLE",
                                EVENT="MODULE_PROGRESS",
                                MODULES_DONE=modules_done,
                                COURSES_DONE=self.run_ctx.snapshot().get("counters", {}).get("moodle_courses_done"),
                                ELAPSED_S=elapsed_s,
                                MODULES_PER_MIN=rate,
                                LAST_COURSE_ID=self.run_ctx.snapshot().get("last_course_id"),
                                LAST_MODULE_ID=self.run_ctx.snapshot().get("last_module_id"),
                                LAST_MODULE_TYPE=self.run_ctx.snapshot().get("last_module_type"),
                            ),
                        )

            # Yield module documents one by one
            if course.topics:
                for topic in course.topics:
                    for module in topic.modules or []:
                        if module is None:
                            continue
                        try:
                            doc = module.to_document(course.id)
                            yield course, doc
                        except Exception as e:
                            self.logger.warning(
                                "Failed to build module Document (course_id=%s module_id=%s): %s",
                                course.id,
                                getattr(module, "id", None),
                                e,
                            )

            # Release memory held by this course (modules can contain huge extracted texts)
            try:
                course.topics = []
            except Exception:
                pass

        # Best-effort: write failed transcripts report
        try:
            save_failed_transcripts_to_excel(transcripts=failedTranscripts, file_name="documentation/FailedTranscripts.xlsx")
        except Exception as e:
            self.logger.warning("Failed to write documentation/FailedTranscripts.xlsx: %s", e)


    def iter_course_documents(self) -> Iterable[tuple[MoodleCourse, list[Document]]]:
        """Streaming extractor.

        Yields (course, docs_for_course) so the ingestion pipeline can delete+upsert
        per course and keep peak memory bounded.
        """

        failedTranscripts: FailedTranscripts = FailedTranscripts(courses=[])

        courses = self.get_courses()
        self.logger.info("Moodle: fetched %s visible courses", len(courses))
        if self.run_ctx:
            self.run_ctx.set_counter("moodle_courses_total", len(courses))
            self.run_ctx.set_counter("moodle_courses_done", 0)
            self.run_ctx.set_counter("moodle_topics_done", 0)
            self.run_ctx.set_counter("moodle_modules_done", 0)
            self.run_ctx.set_counter("moodle_modules_failed", 0)

        progress_every = int(os.getenv("RUN_MOODLE_PROGRESS_EVERY", "50"))

        # Fortschrittsbalken über alle Kurse. Zählt genau einmal pro Kurs weiter.
        course_progress = tqdm(courses, desc="Moodle: Kurse", unit="Kurs")
        for i, course in enumerate(course_progress):
            course_progress.set_postfix_str(f"id={course.id} ({i + 1}/{len(courses)})")
            self.logger.debug("Processing course id=%s (%s/%s)", course.id, i + 1, len(courses))

            if self.run_ctx:
                self.run_ctx.set_last(course_id=course.id)
                self.run_ctx.inc("moodle_courses_done")
                self.run_ctx.checkpoint()
                self.logger.info(
                    "CHECKPOINT %s",
                    format_kv(
                        RUN_ID=self.run_ctx.run_id,
                        STAGE="MOODLE",
                        EVENT="COURSE_START",
                        COURSE_ID=course.id,
                        COURSE_INDEX=f"{i+1}/{len(courses)}",
                    ),
                )

            # Refresh per-course intro cache (avoid unbounded growth)
            self.module_intros_cache = {}
            self._load_module_intros_for_course(course.id)

            course.topics = self.get_course_contents(course.id)
            self.logger.info("Moodle: course_id=%s topics=%s", course.id, len(course.topics))

            h5p_activity_ids = self.get_h5p_module_ids(course.id)
            self.logger.info("Moodle: course_id=%s h5p_activities=%s", course.id, len(h5p_activity_ids))

            for topic in course.topics:
                if self.run_ctx:
                    self.run_ctx.set_last(topic_id=getattr(topic, "id", None))
                    self.run_ctx.inc("moodle_topics_done")
                    self.run_ctx.checkpoint()
                    self.logger.debug(
                        "CHECKPOINT %s",
                        format_kv(
                            RUN_ID=self.run_ctx.run_id,
                            STAGE="MOODLE",
                            EVENT="TOPIC_START",
                            COURSE_ID=course.id,
                            TOPIC_ID=getattr(topic, "id", None),
                            MODULES_IN_TOPIC=len(getattr(topic, "modules", []) or []),
                        ),
                    )
                failed_modules = self.get_module_contents(topic, h5p_activity_ids)
                if failed_modules:
                    failedTranscripts.courses.append(FailedCourse(course=course, modules=failed_modules))

                # Periodic progress log
                if self.run_ctx:
                    modules_done = int(self.run_ctx.snapshot().get("counters", {}).get("moodle_modules_done", 0))
                    if progress_every > 0 and modules_done > 0 and modules_done % progress_every == 0:
                        elapsed_s = max(1, int(time.time() - self._extract_started_ts))
                        rate = round(modules_done / (elapsed_s / 60), 2)
                        self.logger.info(
                            "PROGRESS %s",
                            format_kv(
                                RUN_ID=self.run_ctx.run_id,
                                STAGE="MOODLE",
                                EVENT="MODULE_PROGRESS",
                                MODULES_DONE=modules_done,
                                COURSES_DONE=self.run_ctx.snapshot().get("counters", {}).get("moodle_courses_done"),
                                ELAPSED_S=elapsed_s,
                                MODULES_PER_MIN=rate,
                                LAST_COURSE_ID=self.run_ctx.snapshot().get("last_course_id"),
                                LAST_MODULE_ID=self.run_ctx.snapshot().get("last_module_id"),
                                LAST_MODULE_TYPE=self.run_ctx.snapshot().get("last_module_type"),
                            ),
                        )

            # Convert only this course's objects to documents, then yield.
            docs_for_course = course.to_document()
            yield course, docs_for_course

        # Keep current behavior of saving failed transcript report (best-effort)
        try:
            save_failed_transcripts_to_excel(transcripts=failedTranscripts, file_name="documentation/FailedTranscripts.xlsx")
        except Exception as e:
            self.logger.warning("Failed to write documentation/FailedTranscripts.xlsx: %s", e)

    def get_toc_document(self, courses) -> Document:
        toc_str = "List of all available courses:\n"
        for course in courses:
            toc_str += f"Course ID: {course.id}, Course Name: {course.fullname}\n"

        metadata = {
            "fullname": "Table of Contents",
            "type": "toc",
            "url": "https://ki-campus.org/overview/course",
        }

        toc_doc = Document(text=toc_str, metadata=metadata)

        return toc_doc

    def get_module_contents(self, topic, h5p_activities):
        failed_modules: list[FailedModule] = []

        self.logger.debug(
            "Moodle: topic_id=%s modules=%s", getattr(topic, "id", None), len(topic.modules)
        )
        for module in topic.modules:
            err_message = None
            if module.visible == 0:
                continue

            if self.run_ctx:
                self.run_ctx.set_last(
                    module_id=getattr(module, "id", None),
                    module_type=getattr(getattr(module, "type", None), "value", getattr(module, "type", None)),
                    url=getattr(module, "url", None),
                )
                self.run_ctx.checkpoint()
            match module.type:
                case ModuleTypes.VIDEOTIME:
                    err_message = self.extract_videotime(module)
                case ModuleTypes.PAGE:
                    err_message = self.extract_page(module)
                case ModuleTypes.H5P:
                    for activity in h5p_activities:
                        if activity.coursemodule == module.id:
                            err_message = self.extract_h5p(module, activity)
                case ModuleTypes.GLOSSARY:
                    err_message = self.extract_glossary(module)
                case ModuleTypes.RESOURCE:
                    err_message = self.extract_resource(module)
                case ModuleTypes.FOLDER:
                    err_message = self.extract_folder(module)
                case ModuleTypes.BOOK:
                    err_message = self.extract_book(module)
                case ModuleTypes.URL:
                    err_message = self.extract_url(module)
            if err_message:
                failed_modules.append(FailedModule(modul=module, err_message=err_message))
                if self.run_ctx:
                    self.run_ctx.inc("moodle_modules_failed")
            if self.run_ctx:
                self.run_ctx.inc("moodle_modules_done")

        return failed_modules

    def extract_page(self, module):
        err_message = None

        for content in module.contents:
            # A page bundles its HTML body plus any embedded files. Embedded H5P
            # packages get routed through the H5P pipeline; the HTML body is
            # parsed for text and video links; everything else (images, ...) is
            # skipped since it is not parseable as HTML. The validator leaves
            # `type` as None for ".html" files and otherwise stores the file
            # extension (with any "?forcedownload=1" suffix).
            content_type = (content.type or "html").lower()
            if content_type.startswith("h5p"):
                # Eingebettetes H5P-Paket über dieselbe Pipeline wie eigenständige
                # H5P-Module verarbeiten (befüllt module.h5p_content_type,
                # module.interactive_video, ...).
                try:
                    h5p_err = self._process_h5p_package(
                        module, content.fileurl, content.filename
                    )
                except requests.exceptions.HTTPError as err:
                    self.logger.warning(
                        f"Failed to download embedded H5P {content.fileurl}: {err}"
                    )
                    continue
                if h5p_err:
                    err_message = h5p_err
                continue
            if not content_type.startswith("html"):
                continue
            page_content_caller = APICaller(url=content.fileurl, params=self.download_params)
            try:
                soup = BeautifulSoup(page_content_caller.getText(), "html.parser")
            except ParserRejectedMarkup:
                continue
            except requests.exceptions.HTTPError as err:
                # A single missing/forbidden page resource must not abort the
                # whole ingestion run; skip it and keep processing.
                self.logger.warning(f"Failed to download page content {content.fileurl}: {err}")
                continue
            links = soup.find_all("a")

            if soup.text is not None:
                module.text = soup.get_text("\n")
                module.text = normalize_text_for_rag(module.text)

            for p_link in links:
                pattern = r"https://player\.vimeo\.com/video/\d+"
                match = re.search(pattern, str(p_link))
                src = match.group(0) if match else p_link.get("href")
                if src:
                    if src.find("vimeo") != -1:
                        videotime = Video(id=0, vimeo_url=src)
                        if videotime.video_id is None:
                            self.logger.warning(f"Cannot parse video url: {src}")
                            continue
                        vimeo = Vimeo()
                        texttrack, err_message = vimeo.get_transcript(videotime.video_id)
                    elif src.find("youtu") != -1:
                        try:
                            videotime = Video(id=0, vimeo_url=src)
                        except ValidationError:
                            self.logger.warning(f"Cannot parse video url: {src}")
                            continue
                        if videotime.video_id is None:
                            # Link refers not to a specific video, but to a channel overview page
                            texttrack = None
                        else:
                            youtube = Youtube()
                            texttrack, err_message = youtube.get_transcript(videotime.video_id)
                    else:
                        texttrack = None
                    module.transcripts.append(texttrack)
        return err_message if err_message is not None else None

    def extract_videotime(self, module):  # TODO: rename method
        try:
            videotime_data = self.get_videotime_content(module.id)
        except MoodleMaintenanceError:
            # Whole site is down — let this propagate to abort the run.
            raise
        except Exception as e:
            error_msg = f"Fehler beim Laden von VideoTime-Modul {module.id}: {str(e)}"
            self.logger.error(error_msg)
            return error_msg

        # Übertrage intro, falls vorhanden und nicht leer
        if videotime_data.get('intro') and videotime_data['intro'].strip():
            module.intro = videotime_data['intro']

        try:
            videotime = Video(**videotime_data)
        except ValidationError as e:
            self.logger.error(f"Validation error for VideoTime module {module.id}: {e}")
            return

        err_message = None

        if videotime.video_id is None:
            return

        match videotime.type:
            case VideoPlatforms.VIMEO:
                vimeo = Vimeo()
                texttrack, err_message = vimeo.get_transcript(videotime.video_id)
                module.transcripts.append(texttrack)
            case VideoPlatforms.YOUTUBE:
                youtube = Youtube()
                texttrack, err_message = youtube.get_transcript(videotime.video_id)
                module.transcripts.append(texttrack)
            case VideoPlatforms.SELF_HOSTED:
                # found no subtitles in self-hosted videos, if this ever changes add code here
                pass

        return err_message if err_message is not None else None

    # A H5P Module is a zipped bundle of js, css and a content.json, describing the content.
    # If a video is wrapped in a H5P Module we are only interested in the content.json.
    # In the content.json we find the link to the video. Based on that link we can construct
    # the link to the transcript of that video. Then we download that transcript and add it to the
    # module.transcripts list.
    def extract_h5p(self, module, activity):
        """Extrahiert H5P-Inhalte und routet zu entsprechender Klasse."""
        if activity.intro:
            module.intro = activity.intro
        return self._process_h5p_package(module, activity.fileurl, activity.filename)

    def _process_h5p_package(self, module, fileurl, filename):
        """Lädt ein H5P-Paket, ermittelt den Typ und ruft den passenden Handler.

        Geteilte Kernlogik, die sowohl von eigenständigen H5P-Modulen
        (extract_h5p) als auch von in Seiten eingebetteten H5P-Dateien
        (extract_page) genutzt wird. Befüllt das übergebene Modul und gibt eine
        Fehlermeldung zurück (oder None bei Erfolg).
        """
        h5pfile_call = APICaller(url=fileurl, params=self.download_params)

        with tempfile.TemporaryDirectory() as tmp_dir:
            # H5P-Package herunterladen
            local_filename = h5pfile_call.getFile(filename, tmp_dir)

            # h5p.json für library-Feld extrahieren
            try:
                with zipfile.ZipFile(local_filename, "r") as zip_ref:
                    zip_ref.extract("h5p.json", tmp_dir)
                    zip_ref.extract("content/content.json", tmp_dir)
            except zipfile.BadZipFile as exc:
                # Korruptes/kein H5P-Paket (z.B. HTML-Fehlerseite statt ZIP).
                # Modul überspringen statt den ganzen Lauf abzubrechen; lokal das
                # defekte Paket für die Fehleranalyse aufheben.
                self.logger.warning(
                    f"Ungültiges H5P-ZIP für Modul {module.id} ({filename}): {exc}"
                )
                self._save_bad_h5p_zip(local_filename, module, filename)
                return f"Ungültiges H5P-ZIP-Paket: {exc}"

            # Lade h5p.json für library-Informationen
            h5p_json = f"{tmp_dir}/h5p.json"
            with open(h5p_json, "r") as json_file:
                h5p_data = json.load(json_file)
            
            # Lade content.json für eigentliche Inhalte
            content_json = f"{tmp_dir}/content/content.json"
            with open(content_json, "r") as json_file:
                content = json.load(json_file)
            
            # H5P Content-Typ aus h5p.json extrahieren (nicht content.json!)
            library = h5p_data.get("mainLibrary", "") or h5p_data.get("preloadedDependencies", [{}])[0].get("machineName", "")
            if not library:
                self.logger.error(f"Kein library-Feld in h5p.json für Modul {module.id}")
                return "Kein library-Feld in h5p.json gefunden"
            
            module.h5p_content_type = library
            self.logger.info(f"Verarbeite H5P-Typ: {library} für Modul {module.id}")
            
            # Initialisiere Registry beim ersten Aufruf
            initialize_registry()
            
            # Finde passenden Handler via Registry
            handler_class = get_handler_for_library(library)
            
            if not handler_class:
                self.logger.warning(f"H5P-Typ '{library}' wird noch nicht unterstützt (Modul {module.id})")
                return f"H5P-Typ '{library}' wird noch nicht unterstützt"
            
            self.logger.info(f"Rufe Handler {handler_class.__name__} auf für Modul {module.id}")
            
            # Rufe from_h5p_package direkt auf (befüllt module und gibt error zurück)
            err = handler_class.from_h5p_package(
                module=module,
                content=content,
                h5p_zip_path=local_filename,
                vimeo_service=Vimeo(),
                video_service=Video
            )
            
            if err:
                self.logger.error(f"Fehler beim Verarbeiten von Modul {module.id}: {err}")
            else:
                self.logger.info(f"Modul {module.id} erfolgreich verarbeitet")
            
            return err

        return None

    @staticmethod
    def _is_local_run() -> bool:
        """True, wenn nicht in Azure (App Service/Functions) ausgeführt.

        Azure injiziert WEBSITE_INSTANCE_ID auf jeder Instanz; lokal ist die
        Variable nie gesetzt. So bleibt das Aufheben defekter Pakete eine reine
        Entwickler-Hilfe und produziert in der Cloud keine verwaisten Dateien.
        """
        return not os.getenv("WEBSITE_INSTANCE_ID")

    def _save_bad_h5p_zip(self, local_filename: str, module, filename) -> None:
        """Kopiert ein defektes H5P-ZIP zur Fehleranalyse beiseite (nur lokal).

        Muss aufgerufen werden, solange das umgebende TemporaryDirectory noch
        existiert, da local_filename sonst bereits gelöscht ist.
        """
        if not self._is_local_run():
            return
        try:
            dest_dir = Path(os.getenv("H5P_BAD_ZIP_DIR", "bad_h5p_zips"))
            dest_dir.mkdir(parents=True, exist_ok=True)
            # Modul-ID voranstellen, damit der Dateiname eindeutig und auffindbar ist.
            safe_name = os.path.basename(filename or "unknown.h5p")
            dest = dest_dir / f"{module.id}_{safe_name}"
            shutil.copyfile(local_filename, dest)
            self.logger.info(f"Defektes H5P-Paket gespeichert unter {dest}")
        except Exception as exc:
            # Das Aufheben ist Best-Effort und darf den Lauf nie beeinflussen.
            self.logger.warning(f"Konnte defektes H5P-Paket nicht speichern: {exc}")

    def extract_glossary(self, module):
        """
        Extrahiert Glossar-Einträge aus einem Moodle Glossary Modul.
        
        Verwendet module.instance (glossary_id) aus core_course_get_contents.
        
        Args:
            module: Module-Objekt mit modname="glossary" und instance=glossary_id
            
        Returns:
            Optional[str]: Fehlermeldung oder None
        """
        try:
            # module.instance enthält bereits die glossary_id von core_course_get_contents
            glossary_id = module.instance
            
            # API-Call für alle Glossar-Einträge
            caller = APICaller(
                url=self.api_endpoint,
                params=self.function_params,
                wsfunction="mod_glossary_get_entries_by_letter",
                id=glossary_id,
                letter="ALL",  # Alle Einträge
                **{"from": 0, "limit": 1000}  # Max 1000 Einträge
            )
            
            response = caller.getJSON()
            
            # Parse Response
            entries_data = response.get("entries", [])
            total_count = response.get("count", 0)
            
            if not entries_data:
                self.logger.info(f"Glossary {glossary_id} (Module {module.id}) hat keine Einträge")
                return None
            
            # Erstelle GlossaryEntry Objekte (nur concept + definition relevant)
            entries = []
            for entry_data in entries_data:
                try:
                    entry = GlossaryEntry(
                        id=entry_data["id"],
                        concept=entry_data["concept"],
                        definition=entry_data["definition"]
                    )
                    entries.append(entry)
                except Exception as e:
                    self.logger.warning(f"Fehler beim Parsen von Glossary Entry {entry_data.get('id')}: {e}")
                    continue
            
            # Erstelle Glossary-Objekt und befülle Module
            module.glossary = Glossary(
                glossary_id=glossary_id,
                module_id=module.id,
                entries=entries
            )
            
            self.logger.info(f"Glossary {glossary_id} erfolgreich geladen: {len(entries)}/{total_count} Einträge")
            return None

        except MoodleMaintenanceError:
            # Whole site is down — abort the run instead of marking this module failed.
            raise
        except Exception as e:
            error_msg = f"Fehler beim Laden von Glossary {module.id}: {str(e)}"
            self.logger.error(error_msg)
            return error_msg

    def extract_resource(self, module):
        """
        Extrahiert Inhalte aus einem Resource-Modul.
        
        Kann mehrere Dateien enthalten - alle werden verarbeitet und kombiniert.
        
        Args:
            module: Module-Objekt mit modname="resource" und contents mit Dateien
            
        Returns:
            Optional[str]: Fehlermeldung oder None
        """
        try:
            if not module.contents or len(module.contents) == 0:
                self.logger.warning(f"Resource-Modul {module.id} hat keine Dateiinhalte")
                return None
            
            # Hole Intro-Text aus Cache (wurde pro Kurs geladen)
            intro_html = self.module_intros_cache.get(module.id, "")
            self.logger.info(
                f"Intro für Modul {module.id}: "
                f"{'GEFUNDEN (' + str(len(intro_html)) + ' Zeichen)' if intro_html else 'NICHT IM CACHE'}"
            )
            intro_text = self._extract_intro_text(intro_html, module.id)
            
            # Sammle extrahierte Texte aller Dateien
            all_extracted_texts = []
            if intro_text:
                all_extracted_texts.append(f"Intro:\n{intro_text}")
            
            # Verarbeite jede Datei im Resource-Modul
            for idx, file_content in enumerate(module.contents):
                self.logger.info(
                    f"Verarbeite Datei {idx+1}/{len(module.contents)} in Resource-Modul {module.id}: "
                    f"{file_content.filename}"
                )
                
                # Bereinige mimetype (entferne Query-Parameter wie "?forcedownload=1")
                mimetype = file_content.type.split('?')[0] if file_content.type else None
                if not mimetype:
                    self.logger.warning(f"Datei {file_content.filename} hat keinen Dateityp")
                    continue
                
                # Erstelle Resource-Objekt
                resource = Resource(
                    filename=file_content.filename,
                    fileurl=file_content.fileurl,
                    mimetype=mimetype,
                    filesize=file_content.filesize if hasattr(file_content, 'filesize') else None
                )
                
                # Prüfe Unterstützung
                if not resource.is_supported:
                    self.logger.info(
                        f"Datei {resource.filename} hat nicht-unterstützten Dateityp: {resource.mimetype}"
                    )
                    all_extracted_texts.append(f"Datei {resource.filename} nicht unterstützt und daher nicht verarbeitet.")
                    continue
                
                # Download Datei
                caller = APICaller(url=file_content.fileurl, params=self.download_params)
                caller.get()
                file_bytes = caller.response.content
                
                if not file_bytes:
                    self.logger.error(f"Konnte {resource.filename} nicht herunterladen")
                    continue
                
                # Extrahiere Text (Logik in Resource-Klasse)
                extracted_text = resource.extract_from_bytes(file_bytes, self.logger)
                
                if extracted_text:
                    # Bei mehreren Dateien: Header hinzufügen
                    if len(module.contents) > 1:
                        all_extracted_texts.append(f"\n--- Datei: {resource.filename} ---\n{extracted_text}")
                    else:
                        all_extracted_texts.append(extracted_text)
            
            # Kombiniere alle Texte und speichere in module.resource
            if all_extracted_texts:
                combined_resource = Resource(
                    filename=f"{len(module.contents)} Dateien" if len(module.contents) > 1 else module.contents[0].filename,
                    fileurl=module.contents[0].fileurl,
                    mimetype="multi" if len(module.contents) > 1 else module.contents[0].type.split('?')[0],
                    extracted_text='\n'.join(all_extracted_texts)
                )
                module.resource = combined_resource
                self.logger.info(
                    f"Resource-Modul {module.id} erfolgreich verarbeitet: "
                    f"{len(module.contents)} Datei(en), {len(combined_resource.extracted_text)} Zeichen gesamt"
                )
            
            return None
            
        except Exception as e:
            error_msg = f"Fehler beim Laden von Resource {module.id}: {str(e)}"
            self.logger.error(error_msg)
            return error_msg
    
    def extract_folder(self, module):
        """
        Extrahiert Inhalte aus einem Folder-Modul.
        
        Folder-Module enthalten mehrere Dateien - alle werden verarbeitet und kombiniert.
        Funktioniert analog zu extract_resource(), da beide ein contents-Array haben.
        
        Args:
            module: Module-Objekt mit modname="folder" und contents mit Dateien
            
        Returns:
            Optional[str]: Fehlermeldung oder None
        """
        try:
            if not module.contents or len(module.contents) == 0:
                self.logger.warning(f"Folder-Modul {module.id} hat keine Dateiinhalte")
                return None
            
            # Hole Intro-Text aus Cache (wurde pro Kurs geladen)
            intro_html = self.module_intros_cache.get(module.id, "")
            self.logger.info(
                f"Intro für Folder-Modul {module.id}: "
                f"{'GEFUNDEN (' + str(len(intro_html)) + ' Zeichen)' if intro_html else 'NICHT IM CACHE'}"
            )
            intro_text = self._extract_intro_text(intro_html, module.id)
            
            # Sammle extrahierte Dateien
            extracted_files: list[Resource] = []
            all_extracted_texts = []
            
            if intro_text:
                all_extracted_texts.append(intro_text)
            
            # Verarbeite jede Datei im Folder
            for idx, file_content in enumerate(module.contents):
                self.logger.info(
                    f"Verarbeite Datei {idx+1}/{len(module.contents)} in Folder-Modul {module.id}: "
                    f"{file_content.filename}"
                )
                
                # Bereinige mimetype (entferne Query-Parameter wie "?forcedownload=1")
                mimetype = file_content.type.split('?')[0] if file_content.type else None
                if not mimetype:
                    self.logger.warning(f"Datei {file_content.filename} hat keinen Dateityp")
                    continue
                
                # Erstelle Resource-Objekt
                resource = Resource(
                    filename=file_content.filename,
                    fileurl=file_content.fileurl,
                    mimetype=mimetype,
                    filesize=file_content.filesize if hasattr(file_content, 'filesize') else None
                )
                
                # Prüfe Unterstützung (PDF, Audio, ZIP, TXT, HTML)
                if not resource.is_supported:
                    self.logger.info(
                        f"Datei {resource.filename} hat nicht-unterstützten Dateityp: {resource.mimetype}"
                    )
                    continue
                
                # Download Datei
                caller = APICaller(url=file_content.fileurl, params=self.download_params)
                caller.get()
                file_bytes = caller.response.content
                
                if not file_bytes:
                    self.logger.error(f"Konnte {resource.filename} nicht herunterladen")
                    continue
                
                # Extrahiere Text (Logik in Resource-Klasse)
                extracted_text = resource.extract_from_bytes(file_bytes, self.logger)
                
                if extracted_text:
                    # Speichere Resource mit extrahiertem Text
                    resource.extracted_text = extracted_text
                    extracted_files.append(resource)
                    
                    # Bei mehreren Dateien: Header hinzufügen
                    if len(module.contents) > 1:
                        all_extracted_texts.append(f"\n--- Datei: {resource.filename} ---\n{extracted_text}")
                    else:
                        all_extracted_texts.append(extracted_text)
            
            # Erstelle Folder-Objekt und speichere in module.folder
            if extracted_files:
                combined_folder = Folder(
                    folder_id=module.instance,
                    module_id=module.id,
                    files=extracted_files,
                    combined_text='\n'.join(all_extracted_texts)
                )
                module.folder = combined_folder
                self.logger.info(
                    f"Folder-Modul {module.id} erfolgreich verarbeitet: "
                    f"{len(extracted_files)} Datei(en), {combined_folder.total_extracted_chars} Zeichen gesamt"
                )
            
            return None
            
        except Exception as e:
            error_msg = f"Fehler beim Laden von Folder {module.id}: {str(e)}"
            self.logger.error(error_msg)
            return error_msg
    
    def extract_book(self, module):
        """
        Extrahiert Inhalte aus einem Book-Modul.
        
        Books enthalten Kapitel mit HTML-Inhalten, Videos und Anhängen.
        Verwendet die gleiche Logik wie extract_page() für HTML-Parsing und Video-Extraktion.
        
        Args:
            module: Module-Objekt mit modname="book" und contents mit Kapiteln
            
        Returns:
            Optional[str]: Fehlermeldung oder None
        """
        try:
            if not module.contents or len(module.contents) == 0:
                self.logger.warning(f"Book-Modul {module.id} hat keine Inhalte")
                return None
            
            # Hole Intro-Text aus Cache (wurde pro Kurs geladen)
            intro_html = self.module_intros_cache.get(module.id, "")
            self.logger.info(
                f"Intro für Book-Modul {module.id}: "
                f"{'GEFUNDEN (' + str(len(intro_html)) + ' Zeichen)' if intro_html else 'NICHT IM CACHE'}"
            )
            intro_text = self._extract_intro_text(intro_html, module.id)
            
            # Gruppiere contents nach Kapiteln (filepath)
            # Format: /287/ -> Kapitel 287
            chapters_data = {}
            structure_json = None
            
            for content in module.contents:
                # Structure-Content (JSON mit Kapitel-Hierarchie)
                if content.type == "content" and content.filename == "structure":
                    try:
                        structure_json = json.loads(content.content) if hasattr(content, 'content') else None
                    except json.JSONDecodeError:
                        self.logger.warning(f"Konnte structure.json für Book {module.id} nicht parsen")
                    continue
                
                # Ignoriere Dateien ohne fileurl
                if not content.fileurl:
                    continue
                
                # Extrahiere chapter_id aus fileurl
                # Format: https://moodle.ki-campus.org/webservice/pluginfile.php/40628/mod_book/chapter/287/index.html
                # -> chapter_id = "287"
                import re
                chapter_match = re.search(r'/chapter/(\d+)/', content.fileurl)
                if not chapter_match:
                    self.logger.debug(f"Konnte chapter_id nicht aus URL extrahieren: {content.fileurl}")
                    continue
                
                chapter_id = chapter_match.group(1)
                
                # Initialisiere Kapitel-Dict
                if chapter_id not in chapters_data:
                    chapters_data[chapter_id] = {
                        'title': None,
                        'html_url': None,
                        'html_content': None,
                        'attachments': []
                    }
                
                # index.html = Kapitel-Inhalt
                if content.filename == "index.html":
                    chapters_data[chapter_id]['html_url'] = content.fileurl
                    # content-Feld enthält oft den Titel
                    if hasattr(content, 'content') and content.content:
                        chapters_data[chapter_id]['title'] = content.content
                # Andere Dateien = Anhänge (PDFs, ZIPs, etc.)
                elif content.filename != "index.html":
                    chapters_data[chapter_id]['attachments'].append(content)
            
            self.logger.info(f"Book {module.id}: {len(chapters_data)} Kapitel gefunden")
            
            # Verarbeite jedes Kapitel
            book_chapters = []
            for chapter_id, data in sorted(chapters_data.items()):
                chapter_title = data['title'] or f"Kapitel {chapter_id}"
                self.logger.info(f"Verarbeite Kapitel '{chapter_title}' (ID: {chapter_id})")
                
                # Erstelle Chapter-Objekt
                chapter = BookChapter(
                    chapter_id=chapter_id,
                    title=chapter_title
                )
                
                # Extrahiere HTML-Text und Videos (wie bei PAGE)
                if data['html_url']:
                    html_text, transcripts, err = self._extract_html_content(data['html_url'], module.id)
                    chapter.html_text = html_text
                    chapter.transcripts = transcripts
                    
                    if err:
                        self.logger.warning(f"Fehler beim HTML-Parsing in Kapitel {chapter_id}: {err}")
                
                # Verarbeite Anhänge (PDFs, ZIPs, etc.)
                for attachment_content in data['attachments']:
                    # Bereinige mimetype
                    mimetype = attachment_content.type.split('?')[0] if attachment_content.type else None
                    if not mimetype:
                        continue
                    
                    # Erstelle Resource-Objekt
                    resource = Resource(
                        filename=attachment_content.filename,
                        fileurl=attachment_content.fileurl,
                        mimetype=mimetype,
                        filesize=attachment_content.filesize if hasattr(attachment_content, 'filesize') else None
                    )
                    
                    # Nur unterstützte Formate verarbeiten
                    if not resource.is_supported:
                        self.logger.debug(f"Anhang {resource.filename} hat nicht-unterstützten Typ: {resource.mimetype}")
                        continue
                    
                    # Download und Extraktion
                    try:
                        caller = APICaller(url=attachment_content.fileurl, params=self.download_params)
                        caller.get()
                        file_bytes = caller.response.content
                        
                        if file_bytes:
                            extracted_text = resource.extract_from_bytes(file_bytes, self.logger)
                            if extracted_text:
                                resource.extracted_text = extracted_text
                                chapter.attachments.append(resource)
                    except Exception as e:
                        self.logger.warning(f"Fehler beim Download von {resource.filename}: {e}")
                
                book_chapters.append(chapter)
            
            # Erstelle Book-Objekt
            if book_chapters:
                book = Book(
                    book_id=module.instance,
                    module_id=module.id,
                    intro=intro_text if intro_text else None,
                    chapters=book_chapters,
                    structure=structure_json
                )
                module.book = book
                
                self.logger.info(
                    f"Book-Modul {module.id} erfolgreich verarbeitet: "
                    f"{book.total_chapters} Kapitel, {book.total_videos} Videos, {book.total_attachments} Anhänge"
                )
            
            return None
            
        except Exception as e:
            error_msg = f"Fehler beim Laden von Book {module.id}: {str(e)}"
            self.logger.error(error_msg)
            return error_msg
    
    def _extract_html_content(self, html_url: str, module_id: int) -> tuple[str | None, list, str | None]:
        """
        Extrahiert HTML-Text und Videos aus einer HTML-URL.
        
        Verwendet die gleiche Logik wie extract_page() für HTML-Parsing und Video-Extraktion.
        
        Args:
            html_url: URL der HTML-Datei
            module_id: Module-ID für Logging
            
        Returns:
            tuple: (html_text, transcripts, error_message)
        """
        err_message = None
        transcripts = []
        html_text = None
        
        try:
            page_content_caller = APICaller(url=html_url, params=self.download_params)
            soup = BeautifulSoup(page_content_caller.getText(), "html.parser")
            
            # Extrahiere Text
            if soup.text is not None:
                html_text = soup.get_text("\n")
                html_text = normalize_text_for_rag(html_text)
            
            # Extrahiere Videos aus Links (wie bei PAGE)
            links = soup.find_all("a")
            for p_link in links:
                pattern = r"https://player\.vimeo\.com/video/\d+"
                match = re.search(pattern, str(p_link))
                src = match.group(0) if match else p_link.get("href")
                
                if src:
                    if src.find("vimeo") != -1:
                        videotime = Video(id=0, vimeo_url=src)
                        if videotime.video_id is None:
                            self.logger.warning(f"Cannot parse video url: {src}")
                            continue
                        vimeo = Vimeo()
                        texttrack, err_message = vimeo.get_transcript(videotime.video_id)
                        transcripts.append(texttrack)
                    elif src.find("youtu") != -1:
                        try:
                            videotime = Video(id=0, vimeo_url=src)
                        except ValidationError:
                            self.logger.warning(f"Cannot parse video url: {src}")
                            continue
                        if videotime.video_id is None:
                            # Link refers not to a specific video, but to a channel overview page
                            continue
                        else:
                            youtube = Youtube()
                            texttrack, err_message = youtube.get_transcript(videotime.video_id)
                            transcripts.append(texttrack)
        
        except ParserRejectedMarkup:
            self.logger.warning(f"HTML-Parser rejected markup für Modul {module_id}")
        except Exception as e:
            err_message = f"Fehler beim HTML-Parsing: {str(e)}"
            self.logger.error(err_message)
        
        return html_text, transcripts, err_message
    
    def _load_module_intros_for_course(self, course_id: int):
        """
        Lädt alle Module-Intros für einen Kurs in den Cache.
        
        core_course_get_contents gibt das intro-Feld nicht zurück, daher müssen wir
        die modulspezifischen APIs nutzen (z.B. mod_resource_get_resources_by_courses).
        
        Aktuell werden nur Resource-Intros geladen. Kann erweitert werden für:
        - mod_glossary_get_glossaries_by_courses
        - mod_page_get_pages_by_courses
        - etc.
        
        Args:
            course_id: ID des Kurses
        """
        try:
            # Lade alle Resource-Module des Kurses mit ihren Intros
            self.logger.info(f"Lade Resource-Intros für Kurs {course_id}...")
            caller = APICaller(
                url=self.api_endpoint,
                params={**self.function_params, "courseids[0]": course_id},
                wsfunction="mod_resource_get_resources_by_courses"
            )
            response = caller.getJSON()
            resources = response.get("resources", [])
            
            self.logger.info(f"API-Antwort: {len(resources)} Resources gefunden")
            
            # Speichere intro pro coursemodule_id im Cache
            for resource in resources:
                coursemodule_id = resource.get("coursemodule")
                intro_html = resource.get("intro", "")
                if coursemodule_id:
                    self.module_intros_cache[coursemodule_id] = intro_html
                    self.logger.debug(
                        f"  - Modul {coursemodule_id}: Intro mit {len(intro_html)} Zeichen"
                    )
            
            self.logger.info(f"Kurs {course_id}: {len(resources)} Resource-Intros in Cache gespeichert")
            self.logger.debug(f"Cache enthält jetzt {len(self.module_intros_cache)} Einträge gesamt")
            
            # Lade Folder-Intros
            self.logger.info(f"Lade Folder-Intros für Kurs {course_id}...")
            folder_caller = APICaller(
                url=self.api_endpoint,
                params={**self.function_params, "courseids[0]": course_id},
                wsfunction="mod_folder_get_folders_by_courses"
            )
            folder_response = folder_caller.getJSON()
            folders = folder_response.get("folders", [])
            
            for folder in folders:
                coursemodule_id = folder.get("coursemodule")
                intro_html = folder.get("intro", "")
                if coursemodule_id:
                    self.module_intros_cache[coursemodule_id] = intro_html
                    self.logger.debug(
                        f"  - Modul {coursemodule_id}: Intro mit {len(intro_html)} Zeichen"
                    )
            
            self.logger.info(f"Kurs {course_id}: {len(folders)} Folder-Intros in Cache gespeichert")
            self.logger.debug(f"Cache enthält jetzt {len(self.module_intros_cache)} Einträge gesamt")
            
            # Lade Book-Intros
            self.logger.info(f"Lade Book-Intros für Kurs {course_id}...")
            book_caller = APICaller(
                url=self.api_endpoint,
                params={**self.function_params, "courseids[0]": course_id},
                wsfunction="mod_book_get_books_by_courses"
            )
            book_response = book_caller.getJSON()
            books = book_response.get("books", [])
            
            for book in books:
                coursemodule_id = book.get("coursemodule")
                intro_html = book.get("intro", "")
                if coursemodule_id:
                    self.module_intros_cache[coursemodule_id] = intro_html
                    self.logger.debug(
                        f"  - Modul {coursemodule_id}: Intro mit {len(intro_html)} Zeichen"
                    )
            
            self.logger.info(f"Kurs {course_id}: {len(books)} Book-Intros in Cache gespeichert")
            self.logger.debug(f"Cache enthält jetzt {len(self.module_intros_cache)} Einträge gesamt")
            
            # Lade URL-Intros
            self.logger.info(f"Lade URL-Intros für Kurs {course_id}...")
            url_caller = APICaller(
                url=self.api_endpoint,
                params={**self.function_params, "courseids[0]": course_id},
                wsfunction="mod_url_get_urls_by_courses"
            )
            url_response = url_caller.getJSON()
            urls = url_response.get("urls", [])
            
            for url in urls:
                coursemodule_id = url.get("coursemodule")
                intro_html = url.get("intro", "")
                if coursemodule_id:
                    self.module_intros_cache[coursemodule_id] = intro_html
                    self.logger.debug(
                        f"  - Modul {coursemodule_id}: Intro mit {len(intro_html)} Zeichen"
                    )
            
            self.logger.info(f"Kurs {course_id}: {len(urls)} URL-Intros in Cache gespeichert")
            self.logger.debug(f"Cache enthält jetzt {len(self.module_intros_cache)} Einträge gesamt")
            
            # Hier können weitere Modultypen hinzugefügt werden:
            # - mod_glossary_get_glossaries_by_courses
            # - mod_page_get_pages_by_courses
            # etc.

        except MoodleMaintenanceError:
            # Whole site is down — abort the run instead of continuing with empty intros.
            raise
        except Exception as e:
            self.logger.warning(f"Fehler beim Laden der Module-Intros für Kurs {course_id}: {e}")
    
    def _extract_intro_text(self, intro_html: str | None, module_id: int) -> str:
        """
        Extrahiert und bereinigt Intro-Text aus HTML.
        
        Args:
            intro_html: HTML-String mit Intro-Text
            module_id: Module-ID für Logging
            
        Returns:
            str: Bereinigter Text oder leerer String
        """
        if not intro_html:
            return ""
        
        from bs4 import BeautifulSoup
        try:
            soup = BeautifulSoup(intro_html, "html.parser")
            text = soup.get_text("\n")
            text = normalize_text_for_rag(text)
            return text + "\n\n" if text else ""
        except Exception as e:
            self.logger.warning(f"Fehler beim Parsen des Intro-Texts für Modul {module_id}: {e}")
            return ""
    
    def extract_url(self, module):
        """
        Extrahiert Inhalte aus einem URL-Modul.
        
        URL-Module können auf externe Websites oder verarbeitbare Dateien verweisen.
        - Wenn die URL auf eine verarbeitbare Datei (PDF, HTML, Audio, TXT) verweist,
          wird diese heruntergeladen und als Resource extrahiert.
        - Andernfalls wird nur der Link mit Intro-Text gespeichert.
        
        Args:
            module: Module-Objekt mit modname="url" und instance=url_id
            
        Returns:
            Optional[str]: Fehlermeldung oder None
        """
        try:
            # Hole externe URL aus contents (von core_course_get_contents)
            if not module.contents or len(module.contents) == 0:
                self.logger.warning(f"URL-Modul {module.id} hat keine URL in contents")
                return None
            
            external_url = module.contents[0].fileurl  # Die tatsächliche externe URL
            
            # Hole Intro-Text aus Cache (wurde pro Kurs geladen)
            intro_html = self.module_intros_cache.get(module.id, "")
            self.logger.info(
                f"Intro für URL-Modul {module.id}: "
                f"{'GEFUNDEN (' + str(len(intro_html)) + ' Zeichen)' if intro_html else 'NICHT IM CACHE'}"
            )
            intro_text = self._extract_intro_text(intro_html, module.id)
            
            # Prüfe, ob die URL auf eine verarbeitbare Datei verweist
            file_extension = self._get_url_file_extension(external_url)
            
            if file_extension in ['pdf', 'html', 'htm', 'wav', 'mp3', 'm4a', 'txt']:
                self.logger.info(f"URL-Modul {module.id} verweist auf verarbeitbare Datei: {file_extension}")
                
                # Erstelle Resource-Objekt für die externe Datei
                resource = Resource(
                    filename=module.contents[0].filename or f"external_file.{file_extension}",
                    fileurl=external_url,
                    mimetype=file_extension,
                    filesize=module.contents[0].filesize if hasattr(module.contents[0], 'filesize') else None
                )
                
                # Download und extrahiere Text
                try:
                    caller = APICaller(url=external_url, params={})  # Keine Auth-Parameter für externe URLs
                    caller.get()
                    file_bytes = caller.response.content
                    
                    if file_bytes:
                        extracted_text = resource.extract_from_bytes(file_bytes, self.logger)
                        
                        if extracted_text:
                            # Kombiniere Intro + extrahierten Text
                            combined_text_parts = []
                            if intro_text:
                                combined_text_parts.append(intro_text)
                            combined_text_parts.append(extracted_text)
                            
                            resource.extracted_text = '\n'.join(combined_text_parts)
                            module.resource = resource
                            
                            self.logger.info(
                                f"URL-Modul {module.id} erfolgreich verarbeitet: "
                                f"{len(resource.extracted_text)} Zeichen extrahiert"
                            )
                        else:
                            self.logger.warning(
                                f"Keine Textextraktion möglich für {resource.filename} "
                                f"(Typ: {resource.mimetype})"
                            )
                            # Speichere trotzdem als URL-Modul mit Link
                            module.url_module = UrlModule(
                                url_id=module.instance,
                                module_id=module.id,
                                external_url=external_url,
                                intro=intro_text
                            )
                    else:
                        self.logger.error(f"Konnte Datei von {external_url} nicht herunterladen")
                        # Speichere als Link
                        module.url_module = UrlModule(
                            url_id=module.instance,
                            module_id=module.id,
                            external_url=external_url,
                            intro=intro_text
                        )
                        
                except Exception as e:
                    self.logger.warning(
                        f"Fehler beim Download/Verarbeiten von {external_url}: {e}. "
                        f"Speichere als einfachen Link."
                    )
                    # Fallback: Speichere als Link
                    module.url_module = UrlModule(
                        url_id=module.instance,
                        module_id=module.id,
                        external_url=external_url,
                        intro=intro_text
                    )
            else:
                # Externe Website oder nicht-unterstützter Dateityp → nur Link
                self.logger.info(
                    f"URL-Modul {module.id} verweist auf externe Website oder "
                    f"nicht-unterstützten Dateityp: {external_url}"
                )
                module.url_module = UrlModule(
                    url_id=module.instance,
                    module_id=module.id,
                    external_url=external_url,
                    intro=intro_text
                )
                self.logger.info(f"URL-Modul {module.id} als Link gespeichert")
            
            return None
            
        except Exception as e:
            error_msg = f"Fehler beim Laden von URL-Modul {module.id}: {str(e)}"
            self.logger.error(error_msg)
            return error_msg
    
    def _get_url_file_extension(self, url: str) -> str | None:
        """
        Extrahiert die Dateiendung aus einer URL.
        
        Args:
            url: URL-String
            
        Returns:
            str | None: Dateiendung (lowercase) oder None
        """
        from urllib.parse import urlparse, unquote
        
        try:
            parsed = urlparse(url)
            path = unquote(parsed.path)  # Decode URL-encoding
            
            # Extrahiere Dateiendung
            if '.' in path:
                extension = path.rsplit('.', 1)[-1].lower()
                # Bereinige Query-Parameter (falls vorhanden)
                extension = extension.split('?')[0].split('#')[0]
                return extension
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Fehler beim Parsen der URL {url}: {e}")
            return None

if __name__ == "__main__":
    moodle = Moodle()
    courses = moodle.extract()
