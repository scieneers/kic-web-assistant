import re
from io import StringIO

from llama_index.core import Document
from pydantic import BaseModel, Field, computed_field, field_validator
from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import WebVTTFormatter

from src.env import EnvHelper
from src.loaders.APICaller import APICaller
from src.loaders.helper import convert_vtt_to_text
from src.loaders.models.coursetopic import CourseTopic
from src.loaders.models.module import ModuleTypes
from src.loaders.models.moodlecourse import MoodleCourse
from src.loaders.models.videotime import VideoPlatforms, VideoTime
from src.loaders.vimeo import Vimeo


class TextTrack(BaseModel):
    """Vimeo TextTrack"""

    id: int
    display_language: str
    language: str
    link: HttpUrl | None = None
    transcript: str | None = None


class VideoPlatforms(StrEnum):
    VIMEO = "vimeo"
    YOUTUBE = "youtube"
    SELF_HOSTED = "ki-campus"


class VideoTime(BaseModel):
    id: int
    video_url: HttpUrl = Field(..., alias="vimeo_url")  # This Field can contain a Vimeo _or_ a Youtube URL
    texttrack: TextTrack | None = None

    @computed_field
    @property
    def type(self) -> VideoPlatforms:
        match self.video_url.host:
            case "vimeo.com" | "player.vimeo.com":
                return VideoPlatforms.VIMEO
            case "www.youtube.com" | "youtu.be":
                return VideoPlatforms.YOUTUBE
            case "ki-campus-test.fernuni-hagen.de" | "moodle.ki-campus.org":
                return VideoPlatforms.SELF_HOSTED

    @computed_field
    @property
    def video_id(self) -> int:
        match self.type:
            case VideoPlatforms.VIMEO:
                vimeo_video_id_pattern = re.compile(r"\d+")
                return re.findall(vimeo_video_id_pattern, self.video_url.path)[0]
            case VideoPlatforms.YOUTUBE:
                youtube_video_id_pattern = (
                    r"(?:youtube\.com\/(?:[^\/]+\/.+\/|(?:v|e(?:mbed)?)\/|.*[?&]v=)|youtu\.be\/)([^\"&?\/\s]{11})"
                )
                return re.findall(youtube_video_id_pattern, str(self.video_url))[0]


class DownloadableContent(BaseModel):
    filename: str | None = None
    fileurl: str | None = None
    type: str | None = None

    @field_validator("type", mode="before")
    def set_type(cls, type, values):
        fileurl = values.data.get("fileurl")

        file_extension = fileurl.split(".")[-1] if fileurl is not None else ""
        if file_extension not in ["html", "hp5"]:
            # raise ValueError(f'File extension {file_extension} is not supported.')
            return file_extension


class Module(BaseModel):
    """Lowest level content block of a course. Can be a file, video, hp5, etc."""

    id: int
    visible: int
    name: str
    url: HttpUrl | None = None
    modname: str  # content type
    contents: list[DownloadableContent] | None = None
    videotime: VideoTime | None = None


class ModuleTypes(StrEnum):
    VIDEOTIME = "videotime"
    PAGE = "page"
    H5P = "h5pactivity"


class CourseTopic(BaseModel):
    id: int
    name: str
    summary: str | None

    @field_validator("summary")
    @classmethod
    def remove_html(cls, summary: str) -> str:
        if not summary:
            return None
        return process_html_summaries(summary)

    def __str__(self) -> str:
        # add transcripts
        modules = "\n ".join([str(module) for module in self.modules])
        text = f"Topic Summary: {self.summary}\n" "Topic Modules :\n" f"{modules}"
        return text


class MoodleCourse(BaseModel):
    """Highest level representation of a Moodle course. Has 1 to many topics (called modules in ki-campus frontend)."""

    id: int
    shortname: str
    fullname: str
    displayname: str
    summary: str | None = None
    lang: str
    url: str
    topics: list[CourseTopic] = None

    @field_validator("summary")
    @classmethod
    def remove_html(cls, summary: str) -> str:
        if not summary:
            raise ValueError("Summary should not be empty")
        return process_html_summaries(summary)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.url = f"{self.url}{self.id}"

    def __str__(self) -> str:
        topics = "\n ".join([str(topic) for topic in self.topics])
        text = f"Course Summary: {self.summary}\n" "Course Topics:\n" f"{topics}"
        return text

    def to_document(self) -> Document:
        text = str(self)
        metadata = {
            "id": self.id,
            "shortname": self.shortname,
            "fullname": self.fullname,
            "type": "course",
            "url": self.url,
        }
        if self.lang:
            metadata["lang"] = self.lang
        return Document(text=text, metadata=metadata)


class Moodle:
    """class for Moodle API clients
    base_url: base url of moodle instance
    token: token for moodle api"""

    def __init__(self, environment: str = "STAGING", base_url: str = "", token: str = "") -> None:
        secrets = EnvHelper(production=True if environment == "PRODUCTION" else False)
        self.base_url = secrets.DATA_SOURCE_MOODLE_URL if base_url == "" else base_url
        self.api_endpoint = f"{self.base_url}webservice/rest/server.php"
        self.token = secrets.DATA_SOURCE_MOODLE_TOKEN if token == "" else token
        self.function_params = {
            "wstoken": self.token,
            "moodlewsrestformat": "json",
        }
        self.download_params = {
            "token": self.token,
            "forcedownload": 1,
        }

    def get_courses(self) -> list[MoodleCourse]:
        """get all courses that are set to visible on moodle"""
        caller = APICaller(
            url=self.api_endpoint,
            params=self.function_params,
            wsfunction="core_course_get_courses",
        )
        courses = caller.getJSON()
        course_url = self.base_url + "course/view.php?id="
        courses = [MoodleCourse(url=course_url, **course) for course in courses if course["visible"]]
        return courses

    def get_course_contents(self, course_id: int) -> list[CourseTopic]:
        """get all contents/topics/modules of a course"""
        course_contents_caller = APICaller(
            url=self.api_endpoint,
            params=self.function_params,
            wsfunction="core_course_get_contents",
            courseid=course_id,
        )
        course_contents = course_contents_caller.getJSON(courseid=course_id)
        course_contents = [CourseTopic(**topic) for topic in course_contents]
        return course_contents

    def get_videotime_content(self, cmid: int):
        caller = APICaller(
            url=self.api_endpoint,
            params=self.function_params,
            wsfunction="mod_videotime_get_videotime",
            cmid=cmid,
        )
        videotime_content = caller.getJSON()
        return videotime_content

    def extract(self) -> list[MoodleCourse]:
        """extracts all courses and their contents from moodle"""
        courses = self.get_courses()
        for course in courses:
            course.topics = self.get_course_contents(course.id)
            for topic in course.topics:
                self.get_module_contents(topic)
        course_documents = [course.to_document() for course in courses]
        return course_documents

    def get_module_contents(self, topic):
        for module in topic.modules:
            if module.visible == 0:
                continue
            match module.type:
                case ModuleTypes.VIDEOTIME:
                    self.extract_videotime(module)
                case ModuleTypes.PAGE:
                    self.extract_page(module)
                case ModuleTypes.H5P:
                    # get contents of h5p page
                    # extract video link(s)
                    pass

    def extract_page(self, module):
        for content in module.contents:
            page_content_caller = APICaller(url=content.fileurl, params=self.download_params)
            soup = BeautifulSoup(page_content_caller.getText(), "html.parser")
            links = soup.find_all("a")
            for p_link in links:
                src = p_link.get("href")
                if src:
                    videotime = VideoTime(id=0, vimeo_url=src)
                    if src.find("vimeo") != -1:
                        vimeo = Vimeo()
                        texttrack = vimeo.get_transcript(videotime.video_id)
                    elif src.find("youtu") != -1:
                        youtube = Youtube()
                        texttrack = youtube.get_transcript(videotime.video_id)
                    else:
                        texttrack = None
                    module.transcripts.append(texttrack)

    def extract_videotime(self, module):  # TODO: rename method
        videotime = VideoTime(**self.get_videotime_content(module.id))

        match videotime.type:
            case VideoPlatforms.VIMEO:
                vimeo = Vimeo()
                texttrack = vimeo.get_transcript(videotime.video_id)
                module.transcripts.append(texttrack)
            case VideoPlatforms.YOUTUBE:
                youtube = Youtube()
                texttrack = youtube.get_transcript(videotime.video_id)
                module.transcripts.append(texttrack)
            case VideoPlatforms.SELF_HOSTED:
                # no subtitles found in self-hosted videos, if this ever changes add code here
                pass


if __name__ == "__main__":
    moodle = Moodle()
    courses = moodle.extract()
    print(courses)
