import json
import tempfile
import zipfile

from bs4 import BeautifulSoup

from src.env import env
from src.loaders.APICaller import APICaller
from src.loaders.models.coursetopic import CourseTopic
from src.loaders.models.hp5activities import H5PActivities
from src.loaders.models.module import ModuleTypes
from src.loaders.models.moodlecourse import MoodleCourse
from src.loaders.models.videotime import Video, VideoPlatforms
from src.loaders.vimeo import Vimeo
from src.loaders.youtube import Youtube


class Moodle:
    """class for Moodle API clients
    base_url: base url of moodle instance
    token: token for moodle api"""

    def __init__(self, base_url: str = "", token: str = "") -> None:
        self.base_url = env.DATA_SOURCE_MOODLE_URL if base_url == "" else base_url
        self.api_endpoint = f"{self.base_url}webservice/rest/server.php"
        self.token = env.DATA_SOURCE_MOODLE_TOKEN if token == "" else token
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
        courses = [MoodleCourse(url=course_url, **course) for course in courses if course["visible"] == 1]
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
        ids_json = h5p_module_ids_caller.getJSON()
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

    def extract(self) -> list[MoodleCourse]:
        """extracts all courses and their contents from moodle"""
        courses = self.get_courses()
        for course in courses:
            course.topics = self.get_course_contents(course.id)
            h5p_activity_ids = self.get_h5p_module_ids(course.id)
            for topic in course.topics:
                self.get_module_contents(topic, h5p_activity_ids)
        course_documents = [doc for course in courses for doc in course.to_document()]
        return course_documents

    def get_module_contents(self, topic, h5p_activities):
        for module in topic.modules:
            if module.visible == 0:
                continue
            match module.type:
                case ModuleTypes.VIDEOTIME:
                    self.extract_videotime(module)
                case ModuleTypes.PAGE:
                    self.extract_page(module)
                case ModuleTypes.H5P:
                    for activity in h5p_activities:
                        if activity.coursemodule == module.id:
                            self.extract_h5p(module, activity)

    def extract_page(self, module):
        for content in module.contents:
            page_content_caller = APICaller(url=content.fileurl, params=self.download_params)
            soup = BeautifulSoup(page_content_caller.getText(), "html.parser")
            links = soup.find_all("a")
            for p_link in links:
                src = p_link.get("href")
                if src:
                    if src.find("vimeo") != -1:
                        videotime = Video(id=0, vimeo_url=src)
                        vimeo = Vimeo()
                        texttrack = vimeo.get_transcript(videotime.video_id)
                    elif src.find("youtu") != -1:
                        videotime = Video(id=0, vimeo_url=src)
                        youtube = Youtube()
                        texttrack = youtube.get_transcript(videotime.video_id)
                    else:
                        texttrack = None
                    module.transcripts.append(texttrack)

    def extract_videotime(self, module):  # TODO: rename method
        videotime = Video(**self.get_videotime_content(module.id))

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
                # found no subtitles in self-hosted videos, if this ever changes add code here
                pass

    # A H5P Module is a zipped bundle of js, css and a content.json, describing the content.
    # If a video is wrapped in a H5P Module we are only interested in the content.json.
    # In the content.json we find the link to the video. Based on that link we can construct
    # the link to the transcript of that video. Then we download that transcript and add it to the
    # module.transcripts list.
    def extract_h5p(self, module, activity):
        h5pfile_call = APICaller(url=activity.fileurl, params=self.download_params)
        with tempfile.TemporaryDirectory() as tmp_dir:
            local_filename = h5pfile_call.getFile(activity.filename, tmp_dir)
            with zipfile.ZipFile(local_filename, "r") as zip_ref:
                zip_ref.extract("content/content.json", tmp_dir)
            content_json = f"{tmp_dir}/content/content.json"
            with open(content_json, "r") as json_file:
                content = json.load(json_file)
            if "interactiveVideo" in content.keys():
                videourl = content["interactiveVideo"]["video"]["files"][0]["path"]
                video = Video(id=0, vimeo_url=videourl)
                vimeo = Vimeo()
                texttrack = vimeo.get_transcript(video.video_id)
                module.transcripts.append(texttrack)


if __name__ == "__main__":
    moodle = Moodle()
    courses = moodle.extract()
