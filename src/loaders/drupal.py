import logging
import os
import unicodedata
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List
import requests
from bs4 import BeautifulSoup
from llama_index.core import Document

from src.env import env
from src.loaders.run_logger import RunContext, format_kv


class PageTypes(Enum):
    """
    The different types of pages that can be found on ki-campus.org
    First element is the internal name of the page,
    second element is the human-readable and translated name of the page
    """

    COURSE = ("course", "Kurs")
    ABOUT_US = ("about_us", "Über uns")
    PAGE = ("page", "Seite")
    BLOGPOST = ("blogpost", "Blogpost")
    SPEZIAL = ("dvv_page", "Spezial")  # Spezial / Stadt Land DatenFluss


DRUPAL_API_BASE_URL = "https://ki-campus.org/jsonapi/node/"


class Drupal:
    def __init__(
        self,
        base_url: str = "",
        username: str = "",
        client_id: str = "",
        client_secret: str = "",
        grant_type: str = "",
        run_ctx: RunContext | None = None,
    ) -> None:
        self.logger = logging.getLogger("loader")
        self.run_ctx = run_ctx
        # This fixes very slow requests, IPV6 is not properly supported by the ki-campus.org server
        # https://stackoverflow.com/questions/62599036/python-requests-is-slow-and-takes-very-long-to-complete-http-or-https-request
        requests.packages.urllib3.util.connection.HAS_IPV6 = False

        self.important_courses = self._load_important_courses()
        self.header = {
            "Accept": "application/vnd.api+json",
            "Accept-Language": "de",
        }
        credentials_set = (
            hasattr(env, "DRUPAL_CLIENT_ID")
            and hasattr(env, "DRUPAL_CLIENT_SECRET")
            and hasattr(env, "DRUPAL_USERNAME")
            and hasattr(env, "DRUPAL_PASSWORD")
        )
        authenticated = False
        if credentials_set:
            oauth_token = self.get_oauth_token("https://ki-campus.org")
            if oauth_token:
                self.header["Authorization"] = f"Bearer {oauth_token}"
                authenticated = True

        if not authenticated:
            if env.DRUPAL_AUTH_REQUIRED:
                raise RuntimeError(
                    "Drupal: DRUPAL_AUTH_REQUIRED is set but no valid credentials/token "
                    "are available — refusing to run unauthenticated, since that would "
                    "only see public content and could stale-delete protected courses."
                )
            self.logger.info("Drupal: no credentials configured, running without authentication")

    def _load_important_courses(self) -> set[int]:
        """Load important course IDs from IMPORTANT_COURSES.txt"""
        important_courses_file = Path(__file__).parents[2] / "documentation" / "IMPORTANT_COURSES.txt"
        try:
            content = important_courses_file.read_text().strip()
            # Parse the list format: [99, 106, 313, ...]
            course_ids = eval(content)
            return set(course_ids)
        except Exception as e:
            self.logger.warning(f"Could not load IMPORTANT_COURSES.txt: {e}")
            return set()

    def get_oauth_token(self, base_url: str):
        response = requests.post(
            f"{base_url}/oauth2/token",
            data={
                "client_id": env.DRUPAL_CLIENT_ID,
                "client_secret": env.DRUPAL_CLIENT_SECRET,
                "username": env.DRUPAL_USERNAME,
                "password": env.DRUPAL_PASSWORD,
                "grant_type": env.DRUPAL_GRANT_TYPE,
            },
        )

        if response.status_code != 200:
            self.logger.warning(
                "Drupal OAuth failed %s RESPONSE=%s",
                format_kv(
                    STAGE="DRUPAL",
                    EVENT="OAUTH_FAILED",
                    STATUS_CODE=response.status_code,
                ),
                response.text[:500],
            )
            return

        token = response.json().get("access_token")
        if token:
            self.logger.info(
                "Drupal OAuth %s",
                format_kv(STAGE="DRUPAL", EVENT="OAUTH_OK"),
            )
        return token

    def extract(self):
        all_docs: list[Document] = []

        self.logger.info("Drupal: starting extraction for %s page types", len(list(PageTypes)))
        for page_type in PageTypes:
            if self.run_ctx:
                self.run_ctx.set_last(url=f"{DRUPAL_API_BASE_URL}{page_type.value[0]}")
                self.run_ctx.checkpoint()
            docs = self.get_page_type(page_type)
            self.logger.info("Drupal: page_type=%s -> %s documents", page_type.value[0], len(docs))
            all_docs += docs
        self.logger.info("Drupal: total documents=%s", len(all_docs))
        return all_docs

    def get_page_type(self, page_type: PageTypes) -> List[Document]:
        documents: list[Document] = []
        url = f"{DRUPAL_API_BASE_URL}{page_type.value[0]}"
        self.logger.debug("Drupal: fetching %s", url)
        node = self.get_data(url)
        self.logger.info("Drupal: fetched %s raw nodes for type=%s", len(node), page_type.value[0])
        for i, page in enumerate(node):
            self.logger.debug(f"Processing {page_type.value[0]} number: {i+1}/{len(node)}")

            if page["attributes"]["status"]:
                metadata = {
                    "title": page["attributes"]["title"],
                    "source": "Drupal",
                    "type": f"{page_type.value[0]}",
                    "date_created": datetime.fromisoformat(page["attributes"]["created"]).strftime("%Y-%m-%d"),
                    "url": f"https://ki-campus.org/node/{page['attributes']['drupal_internal__nid']}",
                }

                if page.get("attributes", {}).get("field_moodle_course_id") is not None:
                    metadata["course_id"] = page["attributes"]["field_moodle_course_id"]
                    # Mark popular/recommended courses
                    if metadata["course_id"] in self.important_courses:
                        metadata["is_important"] = True
               
                # Fetch author names for BLOGPOST pages
                if page_type == PageTypes.BLOGPOST:
                    author_data = page.get("relationships", {}).get("field_author", {}).get("data", [])
                    if author_data:
                        author_names = self.get_authors(author_data)
                        if author_names:
                            metadata["authors"] = author_names

                documents.append(
                    Document(
                        metadata=metadata,
                        text=self.get_page_representation(page, page_type, metadata),
                    )
                )

        if len(node) > 0 and len(documents) == 0:
            self.logger.warning(
                "Drupal: fetched %s nodes for type=%s but created 0 documents (maybe all nodes are unpublished?)",
                len(node),
                page_type.value[0],
            )

        return documents

    def get_data(self, url: str):
        data = []

        page = 0

        while url:
            page += 1
            if self.run_ctx:
                self.run_ctx.set_last(url=url)
                self.run_ctx.checkpoint()
            response = requests.get(url, headers=self.header)
            
            if response.status_code != 200:
                self.logger.warning(
                    "API request failed STATUS=%s URL=%s RESPONSE=%s",
                    response.status_code,
                    url,
                    response.text[:500],
                )
                return data
            
            result = response.json()
            data.extend(result["data"])
            if page == 1 or page % int(os.getenv("RUN_DRUPAL_PROGRESS_EVERY_PAGES", "5")) == 0:
                self.logger.info(
                    "Drupal paging %s",
                    format_kv(
                        STAGE="DRUPAL",
                        EVENT="PAGE",
                        PAGE=page,
                        NODES_CUMULATIVE=len(data),
                    ),
                )
            next_link = result["links"].get("next")
            url = next_link["href"] if next_link else None
        return data

    def get_page_paragraphs(self, page_id: str, page_type: PageTypes | str):
        if type(page_type) is PageTypes:
            response = requests.get(
                f"{DRUPAL_API_BASE_URL}{page_type.value[0]}/{page_id}/field_paragraphs", headers=self.header
            )
        elif type(page_type) is str:
            response = requests.get(
                f"{DRUPAL_API_BASE_URL}{page_type}/{page_id}/field_content_paragraphs", headers=self.header
            )
        else:
            raise Exception('Bad type: "page_type"')
        paragraphs = response.json()

        _result = ""
        for d in paragraphs["data"]:
            if d["type"] in ["paragraph--simple_text", "paragraph--textblock", "paragraph--text_and_image"]:
                if d["attributes"].get("field_paragraph_title") is not None:
                    _result += d["attributes"]["field_paragraph_title"]
                    _result += "\n"

                if d["attributes"].get("field_paragraph_body") is not None:
                    _result += BeautifulSoup(d["attributes"]["field_paragraph_body"]["value"], "html.parser").getText()
                    _result += "\n\n"

        return _result

    def fetch_data(self, url):
        response = requests.get(url, headers=self.header)
        return response.json()

    def process_lecture_books(self, page) -> str:
        lecture_books = page["relationships"]["field_lecture_books"]["data"]
        books_text = ""

        for lecture_book in lecture_books:
            lecture_book_url = f"{DRUPAL_API_BASE_URL}lecture_book/{lecture_book['id']}"
            chapter_data = self.fetch_data(lecture_book_url)
            books_text += self.process_chapters(chapter_data)
            pass

        return books_text

    def process_chapters(self, chapter_data) -> str:
        chapters = chapter_data["data"]["relationships"]["field_lecture_chapters"]["data"]
        chapters_text = ""

        for single_chapter in chapters:
            # Yes, its really called lecture_chaper
            chapter_url = f"{DRUPAL_API_BASE_URL}lecture_chaper/{single_chapter['id']}"
            lecture_data = self.fetch_data(chapter_url)

            chapters_text += self.process_lectures(lecture_data)

        return chapters_text

    def process_lectures(self, lecture_data) -> str:
        lectures = lecture_data["data"]["relationships"]["field_lectures"]["data"]
        lectures_text = ""

        for lecture in lectures:
            lectures_text += self.get_page_paragraphs(lecture["id"], "lecture")

        return lectures_text

    def get_page_representation(self, page, page_type: PageTypes, metadata):
        final_representations = ""

        match page_type:
            case PageTypes.COURSE:
                final_representations += self.get_course_representation(page, page_type)
            case PageTypes.BLOGPOST:
                final_representations += self.get_blogpost_representation(page, page_type, metadata)
            case PageTypes.PAGE:
                final_representations += self.get_basic_representation(page, page_type)

            case PageTypes.SPEZIAL:
                final_representations += self.get_basic_representation(page, page_type)
                final_representations += self.process_lecture_books(page)

            case _:
                description = ""
                if page.get("relationships", {}).get("field_description") is not None:
                    description = BeautifulSoup(
                        page["attributes"]["field_description"]["value"], "html.parser"
                    ).getText()
                paragraphs = self.get_page_paragraphs(page["id"], page_type)

                type = ""
                if page.get("attributes", {}).get("field_format") is not None:
                    type = f'{page_type.value[1]} Type: {self.get_course_type(page["attributes"]["field_format"])}'

                length = ""
                if page.get("attributes", {}).get("field_umfang") is not None:
                    length = f'{page_type.value[1]} Length: {page["attributes"]["field_umfang"]}'

                difficulty = ""
                if page.get("attributes", {}).get("field_level") is not None:
                    difficulty = f'{page_type.value[1]} Difficulty: {page["attributes"]["field_level"]}'

                language = ""
                if page.get("attributes", {}).get("field_course_language") is not None:
                    language = f'{page_type.value[1]} language code: {page["attributes"]["field_course_language"]}'

                topics = ""
                if page.get("relationships", {}).get("field_occupational_field", {}).get("data") is not None:
                    topics = f"{page_type.value[1]} Topic(s): {self.get_course_topic(page['relationships']['field_occupational_field']['data'])}"

                final_representations = f"""
                {page_type.value[1]} Title: {page["attributes"]["title"]}
                {page_type.value[1]} Description: {description}
                {type}
                {length}
                {difficulty}
                {language}
                {topics}

                {paragraphs}
                """

        # Normalize parsed text (remove \xa0 from str)
        final_representations = unicodedata.normalize("NFKD", final_representations)
        return final_representations

    def get_course_type(self, short_type: str):
        match short_type:
            case "mooc":
                return "Online-Kurse & MOOCs"
            case "blended":
                return "Blended Learning"
            case "micro":
                return "Micro Content"
            case "podcast":
                return "Podcasts"
            case "video":
                return "Lernvideos"
            case "paths":
                return "Lernpfade"
            case _:
                raise Exception(f"Bad type: {short_type}")

    def get_course_topic(self, topic_list: list):
        topics_str = ""

        for idx, topic in enumerate(topic_list):
            topic_data = self.fetch_data(
                f"https://ki-campus.org/jsonapi/taxonomy_term/occupational_field/{topic['id']}"
            )
            if (topic_data.get("data")) is not None:
                if idx != 0:
                    topics_str += ", "
                topics_str += f"{topic_data['data']['attributes']['name']}"

        return topics_str

    def get_institutions(self, institution_list: list) -> List[str]:
        """Fetch institution names from the API by their IDs.
        
        Args:
            institution_list: List of institution data objects containing 'id' and 'type' keys
            
        Returns:
            List of institution names
        """
        institution_names = []

        for institution in institution_list:
            if institution.get("id") is None:
                continue
            
            institution_data = self.fetch_data(
                f"https://ki-campus.org/jsonapi/taxonomy_term/institution/{institution['id']}"
            )
            if institution_data.get("data") is not None:
                name = institution_data["data"].get("attributes", {}).get("name")
                if name:
                    institution_names.append(name)

        return institution_names

    def get_lecturers(self, lecturer_list: list) -> List[str]:
        """Fetch lecturer names from the API by their IDs.
        
        Args:
            lecturer_list: List of lecturer data objects containing 'id' and 'type' keys
            
        Returns:
            List of lecturer names
        """
        lecturer_names = []

        for lecturer in lecturer_list:
            if lecturer.get("id") is None:
                continue
            
            lecturer_data = self.fetch_data(
                f"https://ki-campus.org/jsonapi/node/lecturer/{lecturer['id']}"
            )
            if lecturer_data.get("data") is not None:
                name = lecturer_data["data"].get("attributes", {}).get("title")
                if name:
                    lecturer_names.append(name)

        return lecturer_names

    def get_authors(self, author_list: list) -> List[str]:
        """Fetch author names from the API by their IDs.
        
        Args:
            author_list: List of author data objects containing 'id' and 'type' keys
            
        Returns:
            List of author names
        """
        author_names = []

        for author in author_list:
            if author.get("id") is None:
                continue
            
            author_data = self.fetch_data(
                f"https://ki-campus.org/jsonapi/node/person/{author['id']}"
            )
            if author_data.get("data") is not None:
                name = author_data["data"].get("attributes", {}).get("title")
                if name:
                    author_names.append(name)

        return author_names

    def get_related_name(self, url: str) -> str | None:
        """Fetch the name attribute from a related JSON API URL.
        
        Args:
            url: The related href URL to fetch data from
            
        Returns:
            The name attribute from the response, or None if not found
        """
        try:
            data = self.fetch_data(url)
            if data.get("data") is not None:
                return data["data"].get("attributes", {}).get("name")
        except Exception:
            pass
        return None

    def get_basic_representation(self, page, page_type: PageTypes):
        final_representations = f"""
                    {page_type.value[1]} Title: {page["attributes"]["title"]}
                """
        if page["attributes"]["body"] is not None:
            content = BeautifulSoup(page["attributes"]["body"]["value"], "html.parser").getText()

            if content is not None:
                content_text = f"\n{page_type.value[1]} Content: {content}"
                final_representations += content_text

        return final_representations
    
    def get_course_data(self, page):
        course_data = {}
        institution_data = page.get("relationships", {}).get("field_institution", {}).get("data", [])
        if institution_data:
            institution_names = self.get_institutions(institution_data)
            if institution_names:
                course_data["institutions"] = institution_names

        lecturer_data = page.get("relationships", {}).get("field_lecturer", {}).get("data", [])
        if lecturer_data:
            lecturer_names = self.get_lecturers(lecturer_data)
            if lecturer_names:
                course_data["lecturers"] = lecturer_names

        # description
        if page.get("attributes", {}).get("field_description") is not None:
            course_data["description"] = BeautifulSoup(
                page["attributes"]["field_description"]["value"], "html.parser"
            ).getText()

        # course_type (field_format)
        if page.get("attributes", {}).get("field_format") is not None:
            course_data["course_type"] = self.get_course_type(page["attributes"]["field_format"])

        # field_umfang (course duration/scope)
        if page.get("attributes", {}).get("field_umfang") is not None:
            course_data["field_umfang"] = page["attributes"]["field_umfang"]
        # difficulty (field_level)
        if page.get("attributes", {}).get("field_level") is not None:
            course_data["difficulty"] = page["attributes"]["field_level"]

        # topics (field_occupational_field)
        if page.get("relationships", {}).get("field_occupational_field", {}).get("data") is not None:
            course_data["topics"] = self.get_course_topic(page["relationships"]["field_occupational_field"]["data"])
        # course_level
        course_level_url = page.get("relationships", {}).get("field_course_level", {}).get("links", {}).get("related", {}).get("href")
        if course_level_url:
            course_level_name = self.get_related_name(course_level_url)
            if course_level_name:
                course_data["course_level"] = course_level_name

        # degree (achievement record)
        degree_url = page.get("relationships", {}).get("field_achievement_record", {}).get("links", {}).get("related", {}).get("href")
        if degree_url:
            degree_name = self.get_related_name(degree_url)
            if degree_name:
                course_data["degree"] = degree_name

        # rating_avg (divide by 2 and format as "X Sterne")
        rating_avg_str = page.get("attributes", {}).get("field_rating_avg")
        if rating_avg_str:
            try:
                rating_value = float(rating_avg_str) / 2
                course_data["rating_avg"] = f"{rating_value} Sterne"
            except (ValueError, TypeError):
                pass

        # rating_count
        rating_count = page.get("attributes", {}).get("field_rating_count")
        if rating_count is not None:
            course_data["rating_count"] = rating_count

        # language (content language)
        language_url = page.get("relationships", {}).get("field_content_language", {}).get("links", {}).get("related", {}).get("href")
        if language_url:
            language_name = self.get_related_name(language_url)
            if language_name:
                course_data["language"] = language_name

        # license
        license_url = page.get("relationships", {}).get("field_license", {}).get("links", {}).get("related", {}).get("href")
        if license_url:
            license_name = self.get_related_name(license_url)
            if license_name:
                course_data["license"] = license_name


        return course_data
    
    def get_course_representation(self, page, page_type: PageTypes):
        paragraphs = self.get_page_paragraphs(page["id"], page_type)

        course_data = self.get_course_data(page)

        rating_line = ""
        if course_data.get("rating_avg"):
            rating_line = f"Average {page_type.value[1]} Rating on a scale from 0 to 5, 0 being the worst, 5 being the best: {course_data.get('rating_avg')} at {course_data.get('rating_count', 0)} ratings"
        else:
            rating_line = "No ratings available for this course."

        final_representations = f"""
        {page_type.value[1]} Title: {page["attributes"]["title"]}
        {page_type.value[1]} Description: {course_data.get("description", "")}
        {page_type.value[1]} Type: {course_data.get("course_type", "")}
        {page_type.value[1]} Length: {course_data.get("field_umfang", "")}
        {page_type.value[1]} Difficulty: {course_data.get("difficulty", "")}
        {page_type.value[1]} Target Group: {course_data.get("course_level", "")}
        {page_type.value[1]} Language: {course_data.get("language", "")}
        {page_type.value[1]} Topic(s): {course_data.get("topics", "")}
        {page_type.value[1]} Institutions: {', '.join(course_data.get("institutions", []))}
        {page_type.value[1]} Lecturers: {', '.join(course_data.get("lecturers", []))}
        {page_type.value[1]} Degree: {course_data.get("degree", "")}
        {rating_line}
        {page_type.value[1]} License: {course_data.get("license", "")}

        {paragraphs}
        """

        return final_representations
    
    def get_blogpost_representation(self, page, page_type: PageTypes, metadata):
        final_representations = f"""
                    {page_type.value[1]} Title: {metadata.get("title", "")}
                    {page_type.value[1]} Authors: {', '.join(metadata.get("authors", []))}
                    {page_type.value[1]} Published on: {metadata.get("date_created", "")}
                """
        if page["attributes"]["body"] is not None:
            content = BeautifulSoup(page["attributes"]["body"]["value"], "html.parser").getText()

            if content is not None:
                content_text = f"\n{page_type.value[1]} Content: {content}"
                final_representations += content_text

        return final_representations




if __name__ == "__main__":
    docs = Drupal().extract()
