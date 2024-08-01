from typing import List

import requests
from bs4 import BeautifulSoup
from llama_index.core import Document

from src.env import env


class Drupal:
    def __init__(
        self, base_url: str = "", username: str = "", client_id: str = "", client_secret: str = "", grant_type: str = ""
    ) -> None:
        # This fixes very slow requests, IPV6 is not properly supported by the ki-campus.org server
        # https://stackoverflow.com/questions/62599036/python-requests-is-slow-and-takes-very-long-to-complete-http-or-https-request
        requests.packages.urllib3.util.connection.HAS_IPV6 = False

        self.oauth_token = self.get_oauth_token("https://ki-campus.org")
        self.header = {
            "Authorization": f"Bearer {self.oauth_token}",
            "Accept": "application/vnd.api+json",
            "Accept-Language": "de",
        }

    def get_oauth_token(self, base_url: str):
        response = requests.post(
            f"{base_url}/oauth/token",
            data={
                "client_id": env.DRUPAL_CLIENT_ID,
                "client_secret": env.DRUPAL_CLIENT_SECRET,
                "username": env.DRUPAL_USERNAME,
                "password": env.DRUPAL_PASSWORD,
                "grant_type": env.DRUPAL_GRANT_TYPE,
            },
        )

        print("status_code: ", response.status_code)

        if response.status_code != 200:
            return

        return response.json()["access_token"]

    def get_data(self, url: str):
        data = []

        while url:
            response = requests.get(url, headers=self.header)
            result = response.json()
            data.extend(result["data"])
            next_link = result["links"].get("next")
            url = next_link["href"] if next_link else None
        return data

    def get_course_paragraphs(self, course_id: str):
        auth_header = {"Authorization": f"Bearer {self.oauth_token}"}
        response = requests.get(
            f"https://ki-campus.org/jsonapi/node/course/{course_id}/field_paragraphs", headers=auth_header
        )
        paragraphs = response.json()

        _result = ""
        for d in paragraphs["data"]:
            if d["type"] == "paragraph--simple_text":
                if d["attributes"]["field_paragraph_title"] is not None:
                    _result += d["attributes"]["field_paragraph_title"]
                    _result += "\n"

                if d["attributes"]["field_paragraph_body"] is not None:
                    _result += BeautifulSoup(d["attributes"]["field_paragraph_body"]["value"]).getText()
                    _result += "\n\n"

        return _result

    def get_course_representation(self, course):
        description = ""
        if course["attributes"]["field_description"] is not None:
            description = BeautifulSoup(course["attributes"]["field_description"]["value"], "html.parser").getText()
        paragraphs = self.get_course_paragraphs(course["id"])

        final_representations = f"""
        Course Title: {course["attributes"]["title"]}
        Course Description: {description}

        {paragraphs}
        """
        return final_representations

    def extract(self):
        courses_docs = self.get_courses()
        page_docs = self.get_pages()

        all_docs = page_docs + courses_docs

        return all_docs

    def get_courses(self) -> List[Document]:
        documents: list[Document] = []
        node = self.get_data("https://ki-campus.org/jsonapi/node/course")
        for i, course in enumerate(node):
            documents.append(
                Document(
                    metadata={"title": course["attributes"]["title"], "course_id": course["id"]},
                    text=self.get_course_representation(course),
                )
            )

            print("course number:", i)
        return documents

    def get_pages(self) -> List[Document]:
        documents: list[Document] = []
        node = self.get_data("https://ki-campus.org/jsonapi/node/page")
        for i, page in enumerate(node):
            documents.append(
                Document(
                    metadata={"title": page["attributes"]["title"]},
                    text=self.get_page_representation(page),
                )
            )

        return documents

    def get_page_representation(self, page):
        final_representations = f"""
        Page Title: {page["attributes"]["title"]}
        """

        if page["attributes"]["body"] is not None:
            content = BeautifulSoup(page["attributes"]["body"]["value"], "html.parser").getText()

            if content is not None:
                content_text = f"\nPage Content: {content}"
                final_representations += content_text

        return final_representations
