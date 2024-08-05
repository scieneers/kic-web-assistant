import logging
from enum import Enum
from typing import List

import requests
from bs4 import BeautifulSoup
from llama_index.core import Document

from src.env import env


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


class Drupal:
    def __init__(
        self, base_url: str = "", username: str = "", client_id: str = "", client_secret: str = "", grant_type: str = ""
    ) -> None:
        self.logger = logging.getLogger("loader")
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

        if response.status_code != 200:
            return

        return response.json()["access_token"]

    def extract(self):
        all_docs = []

        for type in PageTypes:
            all_docs += self.get_page_type(type)
        return all_docs

    def get_page_type(self, page_type: PageTypes) -> List[Document]:
        documents: list[Document] = []
        node = self.get_data(f"https://ki-campus.org/jsonapi/node/{page_type.value[0]}")
        for i, page in enumerate(node):
            self.logger.debug(f"Processing {page_type.value[0]} number: {i+1}/{len(node)}")

            if page["attributes"]["status"]:
                metadata = {"title": page["attributes"]["title"], "type": f"Drupal_{page_type.value[0]}"}
                documents.append(
                    Document(
                        metadata=metadata,
                        text=self.get_page_representation(page, page_type),
                    )
                )

        return documents

    def get_data(self, url: str):
        data = []

        while url:
            response = requests.get(url, headers=self.header)
            result = response.json()
            data.extend(result["data"])
            next_link = result["links"].get("next")
            url = next_link["href"] if next_link else None
        return data

    def get_page_paragraphs(self, page_id: str, page_type: PageTypes):
        response = requests.get(
            f"https://ki-campus.org/jsonapi/node/{page_type.value[0]}/{page_id}/field_paragraphs", headers=self.header
        )
        paragraphs = response.json()

        _result = ""
        for d in paragraphs["data"]:
            if d["type"] in ["paragraph--simple_text", "paragraph--textblock"]:
                if d["attributes"]["field_paragraph_title"] is not None:
                    _result += d["attributes"]["field_paragraph_title"]
                    _result += "\n"

                if d["attributes"]["field_paragraph_body"] is not None:
                    _result += BeautifulSoup(d["attributes"]["field_paragraph_body"]["value"]).getText()
                    _result += "\n\n"

        return _result

    def get_page_representation(self, page, page_type: PageTypes):
        if page_type in [PageTypes.PAGE, PageTypes.BLOGPOST]:
            final_representations = f"""
            {page_type.value[1]} Title: {page["attributes"]["title"]}
            """

            if page["attributes"]["body"] is not None:
                content = BeautifulSoup(page["attributes"]["body"]["value"], "html.parser").getText()

                if content is not None:
                    content_text = f"\{page_type.value[1]} Content: {content}"
                    final_representations += content_text
        else:
            description = ""
            if page["attributes"]["field_description"] is not None:
                description = BeautifulSoup(page["attributes"]["field_description"]["value"], "html.parser").getText()
            paragraphs = self.get_page_paragraphs(page["id"], page_type)

            final_representations = f"""
            {page_type.value[1]} Title: {page["attributes"]["title"]}
            {page_type.value[1]} Description: {description}

            {paragraphs}
            """
        return final_representations


if __name__ == "__main__":
    docs = Drupal().extract()
