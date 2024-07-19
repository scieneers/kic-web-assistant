import os
import sys

from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from qdrant_client.http.models import (
    Distance,
    Filter,
    PointStruct,
    SearchRequest,
    VectorParams,
)

from src.env import EnvHelper

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)


class VectorDBQdrant:
    def __init__(self, version: str = "disk"):
        self.secrets = EnvHelper()

        self.version = version
        if version == "memory":
            self.client = QdrantClient(":memory:")
        elif version == "disk":
            self.client = QdrantClient("localhost", port=6333)
            try:
                _ = self.client.get_collections()
            except ResponseHandlingException as e:
                print("Qdrant container not running? Run:")
                print(
                    "docker run -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_storage:/qdrant/storage:z qdrant/qdrant:v1.6.1"
                )
                raise e
        elif version == "remote":
            self.client = QdrantClient(
                url=self.secrets.QDRANT_URL, port=443, https=True, timeout=30, api_key=self.secrets.QDRANT_TOKEN
            )
            _ = self.client.get_collections()
        else:
            raise ValueError("Version must be either 'memory' or 'disk' or 'remote'")

    def as_llama_vector_store(self, collection_name) -> QdrantVectorStore:
        return QdrantVectorStore(client=self.client, collection_name=collection_name)

    def create_collection(self, collection_name, vector_size) -> None:
        try:
            _ = self.client.get_collection(collection_name=collection_name)
        except UnexpectedResponse as e:
            _ = self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.DOT),
            )

    def upsert(self, collection_name, points: list[dict]) -> None:
        qdrant_points = [PointStruct(**point) for point in points]
        operation_info = self.client.upsert(
            collection_name=collection_name,
            wait=True,
            points=qdrant_points,
        )
        print(operation_info)

    def search(self, collection_name, query_vector, query_filter=None, with_payload=True, limit=3) -> list[dict]:
        search_result = self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            query_filter=query_filter,
            with_payload=with_payload,
            limit=limit,
        )
        return search_result

    def get_tree_of_courses(self, collection_name) -> dict:
        current_records = []

        next_page_offset = "first"
        offset = None

        while next_page_offset:
            if next_page_offset != "first":
                offset = next_page_offset

            records = self.client.scroll(
                collection_name=collection_name, with_payload=True, with_vectors=False, limit=10, offset=offset
            )

            next_page_offset = records[1]
            current_records.extend(records[0])

        set_of_courses = set()
        for record in current_records:
            set_of_courses.add((record.payload["course_id"], record.payload["fullname"]))

        sorted_courses_set = sorted(set_of_courses, key=lambda x: x[0])

        tree_data = []
        for value, title in sorted_courses_set:
            node = {"value": value, "title": title}
            tree_data.append(node)

        return tree_data


if __name__ == "__main__":
    pass
    # test_connection = VectorDBQdrant(version="remote")
    test_connection = VectorDBQdrant(version="disk")  # For local testing only
    print(test_connection.get_metadata("web_assistant"))
