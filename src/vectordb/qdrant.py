import os
import sys

from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from qdrant_client.http.models import Distance, PointStruct, VectorParams

from src.env import env

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)


class VectorDBQdrant:
    def __init__(self, version: str = "remote"):
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
            self.client = QdrantClient(url=env.QDRANT_URL, port=443, https=True, timeout=30, api_key=env.QDRANT_API_KEY)
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

    def get_course_module_records(self, collection_name):
        all_records = []

        next_page_offset = "first"
        offset = None

        while next_page_offset:
            if next_page_offset != "first":
                offset = next_page_offset

            records = self.client.scroll(
                collection_name=collection_name,
                with_payload=True,
                with_vectors=False,
                limit=10,
                offset=offset,
            )

            next_page_offset = records[1]

            all_records.extend(records[0])

        courses_records = sorted(
            [
                record
                for record in all_records
                if "module_id" not in record.payload
                and "course_id" in record.payload
                and isinstance(record.payload["course_id"], int)
            ],
            key=lambda x: x.payload["course_id"],
        )
        modules_records = sorted(
            [record for record in all_records if "module_id" in record.payload], key=lambda x: x.payload["module_id"]
        )

        return courses_records, modules_records

    def check_if_course_exists(self, course_id: int) -> bool:
        """Check if a course exists in the database."""

        filter_criteria = {"must": [{"key": "course_id", "match": {"value": course_id}}]}
        return bool(self.query_with_filter("web_assistant", filter_criteria))

    def check_if_module_exists(self, module_id: int) -> bool:
        """Check if a module exists in the database."""

        filter_criteria = {"must": [{"key": "module_id", "match": {"value": module_id}}]}
        return bool(VectorDBQdrant().query_with_filter("web_assistant", filter_criteria))

    def query_with_filter(self, collection_name, filter_criteria) -> list[dict]:
        records = self.client.scroll(
            collection_name=collection_name,
            scroll_filter=filter_criteria,
            with_payload=True,
            with_vectors=False,
            limit=10,
        )

        return records


if __name__ == "__main__":
    test_connection = VectorDBQdrant(version="disk")  # For local testing only
    print(test_connection.get_metadata("web_assistant"))
