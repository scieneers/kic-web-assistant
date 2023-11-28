
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http.models import PointStruct
from qdrant_client.http.models import Filter, FieldCondition, MatchValue


class VectorDBQdrant():
    def __init__(self, collection_name, vector_size):
        self.collection_name = collection_name
        self.client = QdrantClient(":memory:")
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.DOT),
        )
    
    def upsert(self, points: list[dict]):
        qdrant_points = [PointStruct(**point) for point in points]
        operation_info = self.client.upsert(
            collection_name=self.collection_name,
            wait=True,
            points=qdrant_points,
        )
        print(operation_info)

    def search(self, query_vector, query_filter=None, with_payload=False, limit=3):
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=query_filter,
            with_payload=with_payload,
            limit=limit,
        )
        return search_result