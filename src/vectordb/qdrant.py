
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http.models import PointStruct
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from llama_index.vector_stores.qdrant import QdrantVectorStore
from src.env import EnvHelper

class VectorDBQdrant():
    def __init__(self, version='disk'):
        self.version = version
        if version == 'memory':
            self.client = QdrantClient(':memory:')
        elif version == 'disk':
            self.client = QdrantClient('localhost', port=6333)
            try:
                _ = self.client.get_collections()
            except ResponseHandlingException as e:
                print('Qdrant container not running? Run:')
                print('docker run -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_storage:/qdrant/storage:z qdrant/qdrant:v1.6.1')
                raise e
        elif version == 'remote':
            self.client = QdrantClient(url=EnvHelper().QDRANT_URL, 
                                       port=443,
                                       api_key=EnvHelper().QDRANT_TOKEN)
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
    

if __name__ == '__main__':
    test_connection = VectorDBQdrant(version='remote')
    print(test_connection.client.get_collections())
