import pytest
from qdrant_client.http.exceptions import UnexpectedResponse

from src.vectordb.qdrant import VectorDBQdrant


def test_collection_creation():
    remote_db = VectorDBQdrant(version="remote")
    remote_db.create_collection("test", vector_size=100)
    remote_db.client.get_collection("test")
    assert True == remote_db.client.delete_collection("test")
    with pytest.raises(UnexpectedResponse) as excinfo:
        remote_db.client.get_collection("test")
