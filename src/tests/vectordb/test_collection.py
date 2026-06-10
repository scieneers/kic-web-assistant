import uuid

import pytest
from azure.core.exceptions import ResourceNotFoundError

from src.env import env
from src.vectordb.azure_search import VectorDBAzureSearch


@pytest.mark.integration
def test_index_creation():
    """Integration test: create a throwaway index, confirm it exists, delete it.

    Requires a reachable Azure AI Search endpoint and valid credentials
    (`az login` locally, with the caller's IP in allowed_ip_ranges).
    """
    if env.AZURE_SEARCH_ENDPOINT == "UNSET":
        pytest.skip("AZURE_SEARCH_ENDPOINT not configured")

    db = VectorDBAzureSearch()
    index_name = f"test-{uuid.uuid4().hex[:8]}"

    db.create_index(index_name, vector_size=100)
    # Should now exist (no exception).
    db.index_client.get_index(index_name)

    db.index_client.delete_index(index_name)
    with pytest.raises(ResourceNotFoundError):
        db.index_client.get_index(index_name)
