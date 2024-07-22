from langfuse.decorators import observe
from llama_index.core import QueryBundle
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index_client import FilterCondition, MetadataFilter, MetadataFilters

from src.env import EnvHelper
from src.vectordb.qdrant import VectorDBQdrant


class KiCampusRetriever:
    def __init__(self):
        secrets = EnvHelper()

        self.embedder = AzureOpenAIEmbedding(
            model=secrets.AZURE_OPENAI_EMBEDDER_MODEL,
            deployment_name=secrets.AZURE_OPENAI_EMBEDDER_DEPLOYMENT,
            api_key=secrets.AZURE_OPENAI_EMBEDDER_API_KEY,
            azure_endpoint=secrets.AZURE_OPENAI_EMBEDDER_ENDPOINT,
            api_version=secrets.AZURE_OPENAI_EMBEDDER_API_VERSION,
        )

        self.vector_store = VectorDBQdrant("remote").as_llama_vector_store(collection_name="web_assistant")
        super().__init__()

    @observe()
    def retrieve(self, query: str, course_id: int|None=None) -> list[NodeWithScore]:
        embedding = self.embedder.get_query_embedding(query)

        filters = MetadataFilters(filters=[], condition=FilterCondition.AND)
        if course_id is not None:
            filters.filters.append(MetadataFilter(key="course_id", value=course_id))

        vector_store_query = VectorStoreQuery(query_embedding=embedding, similarity_top_k=3, filters=filters)

        query_result = self.vector_store.query(vector_store_query)
        if query_result.nodes is None:
            return []

        nodes_with_scores = []
        for index, node in enumerate(query_result.nodes):
            # how node gets rendered as context for the llm
            # alternatively create a custom node post-processor and pass to query engine https://docs.llamaindex.ai/en/stable/module_guides/querying/node_postprocessors/root.html
            node.text_template = "{metadata_str}\nContent: {content}"

            score: list[float] = None
            if query_result.similarities is not None:
                score = query_result.similarities[index]
            nodes_with_scores.append(NodeWithScore(node=node, score=score))

        return nodes_with_scores
