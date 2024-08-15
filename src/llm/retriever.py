from langfuse.decorators import observe
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores import (
    FilterCondition,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
)

from src.llm.LLMs import LLM
from src.vectordb.qdrant import VectorDBQdrant


class KiCampusRetriever:
    def __init__(self):
        self.embedder = LLM().get_embedder()
        self.vector_store = VectorDBQdrant("remote").as_llama_vector_store(collection_name="web_assistant")
        super().__init__()

    @observe()
    def retrieve(self, query: str, course_id: int | None = None, module_id: int | None = None) -> list[NodeWithScore]:
        embedding = self.embedder.get_query_embedding(query)

        filters = MetadataFilters(filters=[], condition=FilterCondition.AND)
        if course_id is not None:
            filters.filters.append(MetadataFilter(key="course_id", value=course_id))
        if module_id is not None:
            filters.filters.append(MetadataFilter(key="module_id", value=module_id))

        vector_store_query = VectorStoreQuery(query_embedding=embedding, similarity_top_k=12, filters=filters)

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
