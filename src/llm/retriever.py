from langfuse.decorators import observe
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.vector_stores import VectorStoreQuery

from src.llm.LLMs import LLM
from src.vectordb.qdrant import VectorDBQdrant, models


class KiCampusRetriever:
    def __init__(self):
        self.embedder = LLM().get_embedder()
        self.vector_store = VectorDBQdrant("prod_remote").as_llama_vector_store(collection_name="web_assistant")
        super().__init__()

    @observe()
    def retrieve(self, query: str, course_id: int | None = None, module_id: int | None = None) -> list[TextNode]:
        embedding = self.embedder.get_query_embedding(query)

        conditions = []

        if course_id is None and module_id is None:
            conditions.append(
                models.FieldCondition(
                    key="source",
                    match=models.MatchText(text="Drupal"),
                )
            )

        if course_id is not None:
            conditions.append(
                models.FieldCondition(
                    key="course_id",
                    match=models.MatchValue(value=course_id),
                )
            )

        if module_id is not None:
            conditions.append(
                models.FieldCondition(
                    key="module_id",
                    match=models.MatchValue(value=module_id),
                )
            )

        filter = models.Filter(must=conditions) if conditions else None

        vector_store_query = VectorStoreQuery(query_embedding=embedding, similarity_top_k=10)

        query_result = self.vector_store.query(vector_store_query, qdrant_filters=filter)

        if query_result.nodes is None:
            return []

        for node in query_result.nodes:
            # how node gets rendered as context for the llm
            # alternatively create a custom node post-processor and pass to query engine https://docs.llamaindex.ai/en/stable/module_guides/querying/node_postprocessors/root.html
            node.text_template = "{metadata_str}\nContent: {content}"

        return query_result.nodes
