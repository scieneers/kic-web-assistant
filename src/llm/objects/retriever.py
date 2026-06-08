from langfuse.decorators import observe
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.vector_stores import VectorStoreQuery
from qdrant_client.models import Prefetch, Query, Fusion, FusionQuery

from src.env import env
from src.llm.objects.LLMs import LLM
from fastembed import SparseTextEmbedding
from src.vectordb.qdrant import VectorDBQdrant, models
from src.api.models.serializable_text_node import SerializableTextNode


class KiCampusRetriever:
    def __init__(self, use_hybrid: bool = True, n_chunks: int = 10):
        """Initialize retriever with optional hybrid search.
        
        Args:
            use_hybrid: If True, uses both dense and sparse vectors for retrieval.
                       If False, uses only dense vectors (legacy mode).
            n_chunks: Number of chunks to retrieve from vector database.
        """
        self.use_hybrid = use_hybrid
        self.n_chunks = n_chunks
        self.embedder = LLM().get_embedder()
        
        if use_hybrid:
            self.sparse_encoder = SparseTextEmbedding("Qdrant/bm42-all-minilm-l6-v2-attentions")
            # For hybrid search, we use direct Qdrant client instead of LlamaIndex wrapper
            self.vector_db = VectorDBQdrant()
            self.collection_name = env.QDRANT_COLLECTION
        else:
            self.collection_name = env.QDRANT_COLLECTION
            self.vector_store = VectorDBQdrant().as_llama_vector_store(collection_name=env.QDRANT_COLLECTION)

    @observe()
    def retrieve(self, query: str, course_id: int | None = None, module_id: int | None = None) -> list[SerializableTextNode]:
        """Retrieve relevant documents using hybrid search (dense + sparse vectors).
        
        Args:
            query: Search query
            course_id: Optional filter by course ID
            module_id: Optional filter by module ID
            
        Returns:
            List of relevant TextNodes
        """
        if self.use_hybrid:
            return self._retrieve_hybrid(query, course_id, module_id)
        else:
            return self._retrieve_dense_only(query, course_id, module_id)
    
    def _retrieve_dense_only(self, query: str, course_id: int | list[int] | tuple[int, ...] | None, module_id: int | None) -> list[SerializableTextNode]:
        """Legacy dense-only retrieval using LlamaIndex wrapper."""

        # Generate query embedding
        embedding = self.embedder.get_query_embedding(query)

        # Build filter conditions
        conditions = []

        # Exclude internal bookkeeping points (e.g. ModuleFingerprint) from retrieval.
        # We do this via must_not so it doesn't change existing retrieval behavior
        # (e.g. Drupal-only retrieval when no course/module filters are given).
        must_not = [
            models.FieldCondition(
                key="type",
                match=models.MatchValue(value="ModuleFingerprint"),
            )
        ]

        if course_id is None and module_id is None:
            conditions.append(
                models.FieldCondition(
                    key="source",
                    match=models.MatchValue(value="Drupal"),
                )
            )

        if course_id is not None:
            # allow list/tuple of course_ids; falls back to single value
            if isinstance(course_id, (list, tuple)):
                conditions.append(
                    models.FieldCondition(
                        key="course_id",
                        match=models.MatchAny(any=list(course_id)),
                    )
                )
            else:
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

        filter = models.Filter(must=conditions, must_not=must_not) if (conditions or must_not) else None

        # Perform vector store query
        vector_store_query = VectorStoreQuery(query_embedding=embedding, similarity_top_k=self.n_chunks)

        # Get results
        query_result = self.vector_store.query(vector_store_query, qdrant_filters=filter)

        if query_result.nodes is None:
            return []

        # Convert to SerializableTextNode
        return [SerializableTextNode.from_text_node(node) for node in query_result.nodes]
    
    def _retrieve_hybrid(self, query: str, course_id: int | list[int] | tuple[int, ...] | None, module_id: int | None) -> list[SerializableTextNode]:
        """Hybrid retrieval using both dense and sparse vectors.
        
        Qdrant automatically performs fusion (Reciprocal Rank Fusion) when both
        dense and sparse query vectors are provided via prefetch.
        """
        # Generate dense embedding
        dense_embedding = self.embedder.get_query_embedding(query)
        
        # Generate sparse embedding
        sparse_result = list(self.sparse_encoder.embed([query]))[0]
        sparse_embedding = models.SparseVector(
            indices=sparse_result.indices.tolist(),
            values=sparse_result.values.tolist()
        )
        
        # Build filter conditions
        conditions = []
        
        must_not = [
            models.FieldCondition(
                key="type",
                match=models.MatchValue(value="ModuleFingerprint"),
            )
        ]

        if course_id is None and module_id is None:
            conditions.append(
                models.FieldCondition(
                    key="source",
                    match=models.MatchValue(value="Drupal"),
                )
            )
        
        if course_id is not None:
            if isinstance(course_id, (list, tuple)):
                conditions.append(
                    models.FieldCondition(
                        key="course_id",
                        match=models.MatchAny(any=list(course_id)),
                    )
                )
            else:
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
        
        query_filter = models.Filter(must=conditions, must_not=must_not) if (conditions or must_not) else None
        
        # Hybrid search using prefetch + fusion
        # Qdrant performs automatic RRF (Reciprocal Rank Fusion)
        
        search_results = self.vector_db.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                Prefetch(
                    query=dense_embedding,
                    using="dense",
                    limit=self.n_chunks * 3,  # Get more candidates for fusion
                    filter=query_filter,
                ),
                Prefetch(
                    query=sparse_embedding,
                    using="sparse",
                    limit=self.n_chunks * 3,  # Get more candidates for fusion
                    filter=query_filter,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=self.n_chunks,  # Final top-k after fusion
            with_payload=True,
        )
        
        # Convert Qdrant results to SerializableTextNodes
        nodes = []
        for result in search_results.points:
            # Extract text from payload
            text = result.payload.get("text", result.payload.get("content", ""))
            
            # Create metadata without text/content (avoid duplication)
            metadata = {k: v for k, v in result.payload.items() if k not in ("text", "content")}
            
            # Create SerializableTextNode
            node = SerializableTextNode(
                text=text,
                id_=str(result.id),
                metadata=metadata,
                score=result.score if hasattr(result, 'score') else None,
            )
            
            nodes.append(node)
        
        return nodes