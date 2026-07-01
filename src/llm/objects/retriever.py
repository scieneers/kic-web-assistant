import json
import logging

from langfuse.decorators import observe

from src.env import env
from src.llm.objects.LLMs import LLM
from src.vectordb.azure_search import VectorDBAzureSearch
from src.api.models.serializable_text_node import SerializableTextNode

logger = logging.getLogger(__name__)


def _build_odata_filter(
    course_id: int | list[int] | tuple[int, ...] | None,
    module_id: int | None,
) -> str:
    """Translate the retrieval filter into an Azure AI Search OData expression.

    Filter logic:
      - always exclude internal ModuleFingerprint bookkeeping docs
      - with no course/module given, restrict to Drupal content
      - course_id may be a single value or a list (OData ``search.in``)
      - module_id is an exact match
    """
    clauses: list[str] = ["type ne 'ModuleFingerprint'"]
    reasons: list[str] = ["always: exclude bookkeeping docs"]

    if course_id is None and module_id is None:
        clauses.append("source eq 'Drupal'")
        reasons.append("no course/module given: restrict to Drupal content")

    if course_id is not None:
        if isinstance(course_id, (list, tuple)):
            ids = ",".join(str(int(c)) for c in course_id)
            clauses.append(f"search.in(course_id, '{ids}', ',')")
            reasons.append(f"course_id list provided: filter to courses {list(course_id)}")
        else:
            clauses.append(f"course_id eq {int(course_id)}")
            reasons.append(f"course_id provided: filter to course {course_id}")

    if module_id is not None:
        clauses.append(f"module_id eq {int(module_id)}")
        reasons.append(f"module_id provided: filter to module {module_id}")

    odata_filter = " and ".join(clauses)
    logger.debug(
        "OData filter built: %r\n  Clauses:\n%s",
        odata_filter,
        "\n".join(f"    - {r}" for r in reasons),
    )
    return odata_filter


class KiCampusRetriever:
    def __init__(self, use_hybrid: bool = True, n_chunks: int = 10):
        """Initialize retriever.

        Args:
            use_hybrid: If True, hybrid search (vector + BM25 keyword, fused via
                       Azure's Reciprocal Rank Fusion). If False, vector-only.
            n_chunks: Number of chunks to retrieve from the search index.
        """
        self.use_hybrid = use_hybrid
        self.n_chunks = n_chunks
        self.embedder = LLM().get_embedder()
        self.vector_db = VectorDBAzureSearch()
        self.index_name = env.AZURE_SEARCH_INDEX

    @observe()
    def retrieve(
        self, query: str, course_id: int | None = None, module_id: int | None = None
    ) -> list[SerializableTextNode]:
        """Retrieve relevant documents from Azure AI Search.

        Args:
            query: Search query
            course_id: Optional filter by course ID
            module_id: Optional filter by module ID

        Returns:
            List of relevant SerializableTextNodes
        """
        odata_filter = _build_odata_filter(course_id, module_id)
        dense_embedding = self.embedder.get_query_embedding(query)

        # Hybrid passes the query text (BM25 side); vector-only suppresses it.
        results = self.vector_db.hybrid_search(
            query_text=query if self.use_hybrid else None,
            query_vector=dense_embedding,
            index_name=self.index_name,
            odata_filter=odata_filter,
            top=self.n_chunks,
        )

        return [self._to_node(result) for result in results]

    @staticmethod
    def _to_node(result: dict) -> SerializableTextNode:
        """Rebuild a SerializableTextNode from an Azure AI Search result.

        The full original node metadata is restored losslessly from the
        ``metadata_json`` blob rather than reassembled from individual fields.
        """
        raw_metadata = result.get("metadata_json")
        metadata = json.loads(raw_metadata) if raw_metadata else {}
        return SerializableTextNode(
            text=result.get("text", ""),
            id_=str(result.get("id")) if result.get("id") is not None else None,
            metadata=metadata,
            score=result.get("@search.score"),
        )
