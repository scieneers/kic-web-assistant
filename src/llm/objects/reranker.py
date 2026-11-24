from langfuse.decorators import observe
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.schema import NodeWithScore, TextNode

from src.llm.objects.LLMs import LLM, Models
from src.api.models.serializable_text_node import SerializableTextNode


class Reranker:
    """LLM-based reranker for improving retrieval precision.
    
    Uses LlamaIndex's LLMRerank internally to score and reorder nodes
    based on their relevance to the query.
    """
    
    def __init__(self, top_n: int, max_chars_per_node: int = 999999):
        """Initialize the reranker.
        
        Args:
            top_n: Number of top results to return after reranking
            max_chars_per_node: Maximum characters per node text to avoid token limits
        """
        self.llm = LLM()
        self.top_n = top_n
        self.max_chars_per_node = max_chars_per_node
        if self.max_chars_per_node <= 1500:
            self.choice_batch_size = 10
        else:
            self.choice_batch_size = 5
    
    def _truncate(self, text: str) -> str:
        """Truncate text to max_chars_per_node to reduce token usage.
        
        Args:
            text: Text to truncate
            
        Returns:
            Truncated text if longer than max_chars_per_node, otherwise unchanged
        """
        if not text:
            return text
        return text[:self.max_chars_per_node] if len(text) > self.max_chars_per_node else text
    
    @observe(name="rerank")
    def rerank(
        self,
        query: str,
        nodes: list[TextNode],
        model: Models,
        raise_on_error: bool = False,
    ) -> list[SerializableTextNode]:
        """Rerank nodes based on query relevance using LLM.

        Args:
            query: The user's query
            nodes: List of retrieved nodes to rerank (SerializableTextNode or TextNode)
            model: Which LLM model to use for reranking (from LLM selection logic)
            raise_on_error: If True, LLM failures raise instead of silently
                falling back to the original order. Production keeps the
                fallback (a degraded answer beats a crashed request); the
                benchmark uses strict mode so a failed rerank is counted as a
                failure instead of polluting the metrics with passthrough
                results.

        Returns:
            Reranked list of nodes (top_n best matches)
        """
        # No nodes no rerank
        if not nodes:
            return []
        # One node no rerank
        if len(nodes) == 1:
            return nodes[:self.top_n]
        
        # Get LLM instance for reranking
        llm = self.llm.get_model(model)
        
        # Initialize LLMRerank with chosen model
        llm_rerank = LLMRerank(
            llm=llm,
            top_n=self.top_n,
            choice_batch_size=self.choice_batch_size,  # limit batch size
        )
        
        # LLMRerank expects NodeWithScore with TextNode
        # Convert SerializableTextNode to TextNode if needed
        # Truncate long texts to avoid token limits and reduce latency
        nodes_with_score = []
        for node in nodes:
            # Convert SerializableTextNode to TextNode if necessary
            if isinstance(node, SerializableTextNode):
                text_node = node.to_text_node()
                score = node.score or 0.0
            else:
                text_node = node
                score = getattr(node, 'score', 0.0)
            
            # Truncate text on TextNode
            text_node.text = self._truncate(text_node.text or "")
            nodes_with_score.append(NodeWithScore(node=text_node, score=score))
        
        # Perform reranking with error handling
        try:
            reranked_nodes = llm_rerank.postprocess_nodes(
                nodes=nodes_with_score,
                query_str=query
            )
        except Exception as e:
            if raise_on_error:
                raise
            # If reranking fails (network, Azure, parsing errors), return original nodes
            print(f"Reranking failed (model={model}, nodes={len(nodes)}): {e}")
            # Convert original nodes to SerializableTextNode
            return [SerializableTextNode.from_text_node(n) for n in nodes[:self.top_n]]
        
        # Convert TextNode to SerializableTextNode and preserve rerank scores
        result = []
        for nws in reranked_nodes:
            stn = SerializableTextNode.from_text_node(nws.node)
            # Preserve rerank score
            try:
                stn.score = float(nws.score)
            except (AttributeError, ValueError, TypeError):
                pass  # Keep original score if setting fails
            result.append(stn)
        
        return result[:self.top_n]