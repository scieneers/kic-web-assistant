from typing import Dict, Any, Optional
from pydantic import BaseModel
from llama_index.core.schema import TextNode


class SerializableTextNode(BaseModel):
    """
    Serializable version of llama_index TextNode for State persistence.
    
    Contains only the essential attributes needed for RAG operations
    and Langfuse observability.
    """
    text: str
    metadata: Dict[str, Any]
    score: Optional[float] = None
    id_: Optional[str] = None

    def to_text_node(self) -> TextNode:
        """Convert back to llama_index TextNode for component usage."""
        kwargs = {"text": self.text, "metadata": self.metadata}
        if self.id_ is not None:
            kwargs["id_"] = self.id_
        node = TextNode(**kwargs)
        # Note: score is kept in SerializableTextNode for logging/observability
        # but not transferred to TextNode (TextNode doesn't have a score field)
        return node

    @staticmethod
    def from_text_node(node: TextNode) -> "SerializableTextNode":
        """Create from llama_index TextNode."""
        return SerializableTextNode(
            text=node.text,
            metadata=node.metadata,
            score=getattr(node, 'score', None),
            id_=node.id_
        )
