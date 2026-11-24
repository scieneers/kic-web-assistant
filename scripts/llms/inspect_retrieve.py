"""Quick test for retrieve_chunks function."""

from llama_index.core.schema import TextNode
from src.llm.state.models import GraphState
from src.llm.tools.retrieve import retrieve_chunks
from src.llm.objects.LLMs import Models


# Create initial state
state = GraphState(
    user_query="Was ist Deep Learning?",
    chat_history=[],
    contextualized_query="Was ist Deep Learning?",
    runtime_config={
        "model": Models.AZURE_FALLBACK,
    },
    system_config={
        "rerank_top_n": 5,
    }
)

print("🔍 Testing retrieve_chunks...")
print(f"Query: {state['user_query']}\n")

# Execute retrieval node
result_state = retrieve_chunks(state)

# Show results
print(f"✅ Retrieved {len(result_state['retrieved'])} chunks\n")
print("="*80)

for idx, node in enumerate(result_state['retrieved'], 1):
    print(f"\n📄 Chunk {idx}:")
    print(f"   Text: {node.text[:200]}...")
    print(f"   Score: {getattr(node, 'score', 'N/A')}")
    print(f"   Metadata:")
    for key, value in node.metadata.items():
        print(f"      • {key}: {value}")

print("\n" + "="*80)
print("✅ Test complete!")
