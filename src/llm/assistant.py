import uuid
from collections import OrderedDict
from langfuse.decorators import observe, langfuse_context
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from src.api.models.serializable_chat_message import SerializableChatMessage
from src.llm.objects.LLMs import Models
from src.llm.state.models import GraphState
from src.llm.tools.contextualize import contextualize_and_route
from src.llm.graphs.no_vector_db import build_no_vectordb_graph
from src.llm.graphs.simple_hop import build_simple_hop_graph
from src.llm.graphs.multi_hop import build_multi_hop_graph
from src.llm.graphs.socratic import build_socratic_graph


class BoundedMemorySaver(MemorySaver):
    """MemorySaver mit maximalem Thread-Limit (FIFO-Verdrängung).

    Verhindert unbegrenztes Speicherwachstum bei langer Backend-Laufzeit.
    Sobald max_threads überschritten wird, fliegt der älteste Thread raus.
    """

    def __init__(self, max_threads: int = 500):
        super().__init__()
        self.max_threads = max_threads
        self._thread_order: OrderedDict[str, None] = OrderedDict()

    def _register_and_evict(self, thread_id: str) -> None:
        if thread_id in self._thread_order:
            return
        self._thread_order[thread_id] = None
        if len(self._thread_order) > self.max_threads:
            oldest_id, _ = self._thread_order.popitem(last=False)
            # MemorySaver speichert intern unter self.storage und self.writes
            try:
                self.storage.pop(oldest_id, None)
            except AttributeError:
                pass
            try:
                self.writes.pop(oldest_id, None)
            except AttributeError:
                pass

    def put(self, config, checkpoint, metadata, new_versions):
        self._register_and_evict(config["configurable"]["thread_id"])
        return super().put(config, checkpoint, metadata, new_versions)

    async def aput(self, config, checkpoint, metadata, new_versions):
        self._register_and_evict(config["configurable"]["thread_id"])
        return await super().aput(config, checkpoint, metadata, new_versions)


class KICampusAssistant:
    """
    Main RAG assistant orchestrator using LangGraph.
    
    Routes queries to appropriate subgraphs based on scenario classification.
    """
    
    def __init__(self, rerank_top_n: int = 5, retrieve_top_n: int = 10, enable_socratic: bool = False):
        """
        Initialize the assistant with system configuration.

        Args:
            rerank_top_n: Number of top chunks to keep after reranking
            retrieve_top_n: Number of chunks to retrieve from vector database
            enable_socratic: Enable/disable the socratic learning mode
        """
        # System configuration (hardcoded, not exposed to frontend)
        self.system_config = {
            "rerank_top_n": rerank_top_n,
            "retrieve_top_n": retrieve_top_n,
            "enable_socratic": enable_socratic,
        }
        
        # In-memory persistence for the duration of the backend runtime.
        # No database persistence — conversations are lost on server restart.
        # Bounded to 500 threads; oldest conversation is evicted when limit is exceeded.
        self.checkpointer = BoundedMemorySaver(max_threads=500)

        # Compile main router graph
        self.graph = self._build_main_graph()
    
    def _build_main_graph(self) -> StateGraph:
        """
        Builds the main router graph that routes to scenario-specific subgraphs.
        
        Flow:
        START → contextualize_and_route → [conditional routing based on mode] → END
        
        Scenarios:
        - no_vectordb: Conversational queries
        - simple_hop: Standard RAG
        - multi_hop: Complex multi-retrieval (placeholder)
        - socratic: Guided learning (placeholder)
        """
        # Compile subgraphs
        no_vectordb_graph = build_no_vectordb_graph()
        simple_hop_graph = build_simple_hop_graph()
        multi_hop_graph = build_multi_hop_graph()
        socratic_graph = build_socratic_graph()
        
        # Main router graph
        graph = StateGraph(GraphState)
        
        # Add contextualize/routing node
        graph.add_node("contextualize_and_route", contextualize_and_route)
        
        # Add subgraph nodes
        graph.add_node("no_vectordb", no_vectordb_graph)
        graph.add_node("simple_hop", simple_hop_graph)
        graph.add_node("multi_hop", multi_hop_graph)
        graph.add_node("socratic", socratic_graph)
        
        # Start with contextualization and routing
        graph.add_edge(START, "contextualize_and_route")
        
        # Conditional routing based on mode
        def route_by_mode(state: GraphState) -> str:
            """Route to appropriate subgraph based on classified scenario."""
            mode = state["mode"]
            # Special case: exit_complete skips directly to END --> used when exiting socratic mode
            if mode == "exit_complete":
                return END
            return mode  # Returns "no_vectordb", "simple_hop", "multi_hop" or "socratic"
        
        graph.add_conditional_edges(
            "contextualize_and_route",
            route_by_mode
        )
        
        # All subgraphs end at END
        graph.add_edge("no_vectordb", END)
        graph.add_edge("simple_hop", END)
        graph.add_edge("multi_hop", END)
        graph.add_edge("socratic", END)
        
        return graph.compile(checkpointer=self.checkpointer)

    @observe()
    def limit_chat_history(self, chat_history: list[SerializableChatMessage], limit: int) -> list[SerializableChatMessage]:
        """Limit chat history to last N messages to save context window."""
        if len(chat_history) > limit:
            chat_history = chat_history[-limit:]
        return chat_history

    @observe()
    def _get_or_create_state(
        self, 
        query: str, 
        model: Models,
        thread_id: str | None,
        course_id: int | None = None,
        module_id: int | None = None,
    ) -> tuple[GraphState, dict, str]:
        """
        Lädt bestehenden State aus Checkpoint oder erstellt neuen Initial State.
        
        Args:
            query: User's question
            model: LLM model to use
            thread_id: Optional thread ID for persistent conversations
            course_id: Optional course ID filter
            module_id: Optional module ID filter
        
        Returns:
            tuple: (state_update, config, thread_id)
        """
        # Generiere oder nutze bestehende thread_id
        thread_id = thread_id or str(uuid.uuid4())
        
        config = {
            "configurable": {
                "thread_id": thread_id
            }
        }
        
        # Versuche, bestehenden State zu laden
        try:
            checkpoint = self.graph.get_state(config)

            if checkpoint and checkpoint.values:
                # State existiert → Nur neue Query + runtime_config updaten
                # Lade bestehende chat_history (OHNE neue User-Message, die kommt später)
                existing_history = checkpoint.values.get("chat_history", [])

                # Limitiere Chat-History auf letzte 6 Nachrichten
                limited_existing_history = self.limit_chat_history(existing_history, limit=6)

                state_update: GraphState = {
                    "user_query": query,
                    "chat_history": limited_existing_history,
                    "runtime_config": {
                        "model": model,
                        "course_id": course_id,
                        "module_id": module_id,
                        "thread_id": thread_id,
                    }
                }
                return state_update, config, thread_id

        except Exception:
            # Checkpoint existiert nicht oder Fehler beim Laden
            pass
        
        # Kein State vorhanden → Erstelle Initial State
        initial_state: GraphState = {
            "user_query": query,
            "chat_history": [],
            "runtime_config": {
                "model": model,
                "course_id": course_id,
                "module_id": module_id,
                "thread_id": thread_id,
            },
            "system_config": self.system_config
        }

        return initial_state, config, thread_id

    @observe()
    def chat(self, query: str, model: Models, thread_id: str | None = None) -> tuple[SerializableChatMessage, str]:
        """
        Chat with general bot about drupal and functions of ki-campus.
        For frontend integrated in Drupal.
        
        Args:
            query: User's question
            model: LLM model to use
            thread_id: Optional thread ID for persistent conversations
                - If provided: Loads state from checkpoint
                - If None: Creates new conversation with generated ID
            
        Returns:
            tuple: (SerializableChatMessage with answer, thread_id)
        """
        # Lade oder erstelle State
        state, config, thread_id = self._get_or_create_state(
            query=query,
            model=model,
            thread_id=thread_id
        )
        
        # Allow easier tracing of conversations in Langfuse
        langfuse_context.update_current_observation(
            metadata={
                "thread_id": thread_id
            }
        )

        # Execute graph mit State (update oder initial)
        result = self.graph.invoke(state, config=config)
        
        # Generiere Assistant-Response
        assistant_content = result.get("citations_markdown") or result.get("answer") or ""
        assistant_message = SerializableChatMessage(role="assistant", content=assistant_content)
        
        # Füge User-Message und Assistant-Message zur History hinzu
        user_message = SerializableChatMessage(role="user", content=query)
        updated_history = result["chat_history"] + [user_message, assistant_message]
        
        # Update State mit finaler chat_history
        self.graph.update_state(
            config=config,
            values={"chat_history": updated_history}
        )

        # Return SerializableChatMessage and thread_id
        return (assistant_message, thread_id)

    @observe()
    def chat_with_course(
        self,
        query: str,
        model: Models,
        course_id: int | None = None,
        module_id: int | None = None,
        thread_id: str | None = None,
    ) -> tuple[SerializableChatMessage, str]:
        """
        Chat with the contents of a specific course and optionally submodule.
        For frontend hosted on Moodle.
        
        Args:
            query: User's question
            model: LLM model to use
            course_id: Moodle course ID to filter by
            module_id: Optional module/topic ID within course
            thread_id: Optional thread ID for persistent conversations
                - If provided: Loads state from checkpoint
                - If None: Creates new conversation with generated ID
            
        Returns:
            tuple: (SerializableChatMessage with answer, thread_id)
        """
        # Lade oder erstelle State
        state, config, thread_id = self._get_or_create_state(
            query=query,
            model=model,
            thread_id=thread_id,
            course_id=course_id,
            module_id=module_id
        )
        
        # Allow easier tracing of conversations in Langfuse
        langfuse_context.update_current_observation(
            metadata={
                "thread_id": thread_id
            }
        )

        # Execute graph mit State (update oder initial)
        result = self.graph.invoke(state, config=config)
        
        # Generiere Assistant-Response
        assistant_content = result.get("citations_markdown") or result.get("answer") or ""
        assistant_message = SerializableChatMessage(role="assistant", content=assistant_content)
        
        # Füge User-Message und Assistant-Message zur History hinzu
        user_message = SerializableChatMessage(role="user", content=query)
        updated_history = result["chat_history"] + [user_message, assistant_message]
        
        # Update State mit finaler chat_history
        self.graph.update_state(
            config=config,
            values={"chat_history": updated_history}
        )

        # Return SerializableChatMessage and thread_id
        return (assistant_message, thread_id)


    def get_chat_history(self, thread_id: str) -> list[SerializableChatMessage]:
        """Gibt die gespeicherte Konversation zurück. Leer wenn thread_id unbekannt."""
        config = {"configurable": {"thread_id": thread_id}}
        try:
            checkpoint = self.graph.get_state(config)
            if checkpoint and checkpoint.values:
                return checkpoint.values.get("chat_history", [])
        except Exception:
            pass
        return []


if __name__ == "__main__":
    assistant = KICampusAssistant()
    assistant.chat(query="Eklär über den Kurs Deep Learning mit Tensorflow, Keras und Tensorflow.js", model=Models.AZURE_FALLBACK)
