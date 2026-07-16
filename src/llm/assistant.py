import logging
import threading
import uuid
from collections import OrderedDict
from langfuse.decorators import observe, langfuse_context
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from src.api.models.serializable_chat_message import SerializableChatMessage
from src.env import env
from src.llm.objects.LLMs import Models
from src.llm.objects.question_answerer import get_fallback_type
from src.llm.state.models import GraphState, RerankerType
from src.llm.tools.contextualize import contextualize_and_route
from src.llm.graphs.no_vector_db import build_no_vectordb_graph
from src.llm.graphs.simple_hop import build_simple_hop_graph
from src.llm.graphs.summarize import build_summarize_graph
from src.llm.graphs.socratic import build_socratic_graph
from src.llm.graphs.socratic_v2 import build_socratic_v2_graph

logger = logging.getLogger(__name__)


class BoundedMemorySaver(MemorySaver):
    """MemorySaver mit maximalem Thread-Limit (FIFO-Verdrängung).

    Verhindert unbegrenztes Speicherwachstum bei langer Backend-Laufzeit.
    Sobald max_threads überschritten wird, fliegt der älteste Thread raus.
    """

    def __init__(self, max_threads: int = 500):
        super().__init__()
        self.max_threads = max_threads
        self._thread_order: OrderedDict[str, None] = OrderedDict()
        # Guards _thread_order and the eviction below — put()/aput() are called
        # from concurrent request threads (and aput from the event loop), and
        # OrderedDict's check-then-mutate sequence here is not atomic on its own.
        self._lock = threading.Lock()

    def _register_and_evict(self, thread_id: str) -> None:
        with self._lock:
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
    
    def __init__(
        self,
        rerank_top_n: int = 5,
        retrieve_top_n: int = 10,
        enable_socratic: bool = False,
        enable_socratic_v2: bool = False,
        reranker_type: RerankerType = "llm",
        min_reranker_score: float = 0.0,
    ):
        """
        Initialize the assistant with system configuration.

        Args:
            rerank_top_n: Number of top chunks to keep after reranking
            retrieve_top_n: Number of chunks to retrieve from vector database
            enable_socratic: Enable/disable the socratic learning mode (v1)
            enable_socratic_v2: Enable/disable the redesigned socratic learning
                mode ("Lernmodus v2") — independent of v1 so both can be
                compared side by side
            reranker_type: Which reranker to use — "llm" (default), "azure_semantic", "bge"
            min_reranker_score: Relevance cutoff on a normalized 0-1 scale, applied by all
                backends (0 = no filtering). If every chunk falls below the cutoff, the
                answer node produces the no-answer fallback instead of a weak answer.
        """
        if rerank_top_n >= retrieve_top_n:
            raise ValueError(
                f"rerank_top_n ({rerank_top_n}) must be less than retrieve_top_n ({retrieve_top_n})"
            )

        # System configuration (hardcoded, not exposed to frontend)
        self.system_config = {
            "rerank_top_n": rerank_top_n,
            "retrieve_top_n": retrieve_top_n,
            "enable_socratic": enable_socratic,
            "enable_socratic_v2": enable_socratic_v2,
            "reranker_type": reranker_type,
            "min_reranker_score": min_reranker_score,
        }
        
        # In-memory persistence for the duration of the backend runtime.
        # No database persistence — conversations are lost on server restart.
        # Bounded via MAX_CHAT_THREADS; oldest conversation is evicted when limit is exceeded.
        self.checkpointer = BoundedMemorySaver(max_threads=env.MAX_CHAT_THREADS)

        # Compile main router graph
        self.graph = self._build_main_graph()
    
    def _build_main_graph(self) -> StateGraph:
        """
        Builds the main router graph that routes to scenario-specific subgraphs.
        
        Flow:
        START → contextualize_and_route → [conditional routing based on mode] → END
        
        Scenarios:
        - no_vectordb: Conversational queries
        - simple_hop: Single-hop RAG retrieval
        - summarize: Full-scope content summary (course/module), no reranking
        - socratic: Guided learning (v1)
        - socratic_v2: Guided learning ("Lernmodus v2", redesigned tutor)
        """
        # Compile subgraphs
        no_vectordb_graph = build_no_vectordb_graph()
        simple_hop_graph = build_simple_hop_graph()
        summarize_graph = build_summarize_graph()
        socratic_graph = build_socratic_graph()
        socratic_v2_graph = build_socratic_v2_graph()

        # Main router graph
        graph = StateGraph(GraphState)

        # Add contextualize/routing node
        graph.add_node("contextualize_and_route", contextualize_and_route)

        # Add subgraph nodes
        graph.add_node("no_vectordb", no_vectordb_graph)
        graph.add_node("simple_hop", simple_hop_graph)
        graph.add_node("summarize", summarize_graph)
        graph.add_node("socratic", socratic_graph)
        graph.add_node("socratic_v2", socratic_v2_graph)

        # Start with contextualization and routing
        graph.add_edge(START, "contextualize_and_route")

        # Conditional routing based on mode
        def route_by_mode(state: GraphState) -> str:
            """Route to appropriate subgraph based on classified scenario."""
            mode = state["mode"]
            # Special case: exit_complete skips directly to END --> used when exiting socratic mode
            if mode == "exit_complete":
                return END
            return mode  # Returns "no_vectordb", "simple_hop", "summarize" or "socratic"

        graph.add_conditional_edges(
            "contextualize_and_route",
            route_by_mode
        )

        # All subgraphs end at END
        graph.add_edge("no_vectordb", END)
        graph.add_edge("simple_hop", END)
        graph.add_edge("summarize", END)
        graph.add_edge("socratic", END)
        graph.add_edge("socratic_v2", END)

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
        module_id: int | list[int] | None = None,
        start_socratic: bool = False,
        start_socratic_v2: bool = False,
    ) -> tuple[GraphState, dict, str]:
        """
        Lädt bestehenden State aus Checkpoint oder erstellt neuen Initial State.

        Args:
            query: User's question
            model: LLM model to use
            thread_id: Optional thread ID for persistent conversations
            course_id: Optional course ID filter
            module_id: Optional module ID filter — a single ID or a list of IDs
            start_socratic: Explicit request-level trigger to enter the socratic mode
            start_socratic_v2: Explicit request-level trigger to enter the socratic v2 mode

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

                # Limitiere Chat-History auf letzte N Nachrichten. Aktive
                # v2-Lernsessions brauchen den ganzen Sessionverlauf (Learner
                # Model, keine Fragenwiederholung) — 6 Nachrichten reichen
                # dort nicht.
                history_limit = (
                    env.CHAT_HISTORY_LIMIT_SOCRATIC_V2
                    if checkpoint.values.get("socratic_v2_phase")
                    else env.CHAT_HISTORY_LIMIT
                )
                limited_existing_history = self.limit_chat_history(existing_history, limit=history_limit)

                state_update: GraphState = {
                    "user_query": query,
                    "chat_history": limited_existing_history,
                    "runtime_config": {
                        "model": model,
                        "course_id": course_id,
                        "module_id": module_id,
                        "thread_id": thread_id,
                        "start_socratic": start_socratic,
                        "start_socratic_v2": start_socratic_v2,
                    },
                    # Reset per-turn intermediate artifacts so stale checkpoint
                    # values never bleed into the new graph run.
                    # (pending_scope_escalation is deliberately NOT reset — it
                    # carries the module→course escalation offer into exactly
                    # this next turn and is consumed/cleared by
                    # contextualize_and_route.)
                    "retrieved": [],
                    "retrieval_semantic_ranked": False,
                    "reranked": [],
                    "scope_escalated": False,
                    "answer": None,
                    "citations_markdown": None,
                    "detected_language": None,
                    "contextualized_query": None,
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
                "start_socratic": start_socratic,
                "start_socratic_v2": start_socratic_v2,
            },
            "system_config": self.system_config
        }

        return initial_state, config, thread_id

    def _update_langfuse_trace(
        self,
        *,
        surface: str,
        query: str,
        thread_id: str,
        result: GraphState,
        assistant_content: str,
    ) -> None:
        """Reichert den aktuellen Langfuse-Trace mit Analyse-Daten an.

        - session_id = thread_id → Konversationen erscheinen gruppiert in der
          Langfuse Sessions-View.
        - Name/Tags (mode, model, reranker, surface) → filterbar in der Trace-Liste.
        - Metadaten (Chunk-Zahlen, Sprache, contextualized_query, Fallback-Typ)
          → Diagnose einzelner Requests ohne Span-Deep-Dive.
        - Score "no_answer_fallback" (0/1) → Fallback-Quote als Metrik/Dashboard.

        Darf niemals den Chat-Pfad brechen — Fehler werden nur geloggt.
        """
        try:
            mode = result.get("mode")
            runtime_config = result.get("runtime_config", {})
            model = runtime_config.get("model")
            model_name = model.value if isinstance(model, Models) else str(model)
            reranker_type = self.system_config.get("reranker_type")
            # Fallback-Texte kommen nie mit Citations, daher gegen das rohe
            # "answer"-Feld klassifizieren (nicht gegen citations_markdown).
            fallback_type = get_fallback_type(result.get("answer"))

            # Anfrage-Scope: fragt das Frontend global, kursweit oder auf ein
            # konkretes Modul eingeschränkt? Als Tag direkt filter-/zählbar.
            course_id = runtime_config.get("course_id")
            module_id = runtime_config.get("module_id")
            # Truthy check, not `is not None`: an empty module_id list means
            # "no module filter", same as None.
            if module_id:
                scope = "module"
            elif course_id is not None:
                scope = "course"
            else:
                scope = "global"

            tags = [
                f"surface:{surface}",
                f"scope:{scope}",
                f"model:{model_name}",
                f"reranker:{reranker_type}",
            ]
            if mode:
                tags.append(f"mode:{mode}")
            if fallback_type:
                tags.append("fallback")
            if result.get("scope_escalated"):
                tags.append("scope_escalated")

            langfuse_context.update_current_trace(
                name=f"chat:{mode}" if mode else "chat",
                session_id=thread_id,
                input=query,
                output=assistant_content,
                tags=tags,
                metadata={
                    "thread_id": thread_id,
                    "surface": surface,
                    "scope": scope,
                    "mode": mode,
                    "socratic_mode": result.get("socratic_mode"),
                    "socratic_v2_phase": result.get("socratic_v2_phase"),
                    "model": model_name,
                    "course_id": course_id,
                    "module_id": module_id,
                    "detected_language": result.get("detected_language"),
                    "contextualized_query": result.get("contextualized_query"),
                    "history_len": len(result.get("chat_history") or []),
                    "n_retrieved": len(result.get("retrieved") or []),
                    "n_reranked": len(result.get("reranked") or []),
                    "retrieval_semantic_ranked": result.get("retrieval_semantic_ranked"),
                    "scope_escalated": bool(result.get("scope_escalated")),
                    "fallback_type": fallback_type,
                    "system_config": self.system_config,
                },
            )
            langfuse_context.score_current_trace(
                name="no_answer_fallback",
                value=1 if fallback_type else 0,
                data_type="BOOLEAN",
                comment=fallback_type,
            )
        except Exception:
            logger.warning("Langfuse trace enrichment failed", exc_info=True)

    @observe()
    def chat(
        self,
        query: str,
        model: Models,
        thread_id: str | None = None,
        start_socratic: bool = False,
        start_socratic_v2: bool = False,
    ) -> tuple[SerializableChatMessage, str]:
        """
        Chat with general bot about drupal and functions of ki-campus.
        For frontend integrated in Drupal.

        Args:
            query: User's question
            model: LLM model to use
            thread_id: Optional thread ID for persistent conversations
                - If provided: Loads state from checkpoint
                - If None: Creates new conversation with generated ID
            start_socratic: Explicit request-level trigger to enter the socratic mode
            start_socratic_v2: Explicit request-level trigger to enter the socratic v2 mode

        Returns:
            tuple: (SerializableChatMessage with answer, thread_id)
        """
        # Lade oder erstelle State
        state, config, thread_id = self._get_or_create_state(
            query=query,
            model=model,
            thread_id=thread_id,
            start_socratic=start_socratic,
            start_socratic_v2=start_socratic_v2,
        )

        # Session früh setzen, damit auch Fehler-Traces (Exception im Graph)
        # bereits der Konversation zugeordnet sind. Die vollständige
        # Anreicherung passiert nach dem Graph-Lauf.
        langfuse_context.update_current_trace(session_id=thread_id, input=query)

        # Execute graph mit State (update oder initial)
        result = self.graph.invoke(state, config=config)

        # Generiere Assistant-Response
        assistant_content = result.get("citations_markdown") or result.get("answer") or ""
        assistant_message = SerializableChatMessage(role="assistant", content=assistant_content)

        self._update_langfuse_trace(
            surface="drupal",
            query=query,
            thread_id=thread_id,
            result=result,
            assistant_content=assistant_content,
        )

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
        module_id: int | list[int] | None = None,
        thread_id: str | None = None,
        start_socratic: bool = False,
        start_socratic_v2: bool = False,
    ) -> tuple[SerializableChatMessage, str]:
        """
        Chat with the contents of a specific course and optionally submodule(s).
        For frontend hosted on Moodle.

        Args:
            query: User's question
            model: LLM model to use
            course_id: Moodle course ID to filter by
            module_id: Optional module/topic ID (or list of IDs) within the course.
                All given module IDs are assumed to belong to course_id.
            thread_id: Optional thread ID for persistent conversations
                - If provided: Loads state from checkpoint
                - If None: Creates new conversation with generated ID
            start_socratic: Explicit request-level trigger to enter the socratic mode
            start_socratic_v2: Explicit request-level trigger to enter the socratic v2 mode

        Returns:
            tuple: (SerializableChatMessage with answer, thread_id)
        """
        # Lade oder erstelle State
        state, config, thread_id = self._get_or_create_state(
            query=query,
            model=model,
            thread_id=thread_id,
            course_id=course_id,
            module_id=module_id,
            start_socratic=start_socratic,
            start_socratic_v2=start_socratic_v2,
        )

        # Session früh setzen, damit auch Fehler-Traces (Exception im Graph)
        # bereits der Konversation zugeordnet sind. Die vollständige
        # Anreicherung passiert nach dem Graph-Lauf.
        langfuse_context.update_current_trace(session_id=thread_id, input=query)

        # Execute graph mit State (update oder initial)
        result = self.graph.invoke(state, config=config)

        # Generiere Assistant-Response
        assistant_content = result.get("citations_markdown") or result.get("answer") or ""
        assistant_message = SerializableChatMessage(role="assistant", content=assistant_content)

        self._update_langfuse_trace(
            surface="moodle",
            query=query,
            thread_id=thread_id,
            result=result,
            assistant_content=assistant_content,
        )

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
