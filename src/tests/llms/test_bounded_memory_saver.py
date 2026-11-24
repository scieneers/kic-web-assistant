"""
Unit- und Integrationstests für BoundedMemorySaver.

Testet:
1. Eviction-Tracking: Ältester Thread wird aus _thread_order entfernt
2. Speicher-Cleanup: Ältester Thread wird aus MemorySaver-Storage entfernt
3. Idempotenz: Gleiche thread_id mehrfach registriert → kein doppelter Eintrag
4. End-to-End via LangGraph: Konversation bleibt nach Eviction nicht mehr abrufbar
"""

from typing import TypedDict

import pytest
from langgraph.graph import END, START, StateGraph

from src.llm.assistant import BoundedMemorySaver


# ---------------------------------------------------------------------------
# Hilfsgraph für Integrationstests
# ---------------------------------------------------------------------------

class _State(TypedDict, total=False):
    value: str


def _append_node(state: _State) -> _State:
    return {"value": (state.get("value") or "") + "!"}


def _make_graph(saver: BoundedMemorySaver):
    g = StateGraph(_State)
    g.add_node("node", _append_node)
    g.add_edge(START, "node")
    g.add_edge("node", END)
    return g.compile(checkpointer=saver)


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


# ---------------------------------------------------------------------------
# Unit-Tests: _register_and_evict
# ---------------------------------------------------------------------------

class TestRegisterAndEvict:
    def test_first_thread_registered(self):
        saver = BoundedMemorySaver(max_threads=3)
        saver._register_and_evict("t1")
        assert "t1" in saver._thread_order

    def test_duplicate_not_added_twice(self):
        saver = BoundedMemorySaver(max_threads=3)
        saver._register_and_evict("t1")
        saver._register_and_evict("t1")
        assert list(saver._thread_order.keys()) == ["t1"]

    def test_within_limit_no_eviction(self):
        saver = BoundedMemorySaver(max_threads=3)
        for i in range(3):
            saver._register_and_evict(f"t{i}")
        assert len(saver._thread_order) == 3

    def test_exceeding_limit_evicts_oldest(self):
        saver = BoundedMemorySaver(max_threads=2)
        saver._register_and_evict("t1")
        saver._register_and_evict("t2")
        saver._register_and_evict("t3")  # t1 fliegt raus
        assert "t1" not in saver._thread_order
        assert "t2" in saver._thread_order
        assert "t3" in saver._thread_order
        assert len(saver._thread_order) == 2

    def test_fifo_order(self):
        saver = BoundedMemorySaver(max_threads=2)
        for tid in ["t1", "t2", "t3", "t4"]:
            saver._register_and_evict(tid)
        # Nur die zwei jüngsten sollten übrig sein
        assert list(saver._thread_order.keys()) == ["t3", "t4"]


# ---------------------------------------------------------------------------
# Integrationstests: End-to-End via LangGraph
# ---------------------------------------------------------------------------

class TestBoundedMemorySaverIntegration:
    def test_checkpoint_stored_and_retrievable(self):
        saver = BoundedMemorySaver(max_threads=5)
        graph = _make_graph(saver)
        cfg = _config("thread-a")

        graph.invoke({"value": "hello"}, config=cfg)
        state = graph.get_state(cfg)
        assert state.values.get("value") == "hello!"

    def test_evicted_thread_no_longer_in_storage(self):
        saver = BoundedMemorySaver(max_threads=2)
        graph = _make_graph(saver)

        # Thread 1 anlegen
        graph.invoke({"value": "first"}, config=_config("t1"))
        # Thread 2 anlegen
        graph.invoke({"value": "second"}, config=_config("t2"))
        # Thread 3 anlegen → t1 wird verdrängt
        graph.invoke({"value": "third"}, config=_config("t3"))

        # t1 darf nicht mehr im internen Storage sein
        assert "t1" not in saver.storage

    def test_surviving_threads_still_accessible(self):
        saver = BoundedMemorySaver(max_threads=2)
        graph = _make_graph(saver)

        graph.invoke({"value": "first"}, config=_config("t1"))
        graph.invoke({"value": "second"}, config=_config("t2"))
        graph.invoke({"value": "third"}, config=_config("t3"))  # t1 evicted

        state_t2 = graph.get_state(_config("t2"))
        state_t3 = graph.get_state(_config("t3"))
        assert state_t2.values.get("value") == "second!"
        assert state_t3.values.get("value") == "third!"

    def test_max_threads_one_only_last_survives(self):
        saver = BoundedMemorySaver(max_threads=1)
        graph = _make_graph(saver)

        for i in range(5):
            graph.invoke({"value": f"msg{i}"}, config=_config(f"t{i}"))

        # Nur t4 sollte noch im Storage sein
        assert len(saver._thread_order) == 1
        assert "t4" in saver._thread_order
