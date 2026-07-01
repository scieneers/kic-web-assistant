"""Utilities for request-scoped streaming.

We intentionally keep streaming control out of LangGraph state, because the
graph is checkpointed and expects JSON-serializable state.

Instead, we use a ContextVar that can be set by the FastAPI streaming endpoint.
Deep inside the LLM layer we can check whether a token callback is present.
"""

from __future__ import annotations

import contextvars
import re
from typing import Callable, Optional


TokenCallback = Callable[[str], None]
CitationResolve = Callable[[str], str]  # [docN] marker → replacement string (or "" to drop)


# Matches a complete citation marker like [doc12].
_COMPLETE_CITATION_RE = re.compile(r"\[doc\d+\]")
# Literal prefix every citation marker starts with.
_CITATION_PREFIX = "[doc"


def _is_partial_citation(s: str) -> bool:
    """True if `s` (which starts with '[') could still grow into a [docN] marker.

    Examples that are partial: "[", "[d", "[do", "[doc", "[doc1", "[doc12".
    Examples that are not:     "[x", "[doc1x" (diverged from the marker shape).
    """
    if len(s) <= len(_CITATION_PREFIX):
        return _CITATION_PREFIX.startswith(s)
    if not s.startswith(_CITATION_PREFIX):
        return False
    # After "[doc" we expect only digits and no closing bracket yet (a closed
    # marker would already have matched _COMPLETE_CITATION_RE).
    return s[len(_CITATION_PREFIX):].isdigit()


class CitationStreamFilter:
    """Strips [docN] citation markers from a token stream on the fly.

    Tokens can split a marker across chunks (e.g. "[doc", "1", "]"), so we keep a
    small buffer and only release text that can no longer become part of a marker.
    Complete [docN] markers are dropped from the live stream; the final answer
    (with clickable title links) is delivered separately as the non-streamed
    result, so the markers are never lost — only hidden while streaming.
    """

    def __init__(self) -> None:
        self._buffer = ""

    def feed(self, delta: str) -> str:
        """Append `delta` and return the text that is safe to emit right now."""
        self._buffer += delta
        out = ""
        while self._buffer:
            bracket = self._buffer.find("[")
            if bracket == -1:
                # No '[' left -> the whole buffer is safe to emit.
                out += self._buffer
                self._buffer = ""
                break
            # Everything before the first '[' is safe to emit.
            out += self._buffer[:bracket]
            self._buffer = self._buffer[bracket:]
            # Buffer now starts with '['.
            match = _COMPLETE_CITATION_RE.match(self._buffer)
            if match:
                # Drop the complete marker and keep scanning.
                self._buffer = self._buffer[match.end():]
                continue
            if _is_partial_citation(self._buffer):
                # Might still complete into a marker -> hold back for more tokens.
                break
            # A '[' that cannot be a citation -> emit it and continue scanning.
            out += "["
            self._buffer = self._buffer[1:]
        return out

    def flush(self) -> str:
        """Emit whatever remains once the stream ends (e.g. an unclosed marker)."""
        out = self._buffer
        self._buffer = ""
        return out


class CitationStreamResolver:
    """Resolves [docN] citation markers in a token stream on the fly.

    Like CitationStreamFilter but calls a provided ``resolve`` callable on each
    complete marker instead of dropping it.  The resolved text (or "" to drop)
    is forwarded directly to ``callback``.

    Tracks everything emitted via ``emitted_text`` so callers can use the
    resolved output as the authoritative final text.
    """

    def __init__(self, resolve: CitationResolve, callback: TokenCallback) -> None:
        self._resolve = resolve
        self._callback = callback
        self._buffer = ""
        self._emitted = ""

    def feed(self, delta: str) -> None:
        self._buffer += delta
        out = ""
        while self._buffer:
            bracket = self._buffer.find("[")
            if bracket == -1:
                out += self._buffer
                self._buffer = ""
                break
            out += self._buffer[:bracket]
            self._buffer = self._buffer[bracket:]
            match = _COMPLETE_CITATION_RE.match(self._buffer)
            if match:
                out += self._resolve(match.group())
                self._buffer = self._buffer[match.end():]
                continue
            if _is_partial_citation(self._buffer):
                break
            out += "["
            self._buffer = self._buffer[1:]
        if out:
            self._emitted += out
            self._callback(out)

    def flush(self) -> None:
        if self._buffer:
            self._emitted += self._buffer
            self._callback(self._buffer)
            self._buffer = ""

    @property
    def emitted_text(self) -> str:
        """The accumulated text that was emitted to the callback so far."""
        return self._emitted


class SmartStreamCallback:
    """Combines NO ANSWER FOUND sentinel detection with CitationStreamResolver.

    Buffers only the first ~20 characters. Once more text arrives it's clear
    the response is a real answer, so the buffer is flushed and real-time mode
    takes over. If the stream ends with exactly "NO ANSWER FOUND", the buffered
    sentinel is discarded and the friendly message is sent instead.

    Must be called in this order:
        1. feed(delta) — for every streaming token
        2. flush()    — called by LLMs.py at end of stream (before NO ANSWER check)
        3. finalize() — called by question_answerer AFTER NO ANSWER check
    """

    _SENTINEL = "NO ANSWER FOUND"
    _THRESHOLD = len(_SENTINEL) + 5  # buffer until we're sure it's not the sentinel

    def __init__(self, resolver: "CitationStreamResolver", outer_callback: TokenCallback) -> None:
        self._resolver = resolver
        self._outer_callback = outer_callback
        self._peeking = True
        self._peek_buffer: list[str] = []
        self._accumulated_len = 0

    def feed(self, delta: str) -> None:
        if not self._peeking:
            self._resolver.feed(delta)
            return
        self._peek_buffer.append(delta)
        self._accumulated_len += len(delta)
        if self._accumulated_len > self._THRESHOLD:
            self._switch_to_realtime()

    def _switch_to_realtime(self) -> None:
        self._peeking = False
        for t in self._peek_buffer:
            self._resolver.feed(t)
        self._peek_buffer = []

    def flush(self) -> None:
        """Called by LLMs.py when the stream ends — hold off if still peeking."""
        if not self._peeking:
            self._resolver.flush()

    def finalize(self, is_no_answer: bool, friendly_message: str = "") -> None:
        """Called by question_answerer AFTER the NO ANSWER FOUND check.

        For sentinel responses: discards buffered tokens silently.  The caller
        is responsible for setting the correct response.content; that value
        reaches the user via the ``final`` event in rest.py — NOT as a streaming
        token here.  Emitting the friendly message as a token caused a race
        where it arrived in the queue before (or interleaved with) real streamed
        content, corrupting the display.
        For normal responses: switches to real-time (if still peeking) and
        flushes the resolver's internal citation buffer.
        """
        if is_no_answer:
            self._peek_buffer = []
            # Deliberately do NOT call outer_callback here.
            # The friendly message is delivered by the ``final`` NDJSON event.
        else:
            if self._peeking:
                self._switch_to_realtime()
            self._resolver.flush()


# If set (per-request), the LLM layer will push generated token deltas into this callback.
token_callback_var: contextvars.ContextVar[Optional[TokenCallback]] = contextvars.ContextVar(
    "kic_token_callback",
    default=None,
)


# When set, LLMs.py uses this resolver instead of CitationStreamFilter.
citation_resolver_var: contextvars.ContextVar[Optional[CitationStreamResolver]] = contextvars.ContextVar(
    "kic_citation_resolver",
    default=None,
)


# Gate streaming so only the *final* user-facing answer is streamed.
#
# Without this, internal LLM calls (routing/contextualization) will also stream
# their (often short) outputs like "simple_hop".
stream_phase_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "kic_stream_phase",
    default="none",  # none | final
)


class TokenCallbackContext:
    """Convenience context-manager to set/unset the token callback."""

    def __init__(self, callback: Optional[TokenCallback]):
        self._callback = callback
        self._token: Optional[contextvars.Token] = None

    def __enter__(self):
        self._token = token_callback_var.set(self._callback)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._token is not None:
            token_callback_var.reset(self._token)
        return False


class StreamPhaseContext:
    """Context manager to set the current streaming phase.

    Use this to enable streaming only around the final answer generation.
    """

    def __init__(self, phase: str):
        self._phase = phase
        self._token: Optional[contextvars.Token] = None

    def __enter__(self):
        self._token = stream_phase_var.set(self._phase)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._token is not None:
            stream_phase_var.reset(self._token)
        return False
