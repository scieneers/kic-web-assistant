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


# If set (per-request), the LLM layer will push generated token deltas into this callback.
token_callback_var: contextvars.ContextVar[Optional[TokenCallback]] = contextvars.ContextVar(
    "kic_token_callback",
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
