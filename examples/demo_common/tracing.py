# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""Optional LangSmith tracing for an example API. With ``LANGSMITH_TRACING=true`` and
``LANGSMITH_API_KEY`` set, one chat turn is one trace: every model call (through a wrapped
Anthropic client) and every traced backend call (``trace_backend_call``) nests under it.
Nothing changes when tracing is off. A trace holds what the model saw and said, customer
text included, so it goes to the LangSmith project the environment names (``LANGSMITH_PROJECT``,
``LANGSMITH_ENDPOINT``) and nowhere else."""

from __future__ import annotations

import logging
import os
from collections import Counter
from collections.abc import Callable
from typing import Any, TypeVar

from anthropic import AsyncAnthropic

from commerce_common.turn import session_tag

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])
_MAX_TEXT = 500


def tracing_enabled() -> bool:
    return os.environ.get("LANGSMITH_TRACING", "").lower() in {"1", "true"}


def traced_client(**options: Any) -> AsyncAnthropic | None:
    """An ``AsyncAnthropic(**options)`` whose calls are traced, or None when tracing is off
    (the agent then builds its own client)."""
    if not tracing_enabled():
        return None
    try:
        from langsmith.wrappers import wrap_anthropic
    except ImportError:
        logger.warning("LANGSMITH_TRACING is on but langsmith is not installed")
        return None
    return wrap_anthropic(AsyncAnthropic(**options))


def _last_text(messages: list[dict[str, Any]]) -> str:
    content = messages[-1].get("content") if messages else ""
    return (content if isinstance(content, str) else str(content))[:_MAX_TEXT]


def trace_turns(agent: Any, name: str) -> None:
    """Make each ``agent.stream_turn`` call the root run of a trace named ``name``. The run
    records the last message and the session's log tag, and the count of each event type
    the turn produced."""
    if not tracing_enabled():
        return
    try:
        from langsmith import traceable
    except ImportError:
        return
    agent.stream_turn = traceable(
        name=name,
        run_type="chain",
        process_inputs=lambda inputs: {
            "message": _last_text(inputs.get("messages", [])),
            "session": session_tag(getattr(inputs.get("session"), "session_id", None)),
        },
        process_outputs=lambda outputs: {
            "events": dict(
                Counter(
                    getattr(event, "type", "?")
                    for event in (outputs.get("output") if isinstance(outputs, dict) else outputs)
                    or []
                )
            )
        },
    )(agent.stream_turn)


def trace_backend_call(name: str, *, inputs: Callable[..., dict[str, Any]]) -> Callable[[F], F]:
    """Decorator for a backend method: a ``retriever`` run named ``name`` under the turn's
    trace. ``inputs`` receives the call's arguments by name and returns what the run
    records, so the backend object and the session never enter a trace. Returns the method
    unchanged when ``langsmith`` is not installed; the run is a no-op while tracing is off."""

    def decorate(method: F) -> F:
        try:
            from langsmith import traceable
        except ImportError:
            return method
        return traceable(  # type: ignore[return-value]
            name=name,
            run_type="retriever",
            process_inputs=lambda arguments: inputs(**arguments),
            process_outputs=lambda outputs: {
                "product_ids": [
                    getattr(item, "product_id", str(item))
                    for item in (outputs.get("output") if isinstance(outputs, dict) else outputs)
                    or []
                ]
            },
        )(method)

    return decorate
