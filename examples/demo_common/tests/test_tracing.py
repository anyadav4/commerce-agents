# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""LangSmith tracing is opt-in, leaves the agent alone when off, and nests a backend call
under the turn that made it. The tracing client here records runs and sends nothing."""

import json
from types import SimpleNamespace

import pytest
import requests
from anthropic import AsyncAnthropic
from langsmith import Client, tracing_context

from demo_common.tracing import trace_backend_call, trace_turns, traced_client, tracing_enabled


class RecordingSession(requests.Session):
    def __init__(self) -> None:
        super().__init__()
        self.runs: list[dict] = []

    def request(self, method, url, **kwargs):
        body = kwargs.get("data") or kwargs.get("json")
        if isinstance(body, bytes | str):
            body = json.loads(body)
        if isinstance(body, dict) and "name" in body:
            self.runs.append(body)
        response = requests.Response()
        response.status_code = 200
        response._content = b"{}"
        return response


@pytest.fixture
def tracing_on(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "not-a-real-key")


@pytest.fixture(autouse=True)
def tracing_off_by_default(monkeypatch):
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)


def test_tracing_is_off_unless_asked_for(monkeypatch):
    assert not tracing_enabled() and traced_client() is None
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    assert traced_client() is None


def test_a_traced_client_is_an_anthropic_client(tracing_on):
    client = traced_client(api_key="not-a-real-key", timeout=5)
    assert isinstance(client, AsyncAnthropic) and client.timeout == 5


def test_the_agent_is_left_alone_when_tracing_is_off():
    agent = SimpleNamespace(stream_turn=lambda: None)
    original = agent.stream_turn
    trace_turns(agent, "chat")
    assert agent.stream_turn is original


async def test_a_backend_call_nests_under_its_turn_and_records_no_session(tracing_on):
    class Backend:
        @trace_backend_call("search", inputs=lambda query, **_: {"query": query})
        async def search(self, session, query):
            return [SimpleNamespace(product_id="AR-1"), SimpleNamespace(product_id="AR-2")]

    class Agent:
        backend = Backend()

        async def stream_turn(self, messages, session, state):
            yield SimpleNamespace(type="text")
            await self.backend.search(session, "tent")
            yield SimpleNamespace(type="turn_complete")

    agent = Agent()
    trace_turns(agent, "chat")
    recorder = RecordingSession()
    client = Client(
        api_url="http://langsmith.invalid",
        api_key="not-a-real-key",
        session=recorder,
        auto_batch_tracing=False,
    )
    session = SimpleNamespace(session_id="secret-session-id")
    with tracing_context(enabled=True, client=client):
        events = [
            event
            async for event in agent.stream_turn(
                [{"role": "user", "content": "a tent"}], session, None
            )
        ]
    assert [event.type for event in events] == ["text", "turn_complete"]

    turn = next(run for run in recorder.runs if run["name"] == "chat" and run.get("inputs"))
    search = next(run for run in recorder.runs if run["name"] == "search" and run.get("inputs"))
    assert turn.get("parent_run_id") is None and search["parent_run_id"] == turn["id"]
    assert turn["inputs"]["message"] == "a tent"
    assert "secret-session-id" not in json.dumps(recorder.runs)
    assert search["inputs"] == {"query": "tent"}
