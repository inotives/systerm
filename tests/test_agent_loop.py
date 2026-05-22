import json
from pathlib import Path

import httpx
import pytest

from systerm.agent_loop import AgentLoop
from systerm.config import AppConfig
from systerm.providers import OpenAICompatibleClient
from systerm.storage import SessionStore
from systerm.tools import SHELL_TOOL


@pytest.mark.asyncio
async def test_agent_loop_runs_low_risk_shell_tool_call(tmp_path: Path) -> None:
    config = AppConfig.model_validate(
        {
            "models": {"default_model": "fast", "fallback_models": []},
            "providers": {"test": {"base_url": "https://example.test/v1", "supports_tools": True}},
            "model_profiles": {"fast": {"provider": "test", "model": "fast-model"}},
        }
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "shell", "arguments": '{"command": "echo loop-ok"}'},
                                }
                            ],
                        }
                    }
                ]
            },
            request=request,
        )

    published: list[tuple[str, dict[str, object], int | None]] = []

    async def publish(event_type: str, payload: dict[str, object], session_id: int | None) -> None:
        published.append((event_type, payload, session_id))

    store = SessionStore(tmp_path / "systerm.db")
    await store.init()
    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = OpenAICompatibleClient(config, http_client=http_client)
        result = await AgentLoop(client, store, {"shell": SHELL_TOOL}, publish).run(
            "use a tool",
            requested_model="fast",
        )

    assert result.stop_reason == "tool-use"
    assert result.content == "loop-ok\n"
    sessions = await store.list_sessions()
    assert sessions[0]["message_count"] == 2
    assert await store.list_messages(sessions[0]["id"]) == [
        {"role": "user", "content": "use a tool"},
        {"role": "tool", "content": "loop-ok\n"},
    ]
    tool_calls = await store.list_tool_calls(sessions[0]["id"])
    assert len(tool_calls) == 1
    assert await store.list_tool_results(tool_calls[0].id)
    event_types = [event[0] for event in published]
    assert "session.created" in event_types
    assert "message.created" in event_types
    assert "tool_call.created" in event_types
    assert "tool_result.created" in event_types


@pytest.mark.asyncio
async def test_agent_loop_pauses_on_approval_required_tool_call(tmp_path: Path) -> None:
    config = AppConfig.model_validate(
        {
            "models": {"default_model": "fast", "fallback_models": []},
            "providers": {"test": {"base_url": "https://example.test/v1", "supports_tools": True}},
            "model_profiles": {"fast": {"provider": "test", "model": "fast-model"}},
        }
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "shell", "arguments": '{"command": "python script.py"}'},
                                }
                            ],
                        }
                    }
                ]
            },
            request=request,
        )

    store = SessionStore(tmp_path / "systerm.db")
    await store.init()
    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = OpenAICompatibleClient(config, http_client=http_client)
        result = await AgentLoop(client, store, {"shell": SHELL_TOOL}).run("use a tool", requested_model="fast")

    assert result.stop_reason == "approval-required"
    approvals = await store.list_approvals()
    assert len(approvals) == 1
    assert approvals[0].status == "pending"
    sessions = await store.list_sessions()
    assert await store.list_session_approvals(sessions[0]["id"]) == approvals


@pytest.mark.asyncio
async def test_agent_loop_appends_to_existing_session(tmp_path: Path) -> None:
    config = AppConfig.model_validate(
        {
            "models": {"default_model": "fast", "fallback_models": []},
            "providers": {"test": {"base_url": "https://example.test/v1", "supports_tools": False}},
            "model_profiles": {"fast": {"provider": "test", "model": "fast-model"}},
        }
    )
    requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "second reply"}}]},
            request=request,
        )

    store = SessionStore(tmp_path / "systerm.db")
    await store.init()
    session = await store.create_session()
    await store.add_message(session.id, "user", "first")
    await store.add_message(session.id, "assistant", "first reply")
    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(transport=transport) as http_client:
        client = OpenAICompatibleClient(config, http_client=http_client)
        result = await AgentLoop(client, store, {}).run(
            "second",
            requested_model="fast",
            session_id=session.id,
        )

    sessions = await store.list_sessions()
    messages = await store.list_messages(session.id)
    assert result.session_id == session.id
    assert len(sessions) == 1
    assert messages == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "first reply"},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "second reply"},
    ]
    assert requests[0]["messages"] == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "first reply"},
        {"role": "user", "content": "second"},
    ]
