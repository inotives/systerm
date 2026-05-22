import json
from pathlib import Path

import httpx
import pytest

from systerm.daemon_client import DaemonClient, DaemonClientConfig, DaemonClientError, load_daemon_client_config


def test_load_daemon_client_config_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SYSTERM_DAEMON_TOKEN", "token")
    monkeypatch.setenv("SYSTERM_DAEMON_URL", "http://127.0.0.1:9999/")

    config = load_daemon_client_config()

    assert config.base_url == "http://127.0.0.1:9999"
    assert config.token == "token"
    assert config.websocket_url == "ws://127.0.0.1:9999/events"


def test_load_daemon_client_config_from_token_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SYSTERM_DAEMON_TOKEN", raising=False)
    path = tmp_path / "token"
    path.write_text("file-token", encoding="utf-8")

    config = load_daemon_client_config(base_url="https://example.test", token_path=path)

    assert config.token == "file-token"
    assert config.websocket_url == "wss://example.test/events"


def test_load_daemon_client_config_requires_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SYSTERM_DAEMON_TOKEN", raising=False)

    with pytest.raises(DaemonClientError):
        load_daemon_client_config(token_path=tmp_path / "missing")


@pytest.mark.asyncio
async def test_daemon_client_fetches_session() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"id": 3, "messages": []})

    client = DaemonClient(
        DaemonClientConfig(base_url="http://daemon.test", token="token"),
        transport=httpx.MockTransport(handler),
    )

    session = await client.session(3)

    assert session == {"id": 3, "messages": []}
    assert requests[0].url == "http://daemon.test/sessions/3"


@pytest.mark.asyncio
async def test_daemon_client_fetches_sessions_trace_and_approvals() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/sessions":
            if request.method == "POST":
                return httpx.Response(200, json={"id": 4})
            return httpx.Response(200, json=[])
        if request.url.path == "/sessions/3/trace":
            return httpx.Response(200, json={"id": 3, "messages": [], "tool_calls": [], "approvals": []})
        if request.url.path == "/approvals":
            return httpx.Response(200, json=[])
        return httpx.Response(404)

    client = DaemonClient(
        DaemonClientConfig(base_url="http://daemon.test", token="token"),
        transport=httpx.MockTransport(handler),
    )

    assert await client.sessions() == []
    assert await client.create_session() == {"id": 4}
    assert await client.session_trace(3) == {"id": 3, "messages": [], "tool_calls": [], "approvals": []}
    assert await client.approvals(status="all") == []
    assert [str(request.url) for request in requests] == [
        "http://daemon.test/sessions",
        "http://daemon.test/sessions",
        "http://daemon.test/sessions/3/trace",
        "http://daemon.test/approvals?status=all",
    ]


@pytest.mark.asyncio
async def test_daemon_client_creates_job_for_session() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"id": 9, "status": "queued", "session_id": 7})

    client = DaemonClient(
        DaemonClientConfig(base_url="http://daemon.test", token="token"),
        transport=httpx.MockTransport(handler),
    )

    job = await client.create_job_for_session("hello", 7)

    assert job["session_id"] == 7
    assert json.loads(requests[0].content) == {"prompt": "hello", "session_id": 7}


@pytest.mark.asyncio
async def test_daemon_client_manages_jobs() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/jobs" and request.method == "GET":
            return httpx.Response(200, json=[])
        if request.url.path == "/jobs/3" and request.method == "GET":
            return httpx.Response(200, json={"id": 3, "status": "failed"})
        if request.url.path == "/jobs/3/cancel":
            return httpx.Response(200, json={"id": 3, "status": "canceled"})
        if request.url.path == "/jobs/3/retry":
            return httpx.Response(200, json={"id": 4, "status": "queued"})
        return httpx.Response(404)

    client = DaemonClient(
        DaemonClientConfig(base_url="http://daemon.test", token="token"),
        transport=httpx.MockTransport(handler),
    )

    assert await client.jobs() == []
    assert await client.job(3) == {"id": 3, "status": "failed"}
    assert await client.cancel_job(3) == {"id": 3, "status": "canceled"}
    assert await client.retry_job(3) == {"id": 4, "status": "queued"}
    assert [str(request.url) for request in requests] == [
        "http://daemon.test/jobs",
        "http://daemon.test/jobs/3",
        "http://daemon.test/jobs/3/cancel",
        "http://daemon.test/jobs/3/retry",
    ]
