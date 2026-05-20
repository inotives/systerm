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
