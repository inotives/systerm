from pathlib import Path
from shutil import copyfile

import httpx
import pytest
from fastapi.testclient import TestClient

from systerm.daemon import create_app, load_or_create_token, run_job
from systerm.storage import SessionStore, default_db_path


@pytest.fixture(autouse=True)
def isolate_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))


def copy_project_config(source_root: Path, target_root: Path) -> None:
    (target_root / ".systerm").mkdir()
    copyfile(source_root / ".systerm" / "config.toml", target_root / ".systerm" / "config.toml")
    copyfile(source_root / ".systerm" / "model.toml", target_root / ".systerm" / "model.toml")
    copyfile(source_root / "AGENTS.md", target_root / "AGENTS.md")
    tool_dir = target_root / ".agents" / "tools" / "shell"
    tool_dir.mkdir(parents=True)
    copyfile(source_root / ".agents" / "tools" / "shell" / "tool.toml", tool_dir / "tool.toml")


@pytest.mark.asyncio
async def test_daemon_rejects_missing_auth(tmp_path: Path) -> None:
    app = create_app(tmp_path, token="test-token")

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/sessions")

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_daemon_lists_sessions(tmp_path: Path) -> None:
    store = SessionStore(default_db_path(tmp_path))
    await store.init()
    session = await store.create_session(metadata_json='{"source": "test"}')
    await store.add_message(session.id, "user", "hello")
    app = create_app(tmp_path, token="test-token")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
        headers={"Authorization": "Bearer test-token"},
    ) as client:
        response = await client.get("/sessions")

    assert response.status_code == 200
    assert response.json() == [
        {
            "id": session.id,
            "created_at": session.created_at,
            "metadata_json": '{"source": "test"}',
            "message_count": 1,
        }
    ]


@pytest.mark.asyncio
async def test_daemon_approves_pending_approval(tmp_path: Path) -> None:
    store = SessionStore(default_db_path(tmp_path))
    await store.init()
    approval = await store.create_approval("shell", '{"command": "python script.py"}', "medium", "needs approval")
    app = create_app(tmp_path, token="test-token")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
        headers={"Authorization": "Bearer test-token"},
    ) as client:
        response = await client.post(f"/approvals/{approval.id}/approve")

    assert response.status_code == 200
    assert response.json()["status"] == "approved"


@pytest.mark.asyncio
async def test_daemon_lists_jobs(tmp_path: Path) -> None:
    store = SessionStore(default_db_path(tmp_path))
    await store.init()
    job = await store.create_job("hello")
    app = create_app(tmp_path, token="test-token")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
        headers={"Authorization": "Bearer test-token"},
    ) as client:
        response = await client.get("/jobs")

    assert response.status_code == 200
    assert response.json()[0]["id"] == job.id
    assert response.json()[0]["status"] == "queued"


@pytest.mark.asyncio
async def test_daemon_gets_job_detail(tmp_path: Path) -> None:
    store = SessionStore(default_db_path(tmp_path))
    await store.init()
    job = await store.create_job("hello")
    app = create_app(tmp_path, token="test-token")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
        headers={"Authorization": "Bearer test-token"},
    ) as client:
        response = await client.get(f"/jobs/{job.id}")

    assert response.status_code == 200
    assert response.json()["id"] == job.id


@pytest.mark.asyncio
async def test_daemon_lists_events(tmp_path: Path) -> None:
    store = SessionStore(default_db_path(tmp_path))
    await store.init()
    job = await store.create_job("hello")
    event = await store.create_event("job.created", '{"job_id": 1}', job_id=job.id)
    app = create_app(tmp_path, token="test-token")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
        headers={"Authorization": "Bearer test-token"},
    ) as client:
        response = await client.get("/events")

    assert response.status_code == 200
    assert response.json()[0]["id"] == event.id
    assert response.json()[0]["type"] == "job.created"


@pytest.mark.asyncio
async def test_daemon_exposes_runtime_resources(tmp_path: Path) -> None:
    copy_project_config(Path.cwd(), tmp_path)
    app = create_app(tmp_path, token="test-token")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
        headers={"Authorization": "Bearer test-token"},
    ) as client:
        models = await client.get("/models")
        providers = await client.get("/providers")
        agent = await client.get("/agents/current")
        tools = await client.get("/tools")
        schedules = await client.get("/schedules")
        connectors = await client.get("/connectors")

    assert models.status_code == 200
    assert "nvidia-minimax-2.7" in models.json()["model_profiles"]
    assert providers.status_code == 200
    assert "nvidia" in providers.json()
    assert agent.json()["name"] == "systerm"
    assert "shell" in tools.json()
    assert schedules.json() == []
    assert connectors.json() == []


@pytest.mark.asyncio
async def test_daemon_create_job_returns_queued_and_records_events(tmp_path: Path) -> None:
    app = create_app(tmp_path, token="test-token")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
        headers={"Authorization": "Bearer test-token"},
    ) as client:
        response = await client.post("/jobs", json={"prompt": "hello"})
        events = await client.get("/events")

    assert response.status_code == 200
    assert response.json()["status"] == "queued"
    assert events.status_code == 200
    assert events.json()[0]["type"] == "job.created"


def test_daemon_websocket_streams_events(tmp_path: Path) -> None:
    app = create_app(tmp_path, token="test-token")

    with TestClient(app) as client:
        with client.websocket_connect("/events", headers={"Authorization": "Bearer test-token"}) as websocket:
            response = client.post(
                "/jobs",
                json={"prompt": "hello"},
                headers={"Authorization": "Bearer test-token"},
            )
            event = websocket.receive_json()

    assert response.status_code == 200
    assert event["type"] == "job.created"


def test_load_or_create_token_creates_restrictive_token_file(tmp_path: Path) -> None:
    token_path = tmp_path / "token"

    token = load_or_create_token(token_path)

    assert token
    assert token_path.read_text(encoding="utf-8") == token
    assert token_path.stat().st_mode & 0o777 == 0o600


@pytest.mark.asyncio
async def test_run_job_records_failure_event(tmp_path: Path) -> None:
    store = SessionStore(default_db_path(tmp_path))
    await store.init()
    job = await store.create_job("hello")
    subscribers = set()

    await run_job(tmp_path, store, subscribers, job.id, job.prompt)

    completed = await store.get_job(job.id)
    events = await store.list_events()
    assert completed is not None
    assert completed.status == "failed"
    assert events[-1].type == "job.failed"
