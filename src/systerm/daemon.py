from __future__ import annotations

import asyncio
import json
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from systerm.agents import load_agent_profile
from systerm.agent_loop import AgentLoop
from systerm.config import load_config, validate_model_profile
from systerm.providers import OpenAICompatibleClient, ProviderError
from systerm.storage import SessionStore, default_db_path
from systerm.tools import load_tool_registry


TOKEN_ENV = "SYSTERM_DAEMON_TOKEN"


class JobRequest(BaseModel):
    prompt: str


class ApprovalResolution(BaseModel):
    status: str


def create_app(project_root: Path, token: str | None = None) -> FastAPI:
    auth_token = token or load_or_create_token()
    store = SessionStore(default_db_path(project_root))
    subscribers: set[asyncio.Queue[dict[str, object]]] = set()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await store.init()
        yield

    app = FastAPI(title="Systerm Daemon", lifespan=lifespan)

    async def require_auth(authorization: Annotated[str | None, Header()] = None) -> None:
        if authorization != f"Bearer {auth_token}":
            raise HTTPException(status_code=401, detail="invalid token")

    auth = Depends(require_auth)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/sessions", dependencies=[auth])
    async def sessions() -> list[dict[str, object]]:
        return await store.list_sessions()

    @app.get("/jobs", dependencies=[auth])
    async def jobs() -> list[dict[str, object]]:
        return [job.__dict__ for job in await store.list_jobs()]

    @app.get("/jobs/{job_id}", dependencies=[auth])
    async def job_detail(job_id: int) -> dict[str, object]:
        job = await store.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return job.__dict__

    @app.get("/events", dependencies=[auth])
    async def events(after_id: int = 0, limit: int = 100) -> list[dict[str, object]]:
        return [event.__dict__ for event in await store.list_events(after_id=after_id, limit=limit)]

    @app.get("/models", dependencies=[auth])
    async def models() -> dict[str, object]:
        config = load_config(project_root)
        return {
            "default_model": config.models.default_model,
            "fallback_models": config.models.fallback_models,
            "model_profiles": {name: profile.model_dump() for name, profile in config.model_profiles.items()},
        }

    @app.get("/providers", dependencies=[auth])
    async def providers() -> dict[str, object]:
        config = load_config(project_root)
        return {name: provider.model_dump() for name, provider in config.providers.items()}

    @app.get("/agents/current", dependencies=[auth])
    async def current_agent() -> dict[str, object]:
        profile = load_agent_profile(project_root / "AGENTS.md")
        return {
            "name": profile.name,
            "model": profile.model,
            "skills": list(profile.skills),
            "tools": list(profile.tools),
        }

    @app.get("/tools", dependencies=[auth])
    async def tools() -> dict[str, object]:
        profile = load_agent_profile(project_root / "AGENTS.md")
        registry = load_tool_registry(project_root, profile.tools)
        return {name: tool.__dict__ for name, tool in registry.items()}

    @app.get("/skills", dependencies=[auth])
    async def skills() -> dict[str, object]:
        profile = load_agent_profile(project_root / "AGENTS.md")
        return {"skills": list(profile.skills)}

    @app.get("/schedules", dependencies=[auth])
    async def schedules() -> list[object]:
        return []

    @app.get("/connectors", dependencies=[auth])
    async def connectors() -> list[object]:
        return []

    @app.get("/sessions/{session_id}", dependencies=[auth])
    async def session_messages(session_id: int) -> dict[str, object]:
        messages = await store.list_message_records(session_id)
        return {
            "id": session_id,
            "messages": [message.__dict__ for message in messages],
        }

    @app.get("/sessions/{session_id}/trace", dependencies=[auth])
    async def session_trace(session_id: int) -> dict[str, object]:
        tool_calls = await store.list_tool_calls(session_id)
        return {
            "id": session_id,
            "messages": [message.__dict__ for message in await store.list_message_records(session_id)],
            "tool_calls": [
                {
                    **tool_call.__dict__,
                    "results": [result.__dict__ for result in await store.list_tool_results(tool_call.id)],
                }
                for tool_call in tool_calls
            ],
            "approvals": [approval.__dict__ for approval in await store.list_session_approvals(session_id)],
        }

    @app.get("/approvals", dependencies=[auth])
    async def approvals(status: str = "pending") -> list[dict[str, object]]:
        status_filter = None if status == "all" else status
        return [approval.__dict__ for approval in await store.list_approvals(status_filter)]

    @app.post("/approvals/{approval_id}/approve", dependencies=[auth])
    async def approve(approval_id: int) -> dict[str, object]:
        return (await store.resolve_approval(approval_id, "approved")).__dict__

    @app.post("/approvals/{approval_id}/reject", dependencies=[auth])
    async def reject(approval_id: int) -> dict[str, object]:
        return (await store.resolve_approval(approval_id, "rejected")).__dict__

    @app.post("/jobs", dependencies=[auth])
    async def create_job(request: JobRequest) -> dict[str, object]:
        await store.init()
        job = await store.create_job(request.prompt)
        await publish_event(
            store,
            subscribers,
            "job.created",
            {"job_id": job.id, "status": job.status},
            job_id=job.id,
        )
        asyncio.create_task(run_job(project_root, store, subscribers, job.id, request.prompt))
        return job.__dict__

    @app.websocket("/events")
    async def event_stream(websocket: WebSocket) -> None:
        authorization = websocket.headers.get("authorization")
        if authorization != f"Bearer {auth_token}":
            await websocket.close(code=1008)
            return

        await websocket.accept()
        queue: asyncio.Queue[dict[str, object]] = asyncio.Queue()
        subscribers.add(queue)
        try:
            while True:
                await websocket.send_json(await queue.get())
        except WebSocketDisconnect:
            pass
        finally:
            subscribers.discard(queue)

    return app


async def run_job(
    project_root: Path,
    store: SessionStore,
    subscribers: set[asyncio.Queue[dict[str, object]]],
    job_id: int,
    prompt: str,
) -> None:
    await publish_event(store, subscribers, "job.started", {"job_id": job_id}, job_id=job_id)
    try:
        config = load_config(project_root)
        profile = load_agent_profile(project_root / "AGENTS.md")
        validate_model_profile(config, profile.model)
        tools = load_tool_registry(project_root, profile.tools)
        async def publish_runtime_event(
            event_type: str,
            payload: dict[str, object],
            session_id: int | None,
        ) -> None:
            await publish_event(
                store,
                subscribers,
                event_type,
                {**payload, "job_id": job_id},
                job_id=job_id,
                session_id=session_id,
            )

        result = await AgentLoop(OpenAICompatibleClient(config), store, tools, publish_runtime_event).run(
            prompt,
            requested_model=profile.model,
        )
    except Exception as exc:
        completed = await store.complete_job(job_id, "failed", None, error=str(exc))
        await publish_event(
            store,
            subscribers,
            "job.failed",
            completed.__dict__,
            job_id=job_id,
        )
        return

    completed = await store.complete_job(
        job_id,
        result.stop_reason,
        result.session_id,
        result_content=result.content,
    )
    await publish_event(
        store,
        subscribers,
        "job.completed",
        {**completed.__dict__, "model_profile": result.model_profile},
        job_id=job_id,
        session_id=result.session_id,
    )


async def publish_event(
    store: SessionStore,
    subscribers: set[asyncio.Queue[dict[str, object]]],
    event_type: str,
    payload: dict[str, object],
    job_id: int | None = None,
    session_id: int | None = None,
) -> dict[str, object]:
    event = await store.create_event(
        event_type=event_type,
        job_id=job_id,
        session_id=session_id,
        payload_json=json.dumps(payload, sort_keys=True),
    )
    event_data = event.__dict__
    for queue in list(subscribers):
        queue.put_nowait(event_data)
    return event_data


def load_or_create_token(path: Path | None = None) -> str:
    import os

    env_token = os.getenv(TOKEN_ENV)
    if env_token:
        return env_token

    token_path = path or Path.home() / ".config" / "systerm" / "token"
    if token_path.exists():
        return token_path.read_text(encoding="utf-8").strip()

    token_path.parent.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(32)
    token_path.write_text(token, encoding="utf-8")
    token_path.chmod(0o600)
    return token
