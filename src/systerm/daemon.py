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
from systerm.tools import ToolRunner, load_tool_registry


TOKEN_ENV = "SYSTERM_DAEMON_TOKEN"


class JobRequest(BaseModel):
    prompt: str
    session_id: int | None = None


class ApprovalResolution(BaseModel):
    status: str


def create_app(project_root: Path, token: str | None = None) -> FastAPI:
    auth_token = token or load_or_create_token()
    store = SessionStore(default_db_path(project_root))
    subscribers: set[asyncio.Queue[dict[str, object]]] = set()
    job_tasks: dict[int, asyncio.Task[None]] = {}

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

    @app.post("/sessions", dependencies=[auth])
    async def create_session() -> dict[str, object]:
        await store.init()
        session = await store.create_session(metadata_json=json.dumps({"source": "tui"}))
        await publish_event(
            store,
            subscribers,
            "session.created",
            {"session_id": session.id},
            session_id=session.id,
        )
        return session.__dict__

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
        approval = await store.resolve_approval(approval_id, "approved")
        await publish_event(store, subscribers, "approval.approved", approval.__dict__)
        asyncio.create_task(resume_approved_tool(store, subscribers, approval_id))
        return approval.__dict__

    @app.post("/approvals/{approval_id}/reject", dependencies=[auth])
    async def reject(approval_id: int) -> dict[str, object]:
        approval = await store.resolve_approval(approval_id, "rejected")
        await publish_event(store, subscribers, "approval.rejected", approval.__dict__)
        tool_call = await store.get_tool_call_by_approval(approval_id)
        if tool_call is not None and tool_call.session_id is not None:
            await store.add_message(tool_call.session_id, "assistant", "tool call rejected by operator")
            job = await store.get_latest_job_for_session(tool_call.session_id)
            if job is not None and job.status == "approval-required":
                completed = await store.complete_job(
                    job.id,
                    "rejected",
                    tool_call.session_id,
                    result_content="tool call rejected by operator",
                )
                await publish_event(
                    store,
                    subscribers,
                    "job.completed",
                    completed.__dict__,
                    job_id=job.id,
                    session_id=tool_call.session_id,
                )
        return approval.__dict__

    @app.post("/jobs", dependencies=[auth])
    async def create_job(request: JobRequest) -> dict[str, object]:
        await store.init()
        job = await store.create_job(request.prompt, session_id=request.session_id)
        await publish_event(
            store,
            subscribers,
            "job.created",
            {"job_id": job.id, "status": job.status, "session_id": job.session_id},
            job_id=job.id,
            session_id=job.session_id,
        )
        job_tasks[job.id] = asyncio.create_task(
            run_job(project_root, store, subscribers, job.id, request.prompt, session_id=request.session_id)
        )
        return job.__dict__

    @app.post("/jobs/{job_id}/cancel", dependencies=[auth])
    async def cancel_job(job_id: int) -> dict[str, object]:
        try:
            job = await store.cancel_job(job_id)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        task = job_tasks.pop(job_id, None)
        if task is not None and not task.done():
            task.cancel()
        await publish_event(
            store,
            subscribers,
            "job.canceled",
            job.__dict__,
            job_id=job.id,
            session_id=job.session_id,
        )
        return job.__dict__

    @app.post("/jobs/{job_id}/retry", dependencies=[auth])
    async def retry_job(job_id: int) -> dict[str, object]:
        original = await store.get_job(job_id)
        if original is None:
            raise HTTPException(status_code=404, detail="job not found")
        job = await store.create_job(
            original.prompt,
            session_id=original.session_id,
            metadata_json=json.dumps({"retry_of": job_id}),
        )
        await publish_event(
            store,
            subscribers,
            "job.created",
            {"job_id": job.id, "status": job.status, "retry_of": job_id},
            job_id=job.id,
        )
        job_tasks[job.id] = asyncio.create_task(
            run_job(project_root, store, subscribers, job.id, job.prompt, session_id=original.session_id)
        )
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
    session_id: int | None = None,
) -> None:
    job = await store.get_job(job_id)
    if job is None or job.status == "canceled":
        return
    if session_id is None:
        session_id = job.session_id
    running = await store.mark_job_running(job_id)
    await publish_event(store, subscribers, "job.started", running.__dict__, job_id=job_id, session_id=session_id)
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
            session_id=session_id,
        )
    except asyncio.CancelledError:
        job = await store.get_job(job_id)
        if job is not None and job.status != "canceled":
            completed = await store.cancel_job(job_id)
            await publish_event(
                store,
                subscribers,
                "job.canceled",
                completed.__dict__,
                job_id=job_id,
                session_id=completed.session_id,
            )
        return
    except Exception as exc:
        job = await store.get_job(job_id)
        if job is not None and job.status == "canceled":
            return
        completed = await store.complete_job(job_id, "failed", session_id, error=str(exc))
        await publish_event(
            store,
            subscribers,
            "job.failed",
            completed.__dict__,
            job_id=job_id,
            session_id=session_id,
        )
        return

    job = await store.get_job(job_id)
    if job is not None and job.status == "canceled":
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


async def resume_approved_tool(
    store: SessionStore,
    subscribers: set[asyncio.Queue[dict[str, object]]],
    approval_id: int,
) -> None:
    tool_call = await store.get_tool_call_by_approval(approval_id)
    if tool_call is None or tool_call.session_id is None:
        return
    if tool_call.tool_name != "shell":
        return

    arguments = json.loads(tool_call.arguments_json)
    command = arguments.get("command")
    if not isinstance(command, str):
        return

    await publish_event(
        store,
        subscribers,
        "tool_call.resumed",
        {"tool_call_id": tool_call.id, "approval_id": approval_id},
        session_id=tool_call.session_id,
    )
    result = await ToolRunner(store).run_approved_shell(command, tool_call.id)
    await store.add_message(
        tool_call.session_id,
        "tool",
        result.content,
        metadata_json=json.dumps({"approved_tool_call_id": tool_call.id}),
    )
    await publish_event(
        store,
        subscribers,
        "tool_result.created",
        {"tool_call_id": tool_call.id, "status": result.status, "approval_id": approval_id},
        session_id=tool_call.session_id,
    )
    await publish_event(
        store,
        subscribers,
        "message.created",
        {"role": "tool", "tool_call_id": tool_call.id},
        session_id=tool_call.session_id,
    )

    job = await store.get_latest_job_for_session(tool_call.session_id)
    if job is not None and job.status == "approval-required":
        completed = await store.complete_job(
            job.id,
            "tool-use" if result.status == "complete" else "failed",
            tool_call.session_id,
            result_content=result.content,
            error=None if result.status == "complete" else result.content,
        )
        await publish_event(
            store,
            subscribers,
            "job.completed",
            completed.__dict__,
            job_id=job.id,
            session_id=tool_call.session_id,
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
