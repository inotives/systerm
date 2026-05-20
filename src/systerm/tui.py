from __future__ import annotations

import asyncio
import json
from typing import Any

import websockets
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, Input, Static

from systerm.daemon_client import DaemonClient, DaemonClientError, load_daemon_client_config


class SystermTui(App[None]):
    CSS = """
    Screen {
        layout: vertical;
    }

    #workspace {
        height: 1fr;
    }

    #main {
        width: 2fr;
    }

    #side {
        width: 1fr;
        min-width: 36;
    }

    .panel {
        border: solid $primary;
        padding: 1;
        height: 1fr;
    }

    #prompt {
        dock: bottom;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "refresh", "Refresh"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.client: DaemonClient | None = None
        self.events_task: asyncio.Task[None] | None = None
        self.last_events: list[dict[str, Any]] = []
        self.active_session_id: int | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="workspace"):
            with Vertical(id="main"):
                yield Static("connecting to daemon...", id="status", classes="panel")
                yield Static("Transcript\n  waiting for a completed job", id="transcript", classes="panel")
                yield Static("", id="jobs", classes="panel")
            with Vertical(id="side"):
                yield Static("", id="sessions", classes="panel")
                yield Static("", id="approvals", classes="panel")
                yield Static("", id="runtime", classes="panel")
                yield Static("", id="events", classes="panel")
        yield Input(placeholder="Submit prompt to daemon job queue", id="prompt")
        yield Footer()

    async def on_mount(self) -> None:
        try:
            config = load_daemon_client_config()
            self.client = DaemonClient(config)
            await self.client.health()
        except (DaemonClientError, OSError, Exception) as exc:
            self.query_one("#status", Static).update(
                f"daemon unavailable: {exc}\n\nStart it with: uv run systerm daemon"
            )
            return

        await self.refresh_snapshot()
        self.events_task = asyncio.create_task(self.watch_events())

    async def action_refresh(self) -> None:
        await self.refresh_snapshot()

    async def on_unmount(self) -> None:
        if self.events_task is not None:
            self.events_task.cancel()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        prompt = event.value.strip()
        if not prompt or self.client is None:
            return
        event.input.value = ""
        try:
            job = await self.client.create_job(prompt)
        except Exception as exc:
            self.query_one("#status", Static).update(f"job submit failed: {exc}")
            return
        self.query_one("#status", Static).update(f"queued job {job['id']}")
        await self.refresh_snapshot()

    async def refresh_snapshot(self) -> None:
        if self.client is None:
            return
        try:
            snapshot = await self.client.snapshot()
        except Exception as exc:
            self.query_one("#status", Static).update(f"snapshot failed: {exc}")
            return

        self.query_one("#status", Static).update("daemon connected")
        self.query_one("#jobs", Static).update(render_jobs(snapshot["jobs"]))
        self.query_one("#sessions", Static).update(render_sessions(snapshot["sessions"]))
        self.query_one("#approvals", Static).update(render_approvals(snapshot["approvals"]))
        self.query_one("#runtime", Static).update(render_runtime(snapshot))
        self.last_events = snapshot["events"][-12:]
        self.query_one("#events", Static).update(render_events(self.last_events))
        if self.active_session_id is None:
            latest_session_id = latest_completed_session_id(snapshot["jobs"])
            if latest_session_id is not None:
                await self.load_session(latest_session_id)

    async def load_session(self, session_id: int) -> None:
        if self.client is None:
            return
        try:
            session = await self.client.session(session_id)
        except Exception as exc:
            self.query_one("#status", Static).update(f"session load failed: {exc}")
            return
        self.active_session_id = session_id
        self.query_one("#transcript", Static).update(render_transcript(session))

    async def watch_events(self) -> None:
        if self.client is None:
            return
        while True:
            try:
                async with websockets.connect(
                    self.client.config.websocket_url,
                    additional_headers=self.client.headers,
                ) as websocket:
                    async for message in websocket:
                        event = json.loads(message)
                        self.last_events = (self.last_events + [event])[-12:]
                        self.query_one("#events", Static).update(render_events(self.last_events))
                        if event.get("type") == "job.completed":
                            session_id = event.get("session_id")
                            if isinstance(session_id, int):
                                await self.load_session(session_id)
                        if event.get("type", "").startswith(("job.", "approval.", "session.", "message.")):
                            await self.refresh_snapshot()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.query_one("#status", Static).update(f"event stream disconnected: {exc}")
                await asyncio.sleep(2)


def render_jobs(jobs: list[dict[str, Any]]) -> str:
    lines = ["Jobs"]
    if not jobs:
        return "Jobs\n  none"
    for job in jobs[:8]:
        lines.append(f"  #{job['id']} {job['status']} session={job.get('session_id') or '-'}")
    return "\n".join(lines)


def render_sessions(sessions: list[dict[str, Any]]) -> str:
    lines = ["Sessions"]
    if not sessions:
        return "Sessions\n  none"
    for session in sessions[:8]:
        lines.append(f"  #{session['id']} {session['message_count']} messages")
    return "\n".join(lines)


def render_approvals(approvals: list[dict[str, Any]]) -> str:
    lines = ["Approvals"]
    if not approvals:
        return "Approvals\n  none"
    for approval in approvals[:8]:
        lines.append(f"  #{approval['id']} {approval['status']} {approval['risk']} {approval['tool_name']}")
    return "\n".join(lines)


def render_runtime(snapshot: dict[str, Any]) -> str:
    models = snapshot["models"]
    agent = snapshot["agent"]
    providers = ", ".join(snapshot["providers"].keys()) or "none"
    tools = ", ".join(snapshot["tools"].keys()) or "none"
    return "\n".join(
        [
            "Runtime",
            f"  agent: {agent['name']}",
            f"  model: {models['default_model']}",
            f"  fallbacks: {', '.join(models['fallback_models']) or '-'}",
            f"  providers: {providers}",
            f"  tools: {tools}",
        ]
    )


def render_events(events: list[dict[str, Any]]) -> str:
    lines = ["Events"]
    if not events:
        return "Events\n  none"
    for event in events:
        lines.append(f"  #{event['id']} {event['type']} job={event.get('job_id') or '-'}")
    return "\n".join(lines)


def render_transcript(session: dict[str, Any]) -> str:
    lines = [f"Transcript session #{session['id']}"]
    messages = session.get("messages", [])
    if not messages:
        lines.append("  no messages")
        return "\n".join(lines)

    for message in messages:
        role = message.get("role", "unknown")
        model = message.get("model_profile")
        label = f"{role}"
        if model:
            label += f" [{model}]"
        content = str(message.get("content", "")).strip()
        lines.append(f"\n{label}:")
        lines.append(content or "  ")
    return "\n".join(lines)


def latest_completed_session_id(jobs: list[dict[str, Any]]) -> int | None:
    for job in jobs:
        if job.get("status") in {"complete", "tool-use", "approval-required"} and isinstance(job.get("session_id"), int):
            return job["session_id"]
    return None


def run_tui() -> None:
    SystermTui().run()
