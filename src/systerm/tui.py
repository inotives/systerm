from __future__ import annotations

import asyncio
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from rich.markup import escape
import uvicorn
import websockets
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Input, Static

from systerm.daemon import create_app
from systerm.daemon_client import DaemonClient, DaemonClientError, load_daemon_client_config


class SystermTui(App[None]):
    CSS = """
    Screen {
        layout: vertical;
        background: #1f1a2a;
        color: #d8d1c7;
    }

    #welcome {
        height: auto;
        padding: 1 1 0 1;
        color: #c8bdd5;
        background: #20242b;
    }

    #transcript-scroll {
        height: 1fr;
        background: #20242b;
    }

    #transcript {
        width: 1fr;
        padding: 1;
        background: #20242b;
    }

    #status {
        height: 1;
        padding: 0 1;
        color: #9b92aa;
        background: #20242b;
    }

    #prompt {
        height: 3;
        border-top: solid #8b748d;
        padding: 0 1;
        background: #2d2739;
        color: #f1ece4;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "refresh", "Refresh"),
        ("j", "next_job", "Next job"),
        ("k", "previous_job", "Previous job"),
        ("s", "next_session", "Next session"),
        ("w", "previous_session", "Previous session"),
        ("l", "load_selected", "Load"),
        ("t", "load_trace", "Trace"),
        ("c", "cancel_selected_job", "Cancel job"),
        ("y", "retry_selected_job", "Retry job"),
        ("a", "approve_selected", "Approve"),
        ("x", "reject_selected", "Reject"),
        ("pageup", "scroll_transcript_up", "Scroll up"),
        ("pagedown", "scroll_transcript_down", "Scroll down"),
    ]

    def __init__(self, auto_start_daemon: bool = True) -> None:
        super().__init__()
        self.title = "Systerm"
        self.auto_start_daemon = auto_start_daemon
        self.client: DaemonClient | None = None
        self.daemon_server: uvicorn.Server | None = None
        self.daemon_task: asyncio.Task[None] | None = None
        self.events_task: asyncio.Task[None] | None = None
        self.last_events: list[dict[str, Any]] = []
        self.active_session_id: int | None = None
        self.snapshot: dict[str, Any] | None = None
        self.selected_job_id: int | None = None
        self.selected_session_id: int | None = None
        self.selected_approval_id: int | None = None

    def compose(self) -> ComposeResult:
        yield Static(render_welcome(None, "connecting to daemon...", []), id="welcome")
        with VerticalScroll(id="transcript-scroll"):
            yield Static("Transcript\n  waiting for a session", id="transcript")
        yield Static("daemon: connecting", id="status")
        yield Input(placeholder="Submit prompt to daemon job queue", id="prompt")

    async def on_mount(self) -> None:
        try:
            self.client = await self.connect_or_start_daemon()
        except Exception as exc:
            self.query_one("#status", Static).update(
                f"daemon unavailable: {exc}\n\nStart it with: uv run systerm daemon"
            )
            return

        await self.refresh_snapshot()
        await self.ensure_active_session()
        self.events_task = asyncio.create_task(self.watch_events())

    async def action_refresh(self) -> None:
        await self.refresh_snapshot()

    async def action_next_job(self) -> None:
        self.selected_job_id = select_relative(self.snapshot_jobs(), self.selected_job_id, 1)
        self.sync_session_to_selected_job()
        self.render_snapshot()

    async def action_previous_job(self) -> None:
        self.selected_job_id = select_relative(self.snapshot_jobs(), self.selected_job_id, -1)
        self.sync_session_to_selected_job()
        self.render_snapshot()

    async def action_next_session(self) -> None:
        self.selected_session_id = select_relative(self.snapshot_sessions(), self.selected_session_id, 1)
        self.render_snapshot()

    async def action_previous_session(self) -> None:
        self.selected_session_id = select_relative(self.snapshot_sessions(), self.selected_session_id, -1)
        self.render_snapshot()

    async def action_load_selected(self) -> None:
        if self.selected_session_id is not None:
            await self.load_session(self.selected_session_id)
            return
        selected_job = find_by_id(self.snapshot_jobs(), self.selected_job_id)
        session_id = selected_job.get("session_id") if selected_job is not None else None
        if isinstance(session_id, int):
            await self.load_session(session_id)

    async def action_load_trace(self) -> None:
        if self.selected_session_id is not None:
            await self.load_trace(self.selected_session_id)

    async def action_cancel_selected_job(self) -> None:
        if self.client is None or self.selected_job_id is None:
            return
        try:
            job = await self.client.cancel_job(self.selected_job_id)
        except Exception as exc:
            self.query_one("#status", Static).update(f"cancel failed: {exc}")
            return
        self.query_one("#status", Static).update(f"canceled job #{job['id']}")
        await self.refresh_snapshot()

    async def action_retry_selected_job(self) -> None:
        if self.client is None or self.selected_job_id is None:
            return
        try:
            job = await self.client.retry_job(self.selected_job_id)
        except Exception as exc:
            self.query_one("#status", Static).update(f"retry failed: {exc}")
            return
        self.selected_job_id = job["id"] if isinstance(job.get("id"), int) else self.selected_job_id
        self.query_one("#status", Static).update(f"queued retry job #{job['id']}")
        await self.refresh_snapshot()

    async def action_approve_selected(self) -> None:
        if self.client is None or self.selected_approval_id is None:
            return
        try:
            approval = await self.client.approve(self.selected_approval_id)
        except Exception as exc:
            self.query_one("#status", Static).update(f"approval failed: {exc}")
            return
        self.query_one("#status", Static).update(f"approved #{approval['id']}")
        await self.refresh_snapshot()

    async def action_reject_selected(self) -> None:
        if self.client is None or self.selected_approval_id is None:
            return
        try:
            approval = await self.client.reject(self.selected_approval_id)
        except Exception as exc:
            self.query_one("#status", Static).update(f"reject failed: {exc}")
            return
        self.query_one("#status", Static).update(f"rejected #{approval['id']}")
        await self.refresh_snapshot()

    async def action_scroll_transcript_up(self) -> None:
        self.query_one("#transcript-scroll", VerticalScroll).scroll_page_up()

    async def action_scroll_transcript_down(self) -> None:
        self.query_one("#transcript-scroll", VerticalScroll).scroll_page_down()

    async def on_unmount(self) -> None:
        if self.events_task is not None:
            self.events_task.cancel()
        if self.daemon_server is not None:
            self.daemon_server.should_exit = True
        if self.daemon_task is not None:
            try:
                await asyncio.wait_for(self.daemon_task, timeout=3)
            except (TimeoutError, asyncio.CancelledError):
                self.daemon_task.cancel()

    async def connect_or_start_daemon(self) -> DaemonClient:
        try:
            client = DaemonClient(load_daemon_client_config())
            await client.health()
            return client
        except Exception:
            if not self.auto_start_daemon:
                raise

        self.query_one("#status", Static).update("starting daemon...")
        await self.start_embedded_daemon()
        client = DaemonClient(load_daemon_client_config())
        await wait_for_daemon(client)
        return client

    async def start_embedded_daemon(self) -> None:
        config = uvicorn.Config(
            create_app(Path.cwd()),
            host="127.0.0.1",
            port=8765,
            log_level="warning",
            lifespan="on",
        )
        self.daemon_server = uvicorn.Server(config)
        self.daemon_task = asyncio.create_task(self.daemon_server.serve())

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        prompt = event.value.strip()
        if not prompt or self.client is None:
            return
        event.input.value = ""
        if await self.handle_command(prompt):
            return
        session_id = await self.ensure_active_session()
        if session_id is None:
            return
        try:
            job = await self.client.create_job_for_session(prompt, session_id)
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

        self.snapshot = snapshot
        self.ensure_selection()
        self.render_snapshot()

    async def handle_command(self, command: str) -> bool:
        if command.strip() != "/session new":
            return False
        self.active_session_id = None
        session_id = await self.ensure_active_session()
        if session_id is not None:
            self.query_one("#status", Static).update(f"started session #{session_id}")
            self.update_transcript(render_transcript({"id": session_id, "messages": []}))
            await self.refresh_snapshot()
        return True

    async def ensure_active_session(self) -> int | None:
        if self.active_session_id is not None:
            return self.active_session_id
        if self.client is None:
            return None
        try:
            session = await self.client.create_session()
        except Exception as exc:
            self.query_one("#status", Static).update(f"session create failed: {exc}")
            return None
        session_id = session.get("id")
        if not isinstance(session_id, int):
            self.query_one("#status", Static).update("session create failed: daemon returned no session id")
            return None
        self.active_session_id = session_id
        self.selected_session_id = session_id
        self.update_transcript(render_transcript({"id": session_id, "messages": []}))
        return session_id

    def render_snapshot(self) -> None:
        if self.snapshot is None:
            return
        snapshot = self.snapshot
        self.query_one("#welcome", Static).update(render_welcome(snapshot, "daemon connected", self.last_events))
        self.query_one("#status", Static).update(render_status(snapshot, self.selected_job_id, self.selected_session_id))
        self.last_events = snapshot["events"][-12:]

    def ensure_selection(self) -> None:
        jobs = self.snapshot_jobs()
        sessions = self.snapshot_sessions()
        approvals = self.snapshot_approvals()
        self.selected_job_id = keep_or_first(jobs, self.selected_job_id)
        self.selected_session_id = keep_or_first(sessions, self.selected_session_id)
        self.selected_approval_id = keep_or_first(approvals, self.selected_approval_id)

    def snapshot_jobs(self) -> list[dict[str, Any]]:
        if self.snapshot is None:
            return []
        return self.snapshot.get("jobs", [])

    def snapshot_sessions(self) -> list[dict[str, Any]]:
        if self.snapshot is None:
            return []
        return self.snapshot.get("sessions", [])

    def snapshot_approvals(self) -> list[dict[str, Any]]:
        if self.snapshot is None:
            return []
        return self.snapshot.get("approvals", [])

    def sync_session_to_selected_job(self) -> None:
        selected_job = find_by_id(self.snapshot_jobs(), self.selected_job_id)
        session_id = selected_job.get("session_id") if selected_job is not None else None
        if isinstance(session_id, int):
            self.selected_session_id = session_id

    async def load_session(self, session_id: int) -> None:
        if self.client is None:
            return
        try:
            session = await self.client.session(session_id)
        except Exception as exc:
            self.query_one("#status", Static).update(f"session load failed: {exc}")
            return
        self.active_session_id = session_id
        self.update_transcript(render_transcript(session))

    async def load_trace(self, session_id: int) -> None:
        if self.client is None:
            return
        try:
            trace = await self.client.session_trace(session_id)
        except Exception as exc:
            self.query_one("#status", Static).update(f"trace load failed: {exc}")
            return
        self.active_session_id = session_id
        self.update_transcript(render_trace(trace))

    def update_transcript(self, content: str) -> None:
        self.query_one("#transcript", Static).update(content)
        self.call_after_refresh(self.query_one("#transcript-scroll", VerticalScroll).scroll_end, animate=False)

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
                        session_id = event.get("session_id")
                        if isinstance(session_id, int) and session_id == self.active_session_id and event.get("type") in {
                            "message.created",
                            "job.completed",
                            "approval.required",
                            "tool_result.created",
                        }:
                            await self.load_session(session_id)
                        if event.get("type", "").startswith(("job.", "approval.", "session.", "message.", "tool_")):
                            await self.refresh_snapshot()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.query_one("#status", Static).update(f"event stream disconnected: {exc}")
                await asyncio.sleep(2)


def render_jobs(jobs: list[dict[str, Any]], selected_id: int | None = None) -> str:
    lines = ["Jobs  j/k move  c cancel  y retry"]
    if not jobs:
        return "Jobs\n  none"
    for job in jobs[:8]:
        marker = ">" if job["id"] == selected_id else " "
        lines.append(f"{marker} #{job['id']} {job['status']} session={job.get('session_id') or '-'}")
    return "\n".join(lines)


def render_welcome(
    snapshot: dict[str, Any] | None,
    status: str,
    events: list[dict[str, Any]],
) -> str:
    _ = events
    cli_version = systerm_version()
    model_name = "loading"
    fallback_names = "-"

    if snapshot is not None:
        model_name = str(snapshot.get("models", {}).get("default_model", "unknown"))
        fallback_names = ", ".join(snapshot.get("models", {}).get("fallback_models", [])) or "-"

    cwd = str(Path.cwd())
    return "\n".join(
        [
            f"systerm cli v{cli_version}",
            f"model: {model_name}    fallback: {fallback_names}",
            f"cwd: {cwd}",
            f"daemon: {status}",
        ]
    )


def render_status(
    snapshot: dict[str, Any],
    selected_job_id: int | None,
    selected_session_id: int | None,
) -> str:
    selected_job = find_by_id(snapshot.get("jobs", []), selected_job_id)
    selected_session = find_by_id(snapshot.get("sessions", []), selected_session_id)
    job_text = f"job #{selected_job['id']} {selected_job['status']}" if selected_job else "job -"
    session_text = f"session #{selected_session['id']}" if selected_session else "session -"
    return f"daemon connected    {job_text}    {session_text}"


def systerm_version() -> str:
    try:
        return version("systerm")
    except PackageNotFoundError:
        return "0.1.0"


def render_runtime_card(snapshot: dict[str, Any]) -> str:
    models = snapshot["models"]
    agent = snapshot["agent"]
    providers = ", ".join(snapshot["providers"].keys()) or "none"
    jobs = summarize_statuses(snapshot.get("jobs", []))
    approvals = summarize_statuses(snapshot.get("approvals", []))
    return "\n".join(
        [
            f"{agent['name']} * {models['default_model']}",
            f"providers: {providers}",
            f"jobs: {jobs}",
            f"approvals: {approvals}",
        ]
    )


def render_sessions(sessions: list[dict[str, Any]], selected_id: int | None = None) -> str:
    lines = ["Sessions  w/s move  l load  t trace"]
    if not sessions:
        return "Sessions\n  none"
    for session in sessions[:8]:
        marker = ">" if session["id"] == selected_id else " "
        lines.append(f"{marker} #{session['id']} {session['message_count']} messages")
    return "\n".join(lines)


def summarize_statuses(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "none"
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    return ", ".join(f"{status} {count}" for status, count in sorted(counts.items()))


def latest_activity(events: list[dict[str, Any]]) -> str:
    if not events:
        return "none"
    event = events[-1]
    return f"#{event.get('id', '-')} {event.get('type', 'event')} job={event.get('job_id') or '-'}"


def combine_columns(left: list[str], right: list[str], left_width: int = 48) -> str:
    rows = []
    height = max(len(left), len(right))
    for index in range(height):
        left_text = left[index] if index < len(left) else ""
        right_text = right[index] if index < len(right) else ""
        rows.append(f"{left_text:<{left_width}} {right_text}")
    return "\n".join(rows)


def render_approvals(approvals: list[dict[str, Any]], selected_id: int | None = None) -> str:
    lines = ["Approvals  a approve  x reject"]
    if not approvals:
        return "Approvals\n  none"
    for approval in approvals[:8]:
        marker = ">" if approval["id"] == selected_id else " "
        lines.append(f"{marker} #{approval['id']} {approval['status']} {approval['risk']} {approval['tool_name']}")
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


def render_details(
    job: dict[str, Any] | None,
    session: dict[str, Any] | None,
    approval: dict[str, Any] | None,
) -> str:
    lines = ["Details"]
    if job is None and session is None and approval is None:
        return "Details\n  none"

    if job is not None:
        lines.extend(
            [
                f"  job: #{job['id']} {job['status']}",
                f"  prompt: {clip(str(job.get('prompt', '')), 80)}",
            ]
        )
        if job.get("error"):
            lines.append(f"  error: {clip(str(job['error']), 80)}")
        elif job.get("result_content"):
            lines.append(f"  result: {clip(str(job['result_content']), 80)}")

    if session is not None:
        lines.append(f"  session: #{session['id']} {session.get('message_count', '-')} messages")

    if approval is not None:
        lines.extend(
            [
                f"  approval: #{approval['id']} {approval['status']} {approval['risk']} {approval['tool_name']}",
                f"  reason: {clip(str(approval.get('reason', '')), 80)}",
                f"  args: {clip(render_arguments(approval.get('arguments_json')), 80)}",
            ]
        )
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
        if role == "user":
            lines.append("")
            lines.extend(render_user_transcript_block(label, content))
        else:
            lines.append(f"\n{label}:")
            lines.append(escape(content) or "  ")
    return "\n".join(lines)


def render_user_transcript_block(label: str, content: str) -> list[str]:
    user_style = "white on #171717"
    block_lines = [f"{label}:", *(content or "  ").splitlines()]
    return [f"[{user_style}] {escape(line)} [/]" for line in block_lines]


def render_trace(trace: dict[str, Any]) -> str:
    lines = [f"Trace session #{trace['id']}"]
    messages = trace.get("messages", [])
    tool_calls = trace.get("tool_calls", [])
    approvals = trace.get("approvals", [])

    lines.append("\nMessages")
    if not messages:
        lines.append("  none")
    for message in messages:
        lines.append(
            f"  #{message['id']} {message['role']} "
            f"{message.get('model_profile') or ''} {clip(str(message.get('content', '')).strip(), 100)}"
        )

    lines.append("\nTool Calls")
    if not tool_calls:
        lines.append("  none")
    for tool_call in tool_calls:
        lines.append(
            f"  #{tool_call['id']} {tool_call['tool_name']} {tool_call['risk']} "
            f"approval={tool_call.get('approval_id') or '-'} {clip(render_arguments(tool_call.get('arguments_json')), 100)}"
        )
        for result in tool_call.get("results", []):
            lines.append(f"    result #{result['id']} {result['status']} {clip(str(result.get('content', '')), 100)}")

    lines.append("\nApprovals")
    if not approvals:
        lines.append("  none")
    for approval in approvals:
        lines.append(f"  #{approval['id']} {approval['status']} {approval['risk']} {approval['tool_name']}")

    return "\n".join(lines)


def latest_completed_session_id(jobs: list[dict[str, Any]]) -> int | None:
    for job in jobs:
        if job.get("status") in {"complete", "tool-use", "approval-required"} and isinstance(job.get("session_id"), int):
            return job["session_id"]
    return None


def render_arguments(arguments_json: object) -> str:
    if not isinstance(arguments_json, str):
        return ""
    try:
        arguments = json.loads(arguments_json)
    except json.JSONDecodeError:
        return arguments_json
    if isinstance(arguments, dict) and isinstance(arguments.get("command"), str):
        return arguments["command"]
    return json.dumps(arguments, sort_keys=True)


def clip(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def keep_or_first(rows: list[dict[str, Any]], selected_id: int | None) -> int | None:
    if not rows:
        return None
    ids = [row["id"] for row in rows]
    if selected_id in ids:
        return selected_id
    return ids[0]


def select_relative(rows: list[dict[str, Any]], selected_id: int | None, offset: int) -> int | None:
    if not rows:
        return None
    ids = [row["id"] for row in rows]
    if selected_id not in ids:
        return ids[0]
    index = ids.index(selected_id)
    return ids[(index + offset) % len(ids)]


def find_by_id(rows: list[dict[str, Any]], selected_id: int | None) -> dict[str, Any] | None:
    for row in rows:
        if row.get("id") == selected_id:
            return row
    return None


async def wait_for_daemon(client: DaemonClient, attempts: int = 50, delay: float = 0.1) -> None:
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            await client.health()
            return
        except Exception as exc:
            last_error = exc
            await asyncio.sleep(delay)
    if last_error is not None:
        raise last_error
    raise RuntimeError("daemon did not start")


def run_tui(auto_start_daemon: bool = True) -> None:
    SystermTui(auto_start_daemon=auto_start_daemon).run()
