import pytest
from textual.containers import VerticalScroll
from textual.widgets import Static

from systerm.tui import (
    SystermTui,
    clip,
    find_by_id,
    keep_or_first,
    latest_completed_session_id,
    render_arguments,
    render_approvals,
    render_details,
    render_events,
    render_jobs,
    render_runtime,
    render_runtime_card,
    render_sessions,
    render_status,
    render_transcript,
    render_trace,
    render_user_transcript_block,
    render_welcome,
    select_relative,
    summarize_statuses,
    wait_for_daemon,
)


def test_tui_render_helpers_show_empty_state() -> None:
    assert render_jobs([]) == "Jobs\n  none"
    assert render_sessions([]) == "Sessions\n  none"
    assert render_approvals([]) == "Approvals\n  none"
    assert render_events([]) == "Events\n  none"


def test_tui_render_runtime() -> None:
    rendered = render_runtime(
        {
            "models": {"default_model": "fast", "fallback_models": ["slow"]},
            "agent": {"name": "systerm"},
            "providers": {"groq": {}},
            "tools": {"shell": {}},
        }
    )

    assert "agent: systerm" in rendered
    assert "model: fast" in rendered
    assert "providers: groq" in rendered
    assert "tools: shell" in rendered


def test_tui_render_welcome() -> None:
    rendered = render_welcome(
        {
            "models": {"default_model": "fast", "fallback_models": []},
            "agent": {"name": "systerm"},
            "providers": {"groq": {}},
            "jobs": [{"id": 1, "status": "running"}],
            "approvals": [{"id": 2, "status": "pending"}],
        },
        "daemon connected",
        [{"id": 9, "type": "job.started", "job_id": 1}],
    )

    assert "systerm cli v0.1.0" in rendered
    assert "model: fast    fallback: -" in rendered
    assert "cwd:" in rendered
    assert "daemon: daemon connected" in rendered


def test_tui_render_status() -> None:
    rendered = render_status(
        {
            "jobs": [{"id": 1, "status": "running"}],
            "sessions": [{"id": 2, "message_count": 1}],
        },
        selected_job_id=1,
        selected_session_id=2,
    )

    assert rendered == "daemon connected    job #1 running    session #2"


def test_tui_render_runtime_card() -> None:
    rendered = render_runtime_card(
        {
            "models": {"default_model": "fast", "fallback_models": []},
            "agent": {"name": "systerm"},
            "providers": {"groq": {}},
            "jobs": [{"status": "running"}],
            "approvals": [{"status": "pending"}],
        }
    )

    assert "systerm * fast" in rendered
    assert "providers: groq" in rendered
    assert "jobs: running 1" in rendered
    assert "approvals: pending 1" in rendered


def test_tui_render_rows() -> None:
    assert "#1 queued" in render_jobs([{"id": 1, "status": "queued", "session_id": None}])
    assert "> #1 queued" in render_jobs([{"id": 1, "status": "queued", "session_id": None}], selected_id=1)
    assert "c cancel" in render_jobs([{"id": 1, "status": "queued", "session_id": None}])
    assert "#2 3 messages" in render_sessions([{"id": 2, "message_count": 3}])
    assert "#3 pending medium shell" in render_approvals(
        [{"id": 3, "status": "pending", "risk": "medium", "tool_name": "shell"}]
    )
    assert "#4 job.created" in render_events([{"id": 4, "type": "job.created", "job_id": 1}])


def test_tui_render_transcript() -> None:
    rendered = render_transcript(
        {
            "id": 7,
            "messages": [
                {"role": "user", "content": "hello", "model_profile": None},
                {"role": "assistant", "content": "hi", "model_profile": "groq"},
            ],
        }
    )

    assert "Transcript session #7" in rendered
    assert "user:" in rendered
    assert "assistant [groq]:" in rendered
    assert "hi" in rendered
    assert "[white on #171717] user:" in rendered


def test_tui_render_user_transcript_block_escapes_markup() -> None:
    rendered = "\n".join(render_user_transcript_block("user", "hello [red]world[/red]"))

    assert "[white on #171717] user:" in rendered
    assert r"\[red]world\[/red]" in rendered


def test_tui_render_details() -> None:
    rendered = render_details(
        {"id": 1, "status": "approval-required", "prompt": "run command", "result_content": "waiting"},
        {"id": 7, "message_count": 2},
        {
            "id": 3,
            "status": "pending",
            "risk": "medium",
            "tool_name": "shell",
            "reason": "needs review",
            "arguments_json": '{"command": "python script.py"}',
        },
    )

    assert "job: #1 approval-required" in rendered
    assert "session: #7 2 messages" in rendered
    assert "approval: #3 pending medium shell" in rendered
    assert "args: python script.py" in rendered


def test_tui_render_trace() -> None:
    rendered = render_trace(
        {
            "id": 7,
            "messages": [{"id": 1, "role": "user", "content": "hello", "model_profile": None}],
            "tool_calls": [
                {
                    "id": 2,
                    "tool_name": "shell",
                    "risk": "low",
                    "approval_id": None,
                    "arguments_json": '{"command": "date"}',
                    "results": [{"id": 3, "status": "complete", "content": "today"}],
                }
            ],
            "approvals": [],
        }
    )

    assert "Trace session #7" in rendered
    assert "#1 user" in rendered
    assert "#2 shell low approval=- date" in rendered
    assert "result #3 complete today" in rendered


def test_latest_completed_session_id() -> None:
    assert latest_completed_session_id(
        [
            {"id": 2, "status": "queued", "session_id": None},
            {"id": 1, "status": "complete", "session_id": 9},
        ]
    ) == 9
    assert latest_completed_session_id([{"id": 2, "status": "failed", "session_id": None}]) is None


def test_tui_selection_helpers() -> None:
    rows = [{"id": 1}, {"id": 2}, {"id": 3}]

    assert keep_or_first(rows, 2) == 2
    assert keep_or_first(rows, 9) == 1
    assert keep_or_first([], 2) is None
    assert select_relative(rows, 1, 1) == 2
    assert select_relative(rows, 1, -1) == 3
    assert select_relative(rows, None, 1) == 1
    assert find_by_id(rows, 2) == {"id": 2}
    assert find_by_id(rows, 9) is None


def test_tui_argument_and_clip_helpers() -> None:
    assert render_arguments('{"command": "date"}') == "date"
    assert render_arguments("{bad") == "{bad"
    assert clip("abcdef", 6) == "abcdef"
    assert clip("abcdef", 5) == "ab..."
    assert summarize_statuses([]) == "none"
    assert summarize_statuses([{"status": "running"}, {"status": "running"}]) == "running 2"


@pytest.mark.asyncio
async def test_wait_for_daemon_retries_until_health_ok() -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.calls = 0

        async def health(self):
            self.calls += 1
            if self.calls < 3:
                raise OSError("not ready")
            return {"status": "ok"}

    client = FakeClient()

    await wait_for_daemon(client, attempts=3, delay=0)

    assert client.calls == 3


@pytest.mark.asyncio
async def test_tui_start_creates_fresh_active_session() -> None:
    class FakeClient:
        config = None
        headers: dict[str, str] = {}

        async def snapshot(self):
            return {
                "sessions": [{"id": 99, "message_count": 2}],
                "jobs": [{"id": 1, "status": "complete", "session_id": 99}],
                "approvals": [],
                "events": [],
                "models": {"default_model": "fast", "fallback_models": []},
                "providers": {},
                "agent": {"name": "systerm"},
                "tools": {},
            }

        async def create_session(self):
            return {"id": 100}

    class TestTui(SystermTui):
        async def connect_or_start_daemon(self):
            return FakeClient()

        async def watch_events(self) -> None:
            return None

    async with TestTui(auto_start_daemon=False).run_test() as pilot:
        assert pilot.app.active_session_id == 100
        assert pilot.app.selected_session_id == 100


@pytest.mark.asyncio
async def test_tui_uses_single_column_scrollable_transcript() -> None:
    async with SystermTui(auto_start_daemon=False).run_test() as pilot:
        assert list(pilot.app.query("#workspace")) == []
        assert list(pilot.app.query("#sidebar")) == []
        assert pilot.app.query_one("#welcome", Static)
        assert pilot.app.query_one("#transcript-scroll", VerticalScroll)
        assert pilot.app.query_one("#transcript", Static)


@pytest.mark.asyncio
async def test_tui_prompt_has_visible_content_region() -> None:
    async with SystermTui(auto_start_daemon=False).run_test(size=(100, 30)) as pilot:
        prompt = pilot.app.query_one("#prompt")
        await pilot.click("#prompt")
        await pilot.press("h", "i")

        assert prompt.value == "hi"
        assert prompt.content_region.height > 0
