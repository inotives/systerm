from __future__ import annotations

from typer.testing import CliRunner

import systerm.cli as cli


class FakeDaemonClient:
    async def create_job(self, prompt: str) -> dict[str, object]:
        return {"id": 9, "status": "queued", "prompt": prompt}

    async def jobs(self) -> list[dict[str, object]]:
        return [{"id": 9, "status": "failed", "session_id": 2, "metadata_json": "{}", "prompt": "hello"}]

    async def job(self, job_id: int) -> dict[str, object]:
        return {"id": job_id, "status": "failed", "session_id": 2, "metadata_json": "{}", "prompt": "hello"}

    async def cancel_job(self, job_id: int) -> dict[str, object]:
        return {"id": job_id, "status": "canceled"}

    async def retry_job(self, job_id: int) -> dict[str, object]:
        return {"id": job_id + 1, "status": "queued"}

    async def sessions(self) -> list[dict[str, object]]:
        return [{"id": 2, "created_at": "now", "message_count": 3, "metadata_json": "{}"}]

    async def session(self, session_id: int) -> dict[str, object]:
        return {
            "id": session_id,
            "messages": [
                {
                    "id": 4,
                    "role": "assistant",
                    "content": "hello",
                    "model_profile": "groq",
                    "metadata_json": "{}",
                }
            ],
        }

    async def session_trace(self, session_id: int) -> dict[str, object]:
        return {
            "id": session_id,
            "messages": [],
            "tool_calls": [
                {
                    "id": 5,
                    "tool_name": "shell",
                    "risk": "medium",
                    "approval_id": 6,
                    "metadata_json": "{}",
                    "arguments_json": '{"command": "date"}',
                    "results": [],
                }
            ],
            "approvals": [],
        }

    async def approvals(self, status: str = "pending") -> list[dict[str, object]]:
        return [
            {
                "id": 6,
                "status": status,
                "risk": "medium",
                "tool_name": "shell",
                "arguments_json": '{"command": "date"}',
            }
        ]

    async def approve(self, approval_id: int) -> dict[str, object]:
        return {"id": approval_id, "status": "approved", "tool_name": "shell"}

    async def reject(self, approval_id: int) -> dict[str, object]:
        return {"id": approval_id, "status": "rejected", "tool_name": "shell"}


def test_cli_chat_can_submit_daemon_job(monkeypatch) -> None:
    monkeypatch.setattr(cli, "_load_daemon_client", FakeDaemonClient)

    result = CliRunner().invoke(cli.app, ["chat", "hello", "--daemon"])

    assert result.exit_code == 0
    assert "queued job 9\tqueued" in result.output


def test_cli_default_tui_can_disable_auto_daemon(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(cli, "run_tui", lambda auto_start_daemon=True: calls.append(auto_start_daemon))

    result = CliRunner().invoke(cli.app, ["--no-auto-daemon"])

    assert result.exit_code == 0
    assert calls == [False]


def test_cli_tui_command_can_disable_auto_daemon(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(cli, "run_tui", lambda auto_start_daemon=True: calls.append(auto_start_daemon))

    result = CliRunner().invoke(cli.app, ["tui", "--no-auto-daemon"])

    assert result.exit_code == 0
    assert calls == [False]


def test_cli_sessions_can_read_from_daemon(monkeypatch) -> None:
    monkeypatch.setattr(cli, "_load_daemon_client", FakeDaemonClient)

    list_result = CliRunner().invoke(cli.app, ["sessions", "list", "--daemon"])
    show_result = CliRunner().invoke(cli.app, ["sessions", "show", "2", "--daemon"])
    trace_result = CliRunner().invoke(cli.app, ["sessions", "trace", "2", "--daemon"])

    assert list_result.exit_code == 0
    assert "2\tnow\t3 messages\t{}" in list_result.output
    assert show_result.exit_code == 0
    assert "4\tassistant\tgroq\t{}\thello" in show_result.output
    assert trace_result.exit_code == 0
    assert "5\tshell\tmedium\tapproval=6" in trace_result.output


def test_cli_jobs_can_use_daemon(monkeypatch) -> None:
    monkeypatch.setattr(cli, "_load_daemon_client", FakeDaemonClient)

    list_result = CliRunner().invoke(cli.app, ["jobs", "list"])
    show_result = CliRunner().invoke(cli.app, ["jobs", "show", "9"])
    cancel_result = CliRunner().invoke(cli.app, ["jobs", "cancel", "9"])
    retry_result = CliRunner().invoke(cli.app, ["jobs", "retry", "9"])

    assert list_result.exit_code == 0
    assert "9\tfailed\tsession=2\t{}\thello" in list_result.output
    assert show_result.exit_code == 0
    assert "9\tfailed\tsession=2\t{}\thello" in show_result.output
    assert cancel_result.exit_code == 0
    assert "9\tcanceled" in cancel_result.output
    assert retry_result.exit_code == 0
    assert "queued job 10\tqueued" in retry_result.output


def test_cli_approvals_can_use_daemon(monkeypatch) -> None:
    monkeypatch.setattr(cli, "_load_daemon_client", FakeDaemonClient)

    list_result = CliRunner().invoke(cli.app, ["approvals", "list", "--daemon", "--status", "all"])
    approve_result = CliRunner().invoke(cli.app, ["approvals", "approve", "6", "--daemon"])
    reject_result = CliRunner().invoke(cli.app, ["approvals", "reject", "7", "--daemon"])

    assert list_result.exit_code == 0
    assert "6\tall\tmedium\tshell" in list_result.output
    assert approve_result.exit_code == 0
    assert "6\tapproved\tshell" in approve_result.output
    assert reject_result.exit_code == 0
    assert "7\trejected\tshell" in reject_result.output
