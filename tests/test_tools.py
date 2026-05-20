from pathlib import Path

import pytest

from systerm.storage import SessionStore
from systerm.tools import ToolRunner, classify_shell_risk, load_tool_definition


def test_classify_shell_risk() -> None:
    assert classify_shell_risk("echo hello") == "low"
    assert classify_shell_risk("python script.py") == "medium"
    assert classify_shell_risk("rm -rf build") == "high"
    assert classify_shell_risk("ls | wc -l") == "medium"


@pytest.mark.asyncio
async def test_low_risk_shell_command_runs(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "systerm.db")
    await store.init()

    result = await ToolRunner(store).run_shell("echo hello")

    assert result.status == "complete"
    assert result.content == "hello\n"
    assert await store.list_approvals() == []


@pytest.mark.asyncio
async def test_medium_risk_shell_command_creates_approval(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "systerm.db")
    await store.init()

    result = await ToolRunner(store).run_shell("python script.py")

    assert result.status == "approval-required"
    assert result.approval_id is not None
    approvals = await store.list_approvals()
    assert len(approvals) == 1
    assert approvals[0].tool_name == "shell"
    assert approvals[0].status == "pending"
    assert approvals[0].risk == "medium"


def test_load_tool_definition(tmp_path: Path) -> None:
    path = tmp_path / "tool.toml"
    path.write_text(
        """
name = "shell"
description = "Run shell commands."
risk = "medium"

[parameters]
type = "object"
required = ["command"]

[parameters.properties.command]
type = "string"
""",
        encoding="utf-8",
    )

    tool = load_tool_definition(path)

    assert tool.name == "shell"
    assert tool.risk == "medium"
    assert tool.parameters["required"] == ["command"]
