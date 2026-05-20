from __future__ import annotations

import asyncio
import json
import shlex
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from systerm.storage import SessionStore


RiskLevel = Literal["low", "medium", "high"]
ToolStatus = Literal["complete", "approval-required", "error"]


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]
    risk: RiskLevel


class ToolRegistryError(ValueError):
    pass


@dataclass(frozen=True)
class ToolResult:
    status: ToolStatus
    content: str
    approval_id: int | None = None
    tool_call_id: int | None = None


SHELL_TOOL = ToolDefinition(
    name="shell",
    description="Run a local shell command.",
    parameters={
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    },
    risk="medium",
)


def load_tool_registry(project_root: Path, tool_names: tuple[str, ...]) -> dict[str, ToolDefinition]:
    registry: dict[str, ToolDefinition] = {}
    for tool_name in tool_names:
        path = project_root / ".agents" / "tools" / tool_name / "tool.toml"
        registry[tool_name] = load_tool_definition(path)
    return registry


def load_tool_definition(path: Path) -> ToolDefinition:
    if not path.exists():
        raise ToolRegistryError(f"Missing tool definition: {path}")

    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ToolRegistryError(f"Invalid tool definition TOML: {path}: {exc}") from exc

    name = raw.get("name")
    description = raw.get("description")
    risk = raw.get("risk")
    parameters = raw.get("parameters")
    if not isinstance(name, str) or not name:
        raise ToolRegistryError(f"Tool definition {path} must define name")
    if not isinstance(description, str) or not description:
        raise ToolRegistryError(f"Tool definition {path} must define description")
    if risk not in {"low", "medium", "high"}:
        raise ToolRegistryError(f"Tool definition {path} must define risk as low, medium, or high")
    if not isinstance(parameters, dict):
        raise ToolRegistryError(f"Tool definition {path} must define parameters")

    return ToolDefinition(name=name, description=description, parameters=parameters, risk=risk)


def openai_tool_schema(tool: ToolDefinition) -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


LOW_RISK_COMMANDS = {"date", "echo", "ls", "pwd", "whoami"}
HIGH_RISK_TOKENS = {"rm", "rmdir", "sudo", "su", "chmod", "chown", "mkfs", "dd", "reboot", "shutdown"}


def classify_shell_risk(command: str) -> RiskLevel:
    parts = shlex.split(command)
    if not parts:
        return "low"

    first = parts[0]
    if first in HIGH_RISK_TOKENS:
        return "high"
    if any(token in command for token in ["|", ">", "<", "&&", "||", ";", "$("]):
        return "medium"
    if first in LOW_RISK_COMMANDS:
        return "low"
    return "medium"


class ToolRunner:
    def __init__(self, store: SessionStore) -> None:
        self.store = store

    async def run_shell(self, command: str, session_id: int | None = None) -> ToolResult:
        risk = classify_shell_risk(command)
        arguments_json = json.dumps({"command": command}, sort_keys=True)
        if risk != "low":
            approval = await self.store.create_approval(
                tool_name=SHELL_TOOL.name,
                arguments_json=arguments_json,
                risk=risk,
                reason=f"Shell command classified as {risk} risk",
                metadata_json=json.dumps({"source": "shell_risk_policy"}),
            )
            tool_call = await self.store.create_tool_call(
                tool_name=SHELL_TOOL.name,
                arguments_json=arguments_json,
                risk=risk,
                session_id=session_id,
                approval_id=approval.id,
                metadata_json=json.dumps({"approval_required": True}),
            )
            return ToolResult(
                status="approval-required",
                content=f"approval required for shell command: {command}",
                approval_id=approval.id,
                tool_call_id=tool_call.id,
            )

        tool_call = await self.store.create_tool_call(
            tool_name=SHELL_TOOL.name,
            arguments_json=arguments_json,
            risk=risk,
            session_id=session_id,
            metadata_json=json.dumps({"approval_required": False}),
        )

        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        output = stdout.decode(errors="replace")
        error = stderr.decode(errors="replace")
        if process.returncode != 0:
            content = error or f"command exited {process.returncode}"
            await self.store.add_tool_result(tool_call.id, "error", content, metadata_json=json.dumps({"returncode": process.returncode}))
            return ToolResult(status="error", content=content, tool_call_id=tool_call.id)
        await self.store.add_tool_result(tool_call.id, "complete", output, metadata_json=json.dumps({"returncode": 0}))
        return ToolResult(status="complete", content=output, tool_call_id=tool_call.id)

    async def run_approved_shell(self, command: str, tool_call_id: int) -> ToolResult:
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        output = stdout.decode(errors="replace")
        error = stderr.decode(errors="replace")
        if process.returncode != 0:
            content = error or f"command exited {process.returncode}"
            await self.store.add_tool_result(
                tool_call_id,
                "error",
                content,
                metadata_json=json.dumps({"returncode": process.returncode, "approved": True}),
            )
            return ToolResult(status="error", content=content, tool_call_id=tool_call_id)
        await self.store.add_tool_result(
            tool_call_id,
            "complete",
            output,
            metadata_json=json.dumps({"returncode": 0, "approved": True}),
        )
        return ToolResult(status="complete", content=output, tool_call_id=tool_call_id)
