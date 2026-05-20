from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROFILE_BLOCK_RE = re.compile(r"```toml\s+(?P<body>.*?)```", re.DOTALL)


class AgentProfileError(ValueError):
    pass


@dataclass(frozen=True)
class AgentProfile:
    name: str
    model: str
    skills: tuple[str, ...]
    tools: tuple[str, ...]


def load_agent_profile(path: Path) -> AgentProfile:
    if not path.exists():
        raise AgentProfileError(f"Missing agent profile: {path}")

    text = path.read_text(encoding="utf-8")
    match = PROFILE_BLOCK_RE.search(text)
    if not match:
        raise AgentProfileError("AGENTS.md must contain a fenced toml profile block")

    try:
        raw = tomllib.loads(match.group("body"))
    except tomllib.TOMLDecodeError as exc:
        raise AgentProfileError(f"Invalid AGENTS.md toml block: {exc}") from exc

    return parse_agent_profile(raw)


def parse_agent_profile(raw: dict[str, Any]) -> AgentProfile:
    agent = raw.get("agent")
    if not isinstance(agent, dict):
        raise AgentProfileError("AGENTS.md toml block must define [agent]")

    name = agent.get("name")
    model = agent.get("model")
    skills = raw.get("skills")
    tools = raw.get("tools")

    if not isinstance(name, str) or not name:
        raise AgentProfileError("AGENTS.md toml block must define agent.name")
    if not isinstance(model, str) or not model:
        raise AgentProfileError("AGENTS.md toml block must define agent.model")
    if not _is_string_list(skills):
        raise AgentProfileError("AGENTS.md toml block must define skills as a string array")
    if not _is_string_list(tools):
        raise AgentProfileError("AGENTS.md toml block must define tools as a string array")

    return AgentProfile(name=name, model=model, skills=tuple(skills), tools=tuple(tools))


def missing_references(profile: AgentProfile, project_root: Path) -> list[str]:
    missing: list[str] = []
    for skill in profile.skills:
        if not (project_root / ".agents" / "skills" / skill / "SKILL.md").exists():
            missing.append(f"skill:{skill}")
    for tool in profile.tools:
        if not (project_root / ".agents" / "tools" / tool / "tool.toml").exists():
            missing.append(f"tool:{tool}")
    return missing


def _is_string_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)
