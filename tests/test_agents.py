from pathlib import Path

import pytest

from systerm.agents import AgentProfileError, load_agent_profile, missing_references


def test_load_agent_profile(tmp_path: Path) -> None:
    profile_path = tmp_path / "AGENTS.md"
    profile_path.write_text(
        """
Instructions.

```toml
skills = ["diagnose"]
tools = ["shell"]

[agent]
name = "systerm"
model = "local"
```
""",
        encoding="utf-8",
    )

    profile = load_agent_profile(profile_path)

    assert profile.name == "systerm"
    assert profile.model == "local"
    assert profile.skills == ("diagnose",)
    assert profile.tools == ("shell",)


def test_load_agent_profile_requires_toml_block(tmp_path: Path) -> None:
    profile_path = tmp_path / "AGENTS.md"
    profile_path.write_text("no profile", encoding="utf-8")

    with pytest.raises(AgentProfileError):
        load_agent_profile(profile_path)


def test_missing_references(tmp_path: Path) -> None:
    profile_path = tmp_path / "AGENTS.md"
    profile_path.write_text(
        """
```toml
skills = ["diagnose"]
tools = []

[agent]
name = "systerm"
model = "local"
```
""",
        encoding="utf-8",
    )

    profile = load_agent_profile(profile_path)

    assert missing_references(profile, tmp_path) == ["skill:diagnose"]
