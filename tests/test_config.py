from pathlib import Path

import pytest

from systerm.config import ConfigError, load_config, validate_model_profile


def test_load_project_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    config_dir = tmp_path / ".systerm"
    config_dir.mkdir()
    (config_dir / "config.toml").write_text(
        """
[models]
default_model = "fast"
fallback_models = ["slow"]

[providers.test]
base_url = "https://example.test/v1"
api_key_env = "TEST_API_KEY"
models = ["fast-model", "slow-model"]

[model_profiles.fast]
provider = "test"
model = "fast-model"
context_window = 128000
max_tokens = 8192
temperature = 0.2

[model_profiles.slow]
provider = "test"
model = "slow-model"
""",
        encoding="utf-8",
    )

    config = load_config(tmp_path)

    assert config.model_chain() == ["fast", "slow"]
    assert config.providers["test"].base_url == "https://example.test/v1"
    assert config.providers["test"].models == ["fast-model", "slow-model"]
    assert config.model_profiles["fast"].context_window == 128000
    assert config.model_profiles["fast"].max_tokens == 8192
    assert config.model_profiles["fast"].temperature == 0.2


def test_validate_model_profile_rejects_missing_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    config_dir = tmp_path / ".systerm"
    config_dir.mkdir()
    (config_dir / "config.toml").write_text(
        """
[models]
default_model = "fast"

[providers.test]
base_url = "https://example.test/v1"

[model_profiles.fast]
provider = "test"
model = "fast-model"
""",
        encoding="utf-8",
    )

    config = load_config(tmp_path)

    with pytest.raises(ConfigError):
        validate_model_profile(config, "missing")


def test_load_config_merges_model_toml_before_config_toml(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    config_dir = tmp_path / ".systerm"
    config_dir.mkdir()
    (config_dir / "model.toml").write_text(
        """
[providers.test]
base_url = "https://example.test/v1"
models = ["fast-model", "slow-model"]

[model_profiles.fast]
provider = "test"
model = "fast-model"

[model_profiles.slow]
provider = "test"
model = "slow-model"
""",
        encoding="utf-8",
    )
    (config_dir / "config.toml").write_text(
        """
[models]
default_model = "fast"
fallback_models = ["slow"]
""",
        encoding="utf-8",
    )

    config = load_config(tmp_path)

    assert config.model_chain() == ["fast", "slow"]
    assert sorted(config.model_profiles) == ["fast", "slow"]


def test_load_config_supports_quoted_profile_names_with_dots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    config_dir = tmp_path / ".systerm"
    config_dir.mkdir()
    (config_dir / "model.toml").write_text(
        """
[providers.test]
base_url = "https://example.test/v1"
models = ["fast-model"]

[model_profiles."fast-1.2"]
provider = "test"
model = "fast-model"
""",
        encoding="utf-8",
    )
    (config_dir / "config.toml").write_text(
        """
[models]
default_model = "fast-1.2"
""",
        encoding="utf-8",
    )

    config = load_config(tmp_path)

    assert config.model_chain() == ["fast-1.2"]


def test_load_config_rejects_profile_model_missing_from_provider_list(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    config_dir = tmp_path / ".systerm"
    config_dir.mkdir()
    (config_dir / "config.toml").write_text(
        """
[models]
default_model = "fast"

[providers.test]
base_url = "https://example.test/v1"
models = ["other-model"]

[model_profiles.fast]
provider = "test"
model = "fast-model"
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError):
        load_config(tmp_path)
