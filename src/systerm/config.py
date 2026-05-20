from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError


class ProviderConfig(BaseModel):
    base_url: str
    api_key_env: str | None = None
    api_style: str = "openai_chat"
    supports_streaming: bool = True
    supports_tools: bool = False
    models: list[str] = Field(default_factory=list)


class ModelProfile(BaseModel):
    provider: str
    model: str
    timeout_seconds: float = 30
    retries: int = 0
    context_window: int | None = None
    max_tokens: int | None = None
    temperature: float | None = None


class ModelsConfig(BaseModel):
    default_model: str
    fallback_models: list[str] = Field(default_factory=list)


class AppConfig(BaseModel):
    models: ModelsConfig
    providers: dict[str, ProviderConfig]
    model_profiles: dict[str, ModelProfile]

    def model_chain(self, requested: str | None = None) -> list[str]:
        first = requested or self.models.default_model
        chain = [first]
        for profile in self.models.fallback_models:
            if profile not in chain:
                chain.append(profile)
        return chain


class ConfigError(ValueError):
    pass


def load_config(project_root: Path) -> AppConfig:
    load_dotenv(project_root / ".env")

    raw: dict[str, Any] = {}
    for path in config_paths(project_root):
        if path.exists():
            raw = _deep_merge(raw, _load_toml(path))

    if not raw:
        raise ConfigError("No config found. Expected ~/.config/systerm/config.toml or .systerm/config.toml")

    try:
        config = AppConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(str(exc)) from exc

    validate_config(config)
    return config


def config_paths(project_root: Path) -> list[Path]:
    return [
        Path.home() / ".config" / "systerm" / "model.toml",
        Path.home() / ".config" / "systerm" / "config.toml",
        project_root / ".systerm" / "model.toml",
        project_root / ".systerm" / "config.toml",
    ]


def resolve_api_key(provider: ProviderConfig) -> str:
    if not provider.api_key_env:
        return ""
    return os.getenv(provider.api_key_env, "")


def validate_config(config: AppConfig) -> None:
    referenced = set(config.model_profiles)
    for name, profile in config.model_profiles.items():
        provider = config.providers.get(profile.provider)
        if provider is None:
            raise ConfigError(f"Model profile {name!r} references missing provider {profile.provider!r}")
        if provider.models and profile.model not in provider.models:
            raise ConfigError(
                f"Model profile {name!r} references model {profile.model!r} "
                f"not listed by provider {profile.provider!r}"
            )

    for profile in config.model_chain():
        if profile not in referenced:
            raise ConfigError(f"Configured model profile {profile!r} does not exist")


def validate_model_profile(config: AppConfig, profile_name: str) -> None:
    if profile_name not in config.model_profiles:
        raise ConfigError(f"Agent references missing model profile {profile_name!r}")


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as file:
        return tomllib.load(file)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
