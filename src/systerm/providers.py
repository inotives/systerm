from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from systerm.config import AppConfig, resolve_api_key


class ProviderError(RuntimeError):
    pass


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class ChatResult:
    content: str
    model_profile: str
    attempted_profiles: tuple[str, ...]
    tool_calls: tuple[ToolCall, ...] = ()


class OpenAICompatibleClient:
    def __init__(self, config: AppConfig, http_client: httpx.AsyncClient | None = None) -> None:
        self._config = config
        self._http_client = http_client

    async def chat(
        self,
        messages: list[dict[str, Any]],
        requested_model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatResult:
        attempted: list[str] = []
        last_error: Exception | None = None

        for profile_name in self._config.model_chain(requested_model):
            attempted.append(profile_name)
            try:
                content, tool_calls = await self._chat_once(profile_name, messages, tools)
            except (httpx.TimeoutException, httpx.TransportError, ProviderError) as exc:
                last_error = exc
                continue

            if content or tool_calls:
                return ChatResult(
                    content=content,
                    model_profile=profile_name,
                    attempted_profiles=tuple(attempted),
                    tool_calls=tool_calls,
                )
            last_error = ProviderError(f"Model profile {profile_name!r} returned no assistant response")

        raise ProviderError(f"All model profiles failed: {last_error}") from last_error

    async def _chat_once(
        self,
        profile_name: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> tuple[str, tuple[ToolCall, ...]]:
        profile = self._config.model_profiles[profile_name]
        provider = self._config.providers[profile.provider]
        headers = {"Content-Type": "application/json"}
        api_key = resolve_api_key(provider)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        payload: dict[str, Any] = {
            "model": profile.model,
            "messages": messages,
            "stream": False,
        }
        if profile.max_tokens is not None:
            payload["max_tokens"] = profile.max_tokens
        if profile.temperature is not None:
            payload["temperature"] = profile.temperature
        if tools and provider.supports_tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        client = self._http_client or httpx.AsyncClient(timeout=profile.timeout_seconds)
        should_close = self._http_client is None
        try:
            response = await client.post(
                f"{provider.base_url.rstrip('/')}/chat/completions",
                headers=headers,
                json=payload,
            )
            if response.status_code >= 500 or response.status_code in {408, 409, 425, 429}:
                raise ProviderError(f"Retriable provider error: HTTP {response.status_code}")
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise ProviderError(
                    f"Provider rejected request: HTTP {response.status_code} from {profile_name}"
                ) from exc
            return _extract_assistant_response(response.json())
        finally:
            if should_close:
                await client.aclose()


def _extract_assistant_response(data: dict[str, Any]) -> tuple[str, tuple[ToolCall, ...]]:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return "", ()

    message = choices[0].get("message")
    if not isinstance(message, dict):
        return "", ()

    content = message.get("content")
    tool_calls = message.get("tool_calls")
    return (
        content if isinstance(content, str) else "",
        _extract_tool_calls(tool_calls),
    )


def _extract_tool_calls(value: Any) -> tuple[ToolCall, ...]:
    if not isinstance(value, list):
        return ()

    calls: list[ToolCall] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        function = item.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        arguments = function.get("arguments")
        if not isinstance(name, str) or not isinstance(arguments, str):
            continue
        call_id = item.get("id")
        calls.append(ToolCall(id=call_id if isinstance(call_id, str) else "", name=name, arguments=arguments))
    return tuple(calls)
