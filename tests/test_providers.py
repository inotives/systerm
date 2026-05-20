import httpx
import pytest

from systerm.config import AppConfig
from systerm.providers import OpenAICompatibleClient, ProviderError


@pytest.mark.asyncio
async def test_provider_falls_back_after_retriable_error() -> None:
    config = AppConfig.model_validate(
        {
            "models": {"default_model": "fast", "fallback_models": ["slow"]},
            "providers": {
                "test": {
                    "base_url": "https://example.test/v1",
                    "api_key_env": "TEST_API_KEY",
                }
            },
            "model_profiles": {
                "fast": {"provider": "test", "model": "fast-model", "max_tokens": 64, "temperature": 0.1},
                "slow": {"provider": "test", "model": "slow-model"},
            },
        }
    )

    seen_payloads: list[dict[str, object]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = request.read().decode()
        seen_payloads.append(__import__("json").loads(body))
        if "fast-model" in body:
            return httpx.Response(503)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = OpenAICompatibleClient(config, http_client=http_client)
        result = await client.chat([{"role": "user", "content": "hello"}])

    assert result.content == "ok"
    assert result.model_profile == "slow"
    assert result.attempted_profiles == ("fast", "slow")
    assert seen_payloads[0]["model"] == "fast-model"
    assert seen_payloads[0]["max_tokens"] == 64
    assert seen_payloads[0]["temperature"] == 0.1


@pytest.mark.asyncio
async def test_provider_reports_non_retriable_http_errors() -> None:
    config = AppConfig.model_validate(
        {
            "models": {"default_model": "fast", "fallback_models": []},
            "providers": {"test": {"base_url": "https://example.test/v1"}},
            "model_profiles": {"fast": {"provider": "test", "model": "fast-model"}},
        }
    )

    transport = httpx.MockTransport(lambda request: httpx.Response(413, request=request))
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = OpenAICompatibleClient(config, http_client=http_client)
        with pytest.raises(ProviderError, match="HTTP 413"):
            await client.chat([{"role": "user", "content": "hello"}])


@pytest.mark.asyncio
async def test_provider_extracts_tool_calls() -> None:
    config = AppConfig.model_validate(
        {
            "models": {"default_model": "fast", "fallback_models": []},
            "providers": {"test": {"base_url": "https://example.test/v1", "supports_tools": True}},
            "model_profiles": {"fast": {"provider": "test", "model": "fast-model"}},
        }
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "shell", "arguments": '{"command": "echo hi"}'},
                                }
                            ],
                        }
                    }
                ]
            },
            request=request,
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = OpenAICompatibleClient(config, http_client=http_client)
        result = await client.chat(
            [{"role": "user", "content": "hello"}],
            tools=[{"type": "function", "function": {"name": "shell", "parameters": {}}}],
        )

    assert result.tool_calls[0].id == "call_1"
    assert result.tool_calls[0].name == "shell"
