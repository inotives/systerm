from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Awaitable, Callable

from systerm.providers import OpenAICompatibleClient, ProviderError, ToolCall
from systerm.storage import SessionStore
from systerm.tools import ToolDefinition, ToolRunner, openai_tool_schema


EventPublisher = Callable[[str, dict[str, object], int | None], Awaitable[None]]


@dataclass(frozen=True)
class AgentRunResult:
    session_id: int
    content: str
    model_profile: str
    stop_reason: str


class AgentLoop:
    def __init__(
        self,
        client: OpenAICompatibleClient,
        store: SessionStore,
        tools: dict[str, ToolDefinition],
        publish_event: EventPublisher | None = None,
    ) -> None:
        self.client = client
        self.store = store
        self.tools = tools
        self.publish_event = publish_event

    async def run(
        self,
        prompt: str,
        requested_model: str,
        session_id: int | None = None,
    ) -> AgentRunResult:
        if session_id is None:
            session = await self.store.create_session(metadata_json=json.dumps({"requested_model": requested_model}))
            session_id = session.id
            await self._publish("session.created", {"session_id": session_id}, session_id)

        await self.store.add_message(session_id, "user", prompt)
        await self._publish("message.created", {"role": "user"}, session_id)

        records = await self.store.list_messages(session_id)
        messages: list[dict[str, object]] = [
            {"role": message["role"], "content": message["content"]}
            for message in records
            if message["role"] in {"system", "user", "assistant"}
        ]
        result = await self.client.chat(
            messages,
            requested_model=requested_model,
            tools=[openai_tool_schema(tool) for tool in self.tools.values()],
        )
        if result.content:
            await self.store.add_message(
                session_id,
                "assistant",
                result.content,
                result.model_profile,
                metadata_json=json.dumps({"attempted_profiles": list(result.attempted_profiles)}),
            )
            await self._publish("message.created", {"role": "assistant", "model_profile": result.model_profile}, session_id)

        if not result.tool_calls:
            return AgentRunResult(
                session_id=session_id,
                content=result.content,
                model_profile=result.model_profile,
                stop_reason="complete",
            )

        tool_outputs: list[str] = []
        runner = ToolRunner(self.store)
        for tool_call in result.tool_calls:
            output = await self._run_tool_call(runner, session_id, tool_call)
            tool_outputs.append(output.content)
            if output.status == "approval-required":
                await self.store.add_message(session_id, "assistant", output.content, result.model_profile)
                await self._publish(
                    "approval.required",
                    {"approval_id": output.approval_id, "tool_call_id": output.tool_call_id},
                    session_id,
                )
                await self._publish("message.created", {"role": "assistant", "model_profile": result.model_profile}, session_id)
                return AgentRunResult(
                    session_id=session_id,
                    content=output.content,
                    model_profile=result.model_profile,
                    stop_reason="approval-required",
                )

        content = "\n".join(tool_outputs)
        await self.store.add_message(
            session_id,
            "tool",
            content,
            result.model_profile,
            metadata_json=json.dumps({"tool_call_count": len(result.tool_calls)}),
        )
        await self._publish("message.created", {"role": "tool", "model_profile": result.model_profile}, session_id)
        return AgentRunResult(
            session_id=session_id,
            content=content,
            model_profile=result.model_profile,
            stop_reason="tool-use",
        )

    async def _run_tool_call(self, runner: ToolRunner, session_id: int, tool_call: ToolCall):
        if tool_call.name != "shell":
            raise ProviderError(f"Unsupported tool call: {tool_call.name}")
        if tool_call.name not in self.tools:
            raise ProviderError(f"Tool call not allowed by profile: {tool_call.name}")

        try:
            arguments = json.loads(tool_call.arguments)
        except json.JSONDecodeError as exc:
            raise ProviderError(f"Invalid shell tool arguments: {tool_call.arguments}") from exc

        command = arguments.get("command")
        if not isinstance(command, str):
            raise ProviderError("Shell tool call requires a string command")
        result = await runner.run_shell(command, session_id=session_id)
        await self._publish(
            "tool_call.created",
            {"tool_call_id": result.tool_call_id, "status": result.status},
            session_id,
        )
        if result.status != "approval-required":
            await self._publish(
                "tool_result.created",
                {"tool_call_id": result.tool_call_id, "status": result.status},
                session_id,
            )
        return result

    async def _publish(self, event_type: str, payload: dict[str, object], session_id: int | None) -> None:
        if self.publish_event is not None:
            await self.publish_event(event_type, payload, session_id)
