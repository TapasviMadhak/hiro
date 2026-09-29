"""
Anthropic Claude provider.
Uses the official anthropic SDK with streaming + prompt caching support.
"""
from __future__ import annotations

import time
import json
from typing import AsyncIterator, Any

from .base import (
    BaseProvider, Message, ToolCall, ToolDefinition,
    StreamChunk, CompletionResponse, Usage
)


def _convert_messages(messages: list[Message]) -> list[dict]:
    """Convert internal Message format to Anthropic API format."""
    result = []
    for msg in messages:
        if msg.role == "system":
            continue  # handled separately

        if msg.role == "tool":
            # Tool result
            result.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": msg.tool_call_id,
                    "content": msg.content if isinstance(msg.content, str) else json.dumps(msg.content),
                }]
            })
        elif msg.tool_calls:
            # Assistant message with tool use
            content: list[dict] = []
            if msg.content:
                content.append({"type": "text", "text": str(msg.content)})
            for tc in msg.tool_calls:
                content.append({
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.name,
                    "input": tc.arguments,
                })
            result.append({"role": "assistant", "content": content})
        else:
            content_val: Any = msg.content
            if isinstance(msg.content, str):
                content_val = msg.content
            result.append({"role": msg.role, "content": content_val})

    return result


def _convert_tools(tools: list[ToolDefinition]) -> list[dict]:
    return [
        {
            "name": t.name,
            "description": t.description,
            "input_schema": t.parameters,
        }
        for t in tools
    ]


class AnthropicProvider(BaseProvider):
    name = "anthropic"

    def __init__(self, api_key: str = "", base_url: str = "", **kwargs: Any) -> None:
        super().__init__(api_key, base_url, **kwargs)
        try:
            import anthropic
            self._anthropic = anthropic
            client_kwargs: dict[str, Any] = {"api_key": api_key or "placeholder"}
            if base_url:
                client_kwargs["base_url"] = base_url
            self._client = anthropic.AsyncAnthropic(**client_kwargs)
        except ImportError:
            raise ImportError("Install anthropic: pip install anthropic")

    async def complete(
        self,
        messages: list[Message],
        model: str,
        system: str = "",
        tools: list[ToolDefinition] | None = None,
        max_tokens: int = 8192,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> CompletionResponse:
        t0 = time.monotonic()
        api_messages = _convert_messages(messages)

        kwargs_req: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": api_messages,
        }
        if system:
            kwargs_req["system"] = system
        if tools:
            kwargs_req["tools"] = _convert_tools(tools)
        if temperature != 0.0:
            kwargs_req["temperature"] = temperature

        resp = await self._client.messages.create(**kwargs_req)

        text = ""
        tool_calls: list[ToolCall] = []
        for block in resp.content:
            if block.type == "text":
                text += block.text
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input,
                ))

        usage = Usage(
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            cache_read_tokens=getattr(resp.usage, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(resp.usage, "cache_creation_input_tokens", 0) or 0,
        )

        return CompletionResponse(
            content=text,
            tool_calls=tool_calls,
            usage=usage,
            model=resp.model,
            finish_reason=resp.stop_reason or "stop",
            latency_ms=(time.monotonic() - t0) * 1000,
        )

    async def stream(
        self,
        messages: list[Message],
        model: str,
        system: str = "",
        tools: list[ToolDefinition] | None = None,
        max_tokens: int = 8192,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        api_messages = _convert_messages(messages)

        kwargs_req: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": api_messages,
        }
        if system:
            kwargs_req["system"] = system
        if tools:
            kwargs_req["tools"] = _convert_tools(tools)
        if temperature != 0.0:
            kwargs_req["temperature"] = temperature

        # Track partial tool calls
        current_tool_id = ""
        current_tool_name = ""
        current_tool_args = ""

        async with self._client.messages.stream(**kwargs_req) as stream:
            async for event in stream:
                ev_type = event.type

                if ev_type == "content_block_start":
                    block = event.content_block
                    if block.type == "tool_use":
                        current_tool_id = block.id
                        current_tool_name = block.name
                        current_tool_args = ""

                elif ev_type == "content_block_delta":
                    delta = event.delta
                    if delta.type == "text_delta":
                        yield StreamChunk(text=delta.text)
                    elif delta.type == "input_json_delta":
                        current_tool_args += delta.partial_json
                        yield StreamChunk(tool_call_delta={
                            "id": current_tool_id,
                            "name": current_tool_name,
                            "partial_args": delta.partial_json,
                        })

                elif ev_type == "content_block_stop":
                    if current_tool_id:
                        try:
                            args = json.loads(current_tool_args) if current_tool_args else {}
                        except json.JSONDecodeError:
                            args = {"raw": current_tool_args}
                        yield StreamChunk(
                            tool_call=ToolCall(
                                id=current_tool_id,
                                name=current_tool_name,
                                arguments=args,
                            )
                        )
                        current_tool_id = ""
                        current_tool_name = ""
                        current_tool_args = ""

                elif ev_type == "message_stop":
                    pass

                elif ev_type == "message_delta":
                    if hasattr(event, "usage") and event.usage:
                        yield StreamChunk(
                            finish_reason=getattr(event.delta, "stop_reason", "") or "",
                            usage=Usage(output_tokens=event.usage.output_tokens),
                        )

                elif ev_type == "message_start":
                    if hasattr(event, "message") and hasattr(event.message, "usage"):
                        u = event.message.usage
                        yield StreamChunk(
                            usage=Usage(
                                input_tokens=u.input_tokens,
                                cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
                                cache_write_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
                            )
                        )

    def supports_vision(self) -> bool:
        return True

    async def list_models(self) -> list[str]:
        try:
            models = await self._client.models.list()
            return [m.id for m in models.data]
        except Exception:
            return []
