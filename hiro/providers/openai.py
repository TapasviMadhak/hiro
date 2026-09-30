"""
OpenAI-compatible provider.
Works with: OpenAI, DeepSeek, Mistral, OpenRouter, Together, Fireworks, Ollama, etc.
"""
from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

from .base import (
    BaseProvider,
    CompletionResponse,
    Message,
    StreamChunk,
    ToolCall,
    ToolDefinition,
    Usage,
)


def _convert_messages(messages: list[Message]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for msg in messages:
        if msg.role == "tool":
            result.append({
                "role": "tool",
                "tool_call_id": msg.tool_call_id,
                "content": msg.content if isinstance(msg.content, str) else json.dumps(msg.content),
            })
        elif msg.tool_calls:
            result.append({
                "role": "assistant",
                "content": msg.content or None,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        }
                    }
                    for tc in msg.tool_calls
                ],
            })
        else:
            result.append({
                "role": msg.role,
                "content": msg.content,
            })
    return result


def _convert_tools(tools: list[ToolDefinition]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            }
        }
        for t in tools
    ]


class OpenAIProvider(BaseProvider):
    name = "openai"

    def __init__(self, api_key: str = "", base_url: str = "", **kwargs: Any) -> None:
        super().__init__(api_key, base_url, **kwargs)
        try:
            from openai import AsyncOpenAI
            client_kwargs: dict[str, Any] = {
                "api_key": api_key or "placeholder",
            }
            if base_url:
                client_kwargs["base_url"] = base_url
            self._client = AsyncOpenAI(**client_kwargs)
        except ImportError:
            raise ImportError("Install openai: pip install openai")

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
        api_messages: list[dict] = []
        if system:
            api_messages.append({"role": "system", "content": system})
        api_messages.extend(_convert_messages(messages))

        req: dict[str, Any] = {
            "model": model,
            "messages": api_messages,
            "max_tokens": max_tokens,
        }
        # o1/o3/deepseek-reasoner models don't support custom temperature
        if not any(model.startswith(p) for p in ("o1", "o3", "o4", "deepseek-reasoner", "deepseek-r1", "deepseek-ai/deepseek-r1", "deepseek-ai/deepseek-v4")):
            req["temperature"] = temperature
        if tools:
            req["tools"] = _convert_tools(tools)
            req["tool_choice"] = "auto"

        resp = await self._client.chat.completions.create(**req)
        msg = resp.choices[0].message

        text = msg.content or ""
        tool_calls: list[ToolCall] = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except Exception:
                    args = {"raw": tc.function.arguments}
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=args,
                ))

        usage = Usage(
            input_tokens=resp.usage.prompt_tokens if resp.usage else 0,
            output_tokens=resp.usage.completion_tokens if resp.usage else 0,
        )

        return CompletionResponse(
            content=text,
            tool_calls=tool_calls,
            usage=usage,
            model=resp.model,
            finish_reason=resp.choices[0].finish_reason or "stop",
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
        api_messages: list[dict] = []
        if system:
            api_messages.append({"role": "system", "content": system})
        api_messages.extend(_convert_messages(messages))

        req: dict[str, Any] = {
            "model": model,
            "messages": api_messages,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if not any(model.startswith(p) for p in ("o1", "o3", "o4", "deepseek-reasoner", "deepseek-r1", "deepseek-ai/deepseek-r1", "deepseek-ai/deepseek-v4")):
            req["temperature"] = temperature
        if tools:
            req["tools"] = _convert_tools(tools)
            req["tool_choice"] = "auto"

        # Track accumulated tool call data
        tool_calls_acc: dict[int, dict] = {}

        try:
            stream_resp = await self._client.chat.completions.create(**req)
        except Exception:
            if "stream_options" in req:
                req.pop("stream_options", None)
                stream_resp = await self._client.chat.completions.create(**req)
            else:
                raise

        async for chunk in stream_resp:
            if not chunk.choices:
                # usage chunk
                if chunk.usage:
                    yield StreamChunk(usage=Usage(
                        input_tokens=chunk.usage.prompt_tokens,
                        output_tokens=chunk.usage.completion_tokens,
                    ))
                continue

            delta = chunk.choices[0].delta
            finish = chunk.choices[0].finish_reason

            if delta.content and delta.content.strip():
                yield StreamChunk(text=delta.content)

            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_calls_acc:
                        # First delta for this tool call — capture ALL fields including args.
                        # Some models (Kimi K3, GLM, DeepSeek) send the complete arguments
                        # in the very first chunk alongside the function name.
                        first_args = ""
                        if tc_delta.function and tc_delta.function.arguments:
                            first_args = tc_delta.function.arguments
                        tool_calls_acc[idx] = {
                            "id": tc_delta.id or "",
                            "name": tc_delta.function.name if tc_delta.function else "",
                            "args": first_args,
                        }
                    else:
                        if tc_delta.id:
                            tool_calls_acc[idx]["id"] = tc_delta.id
                        if tc_delta.function:
                            if tc_delta.function.name:
                                tool_calls_acc[idx]["name"] = tc_delta.function.name
                            if tc_delta.function.arguments:
                                tool_calls_acc[idx]["args"] += tc_delta.function.arguments


            if finish:
                # Emit completed tool calls
                for idx, tc_data in sorted(tool_calls_acc.items()):
                    try:
                        args = json.loads(tc_data["args"]) if tc_data["args"] else {}
                    except Exception:
                        args = {"raw": tc_data["args"]}
                    yield StreamChunk(tool_call=ToolCall(
                        id=tc_data["id"],
                        name=tc_data["name"],
                        arguments=args,
                    ))
                tool_calls_acc.clear()
                yield StreamChunk(finish_reason=finish)

    def supports_vision(self) -> bool:
        return True

    async def list_models(self) -> list[str]:
        try:
            models = await self._client.models.list()
            return [m.id for m in models.data]
        except Exception:
            return []
