"""
Google Gemini provider using the Google GenAI SDK.
"""
from __future__ import annotations

import json
import time
from typing import AsyncIterator, Any

from .base import (
    BaseProvider, Message, ToolCall, ToolDefinition,
    StreamChunk, CompletionResponse, Usage
)


class GoogleProvider(BaseProvider):
    name = "google"

    def __init__(self, api_key: str = "", base_url: str = "", **kwargs: Any) -> None:
        super().__init__(api_key, base_url, **kwargs)
        try:
            import google.generativeai as genai
            self._genai = genai
            genai.configure(api_key=api_key)
        except ImportError:
            raise ImportError("Install google-generativeai: pip install google-generativeai")

    def _build_contents(self, messages: list[Message]) -> list[dict]:
        contents = []
        for msg in messages:
            if msg.role == "system":
                continue
            role = "model" if msg.role == "assistant" else "user"
            if msg.role == "tool":
                contents.append({
                    "role": "user",
                    "parts": [{"function_response": {
                        "name": msg.name,
                        "response": {"result": msg.content},
                    }}]
                })
            elif msg.tool_calls:
                parts = []
                if msg.content:
                    parts.append({"text": str(msg.content)})
                for tc in msg.tool_calls:
                    parts.append({"function_call": {"name": tc.name, "args": tc.arguments}})
                contents.append({"role": "model", "parts": parts})
            else:
                contents.append({
                    "role": role,
                    "parts": [{"text": str(msg.content)}],
                })
        return contents

    def _build_tools(self, tools: list[ToolDefinition]) -> list[dict]:
        return [{
            "function_declarations": [
                {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                }
                for t in tools
            ]
        }]

    def _get_system(self, messages: list[Message], system: str) -> str:
        sys_parts = [system] if system else []
        for msg in messages:
            if msg.role == "system":
                sys_parts.append(str(msg.content))
        return "\n\n".join(sys_parts)

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
        sys_text = self._get_system(messages, system)

        gen_config = self._genai.types.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        )

        model_kwargs: dict[str, Any] = {"generation_config": gen_config}
        if sys_text:
            model_kwargs["system_instruction"] = sys_text
        if tools:
            model_kwargs["tools"] = self._build_tools(tools)

        client = self._genai.GenerativeModel(model, **model_kwargs)
        contents = self._build_contents(messages)

        resp = await client.generate_content_async(contents)
        text = ""
        tool_calls: list[ToolCall] = []

        for part in resp.parts:
            if hasattr(part, "text") and part.text:
                text += part.text
            if hasattr(part, "function_call") and part.function_call:
                fc = part.function_call
                tool_calls.append(ToolCall(
                    id=fc.name,
                    name=fc.name,
                    arguments=dict(fc.args),
                ))

        usage = Usage()
        if resp.usage_metadata:
            usage.input_tokens = resp.usage_metadata.prompt_token_count or 0
            usage.output_tokens = resp.usage_metadata.candidates_token_count or 0

        return CompletionResponse(
            content=text,
            tool_calls=tool_calls,
            usage=usage,
            model=model,
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
        sys_text = self._get_system(messages, system)
        gen_config = self._genai.types.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        )
        model_kwargs: dict[str, Any] = {"generation_config": gen_config}
        if sys_text:
            model_kwargs["system_instruction"] = sys_text
        if tools:
            model_kwargs["tools"] = self._build_tools(tools)

        client = self._genai.GenerativeModel(model, **model_kwargs)
        contents = self._build_contents(messages)

        async for chunk in await client.generate_content_async(contents, stream=True):
            for part in chunk.parts:
                if hasattr(part, "text") and part.text:
                    yield StreamChunk(text=part.text)
                if hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    yield StreamChunk(tool_call=ToolCall(
                        id=fc.name,
                        name=fc.name,
                        arguments=dict(fc.args),
                    ))

        yield StreamChunk(finish_reason="stop")

    def supports_vision(self) -> bool:
        return True
