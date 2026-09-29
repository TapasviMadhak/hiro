"""
Base provider interface for all LLM providers.
All providers implement async streaming + non-streaming.
"""
from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Any


@dataclass
class Message:
    role: str  # "user" | "assistant" | "tool" | "system"
    content: str | list[Any]
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str = ""
    name: str = ""  # for tool responses


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema


@dataclass
class StreamChunk:
    """A chunk from a streaming response."""
    text: str = ""
    tool_call: ToolCall | None = None
    tool_call_delta: dict[str, Any] | None = None  # partial tool call
    finish_reason: str = ""  # "stop" | "tool_use" | "length"
    usage: "Usage | None" = None


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
        )


@dataclass
class CompletionResponse:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    finish_reason: str = "stop"
    latency_ms: float = 0.0


class BaseProvider(abc.ABC):
    """Abstract base for all LLM providers."""

    name: str = "base"

    def __init__(self, api_key: str = "", base_url: str = "", **kwargs: Any) -> None:
        self.api_key = api_key
        self.base_url = base_url

    @abc.abstractmethod
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
        """Non-streaming completion."""
        ...

    @abc.abstractmethod
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
        """Streaming completion, yields chunks."""
        ...
        yield StreamChunk()  # satisfy type checker

    def supports_streaming(self) -> bool:
        return True

    def supports_tools(self) -> bool:
        return True

    def supports_vision(self) -> bool:
        return False

    async def list_models(self) -> list[str]:
        return []
