"""
Agent execution loop.
Handles multi-turn conversations with tool use, MCP integration,
and token budget management.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable, Any

from hiro.providers.base import (
    BaseProvider, Message, ToolCall, ToolDefinition,
    StreamChunk, CompletionResponse, Usage
)
from hiro.tools import BuiltinToolExecutor, get_builtin_tools
from hiro.mcp import MCPManager, MCPTool
from hiro.config import Settings


@dataclass
class AgentEvent:
    """Events emitted by the agent loop."""
    type: str  # text | tool_start | tool_end | error | done | thinking
    data: Any = None


@dataclass
class AgentContext:
    """The conversation context (message history)."""
    messages: list[Message] = field(default_factory=list)
    total_usage: Usage = field(default_factory=Usage)
    turn_count: int = 0
    session_start: float = field(default_factory=time.time)

    def add_user(self, content: str) -> None:
        self.messages.append(Message(role="user", content=content))

    def add_assistant(self, content: str, tool_calls: list[ToolCall] = None) -> None:
        self.messages.append(Message(
            role="assistant",
            content=content,
            tool_calls=tool_calls or [],
        ))

    def add_tool_result(self, tool_call_id: str, name: str, result: str) -> None:
        self.messages.append(Message(
            role="tool",
            content=result,
            tool_call_id=tool_call_id,
            name=name,
        ))

    def clear(self) -> None:
        self.messages.clear()
        self.turn_count = 0

    def token_estimate(self) -> int:
        """Rough token estimate: ~4 chars per token."""
        total_chars = sum(
            len(str(m.content)) for m in self.messages
        )
        return total_chars // 4

    def compact(self, keep_recent: int = 10) -> str:
        """Compact the context by summarizing old messages."""
        if len(self.messages) <= keep_recent:
            return ""
        old = self.messages[:-keep_recent]
        self.messages = self.messages[-keep_recent:]
        dropped = len(old)
        return f"Compacted: removed {dropped} old messages"


def _mcp_tool_to_definition(tool: MCPTool, prefixed_name: str) -> ToolDefinition:
    """Convert MCP tool to provider ToolDefinition."""
    schema = tool.input_schema or {}
    if "type" not in schema:
        schema = {"type": "object", "properties": {}}
    return ToolDefinition(
        name=prefixed_name,
        description=f"[{tool.server_name}] {tool.description}",
        parameters=schema,
    )


class AgentLoop:
    """
    The main agent execution loop.
    Handles streaming, tool calls, MCP integration.
    Token-efficient: no mandatory system prompts.
    """

    def __init__(
        self,
        provider: BaseProvider,
        model: str,
        settings: Settings,
        mcp_manager: MCPManager | None = None,
        context: AgentContext | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.settings = settings
        self.mcp = mcp_manager
        self.ctx = context or AgentContext()
        self._tool_executor = BuiltinToolExecutor(
            cwd=settings.working_directory or "",
            allow_shell=settings.allow_shell,
            allow_network=settings.allow_network,
        )

    def _get_all_tools(self) -> list[ToolDefinition]:
        """Collect all available tools: built-in + MCP."""
        tools = get_builtin_tools(allow_shell=self.settings.allow_shell)

        if self.mcp:
            for mcp_tool in self.mcp.get_all_tools():
                # Use a unique name that won't conflict
                prefixed = mcp_tool.name
                tools.append(_mcp_tool_to_definition(mcp_tool, prefixed))

        return tools

    async def _execute_tool(self, tool_call: ToolCall) -> str:
        """Execute a tool call, routing to built-in or MCP."""
        name = tool_call.name
        args = tool_call.arguments

        # Validate required parameters before dispatching
        all_tool_defs = get_builtin_tools()
        if self.mcp:
            for mcp_tool in self.mcp.get_all_tools():
                all_tool_defs.append(_mcp_tool_to_definition(mcp_tool, mcp_tool.name))
        for tdef in all_tool_defs:
            if tdef.name == name:
                required = tdef.parameters.get("required", [])
                missing = [r for r in required if r not in args]
                if missing:
                    return (
                        f"Error: tool '{name}' requires these missing arguments: {missing}. "
                        f"Please call again with all required fields."
                    )
                break

        # Check built-in tools first
        builtin_names = {t.name for t in get_builtin_tools()}
        if name in builtin_names:
            return await self._tool_executor.execute(name, args)

        # Try MCP
        if self.mcp:
            result = await self.mcp.call_tool(name, args)
            return result.to_text()

        return f"Error: Unknown tool: {name}"

    async def run(
        self,
        user_input: str,
        system: str = "",
        on_event: Callable[[AgentEvent], None] | None = None,
    ) -> str:
        """
        Run one full agent turn (potentially multiple LLM calls due to tool use).
        Yields events via on_event callback.
        Returns final text response.
        """
        emit = on_event or (lambda e: None)

        # Add user message
        self.ctx.add_user(user_input)
        self.ctx.turn_count += 1
        self._consecutive_error_rounds = 0  # reset per-turn error counter

        all_tools = self._get_all_tools()
        system_prompt = system or self.settings.system_prompt
        max_turns = self.settings.agent.max_turns
        final_text = ""

        for iteration in range(max_turns):
            # Check token budget
            token_est = self.ctx.token_estimate()
            budget = self.settings.agent.token_budget
            if token_est > budget * self.settings.agent.compact_threshold and self.settings.agent.auto_compact:
                msg = self.ctx.compact()
                emit(AgentEvent("thinking", f"Auto-compacted context: {msg}"))

            # Stream response
            text_parts: list[str] = []
            tool_calls: list[ToolCall] = []
            finish_reason = ""
            turn_usage = Usage()

            try:
                async for chunk in self.provider.stream(
                    messages=self.ctx.messages,
                    model=self.model,
                    system=system_prompt,
                    tools=all_tools if all_tools else None,
                    max_tokens=self.settings.agent.max_tokens_per_turn,
                    temperature=self.settings.temperature,
                ):
                    if chunk.text:
                        text_parts.append(chunk.text)
                        emit(AgentEvent("text", chunk.text))

                    if chunk.tool_call:
                        # Deduplicate by tool call ID to prevent double-emission
                        if not any(tc.id == chunk.tool_call.id for tc in tool_calls):
                            tool_calls.append(chunk.tool_call)
                            emit(AgentEvent("tool_start", chunk.tool_call))

                    if chunk.usage:
                        turn_usage = turn_usage + chunk.usage

                    if chunk.finish_reason:
                        finish_reason = chunk.finish_reason

            except Exception as e:
                emit(AgentEvent("error", str(e)))
                # Fall back to non-streaming
                try:
                    resp = await self.provider.complete(
                        messages=self.ctx.messages,
                        model=self.model,
                        system=system_prompt,
                        tools=all_tools if all_tools else None,
                        max_tokens=self.settings.agent.max_tokens_per_turn,
                        temperature=self.settings.temperature,
                    )
                    text_parts = [resp.content]
                    tool_calls = resp.tool_calls
                    turn_usage = resp.usage
                    finish_reason = resp.finish_reason
                    if resp.content:
                        emit(AgentEvent("text", resp.content))
                    for tc in resp.tool_calls:
                        emit(AgentEvent("tool_start", tc))
                except Exception as e2:
                    emit(AgentEvent("error", f"Fatal: {e2}"))
                    return f"Error: {e2}"

            assistant_text = "".join(text_parts).strip()

            # Guard: if model returned nothing (no text, no tools), nudge it once
            if not assistant_text and not tool_calls:
                emit(AgentEvent("thinking", "Empty model response — retrying with nudge..."))
                # Add a minimal assistant placeholder so context stays valid,
                # then inject a user nudge to get the model moving.
                self.ctx.messages.append(Message(role="assistant", content="..."))
                self.ctx.messages.append(Message(role="user", content="Please continue."))
                continue

            self.ctx.add_assistant(assistant_text, tool_calls)
            self.ctx.total_usage = self.ctx.total_usage + turn_usage

            emit(AgentEvent("usage", turn_usage))

            # If no tool calls, we're done
            if not tool_calls:
                final_text = assistant_text
                break

            # Execute all tool calls — dedupe by id first
            seen_ids: set[str] = set()
            unique_calls = [tc for tc in tool_calls if tc.id not in seen_ids and not seen_ids.add(tc.id)]

            # Guard: if ALL tool results in previous round were errors, cap retries
            # (detect by checking if context already has this exact set of error results)
            consecutive_error_rounds = getattr(self, "_consecutive_error_rounds", 0)

            if self.settings.agent.parallel_tools and len(unique_calls) > 1:
                # Parallel execution
                results = await asyncio.gather(
                    *[self._execute_tool(tc) for tc in unique_calls],
                    return_exceptions=True,
                )
                all_errors = True
                for tc, result in zip(unique_calls, results):
                    if isinstance(result, Exception):
                        result = f"Error: {result}"
                    result_str = str(result)
                    if not result_str.startswith("Error:"):
                        all_errors = False
                    emit(AgentEvent("tool_end", {"call": tc, "result": result_str}))
                    self.ctx.add_tool_result(tc.id, tc.name, result_str)
            else:
                # Sequential execution
                all_errors = True
                for tc in unique_calls:
                    result = await self._execute_tool(tc)
                    if not result.startswith("Error:"):
                        all_errors = False
                    emit(AgentEvent("tool_end", {"call": tc, "result": result}))
                    self.ctx.add_tool_result(tc.id, tc.name, result)

            # Track consecutive all-error rounds to break infinite loops
            if all_errors:
                self._consecutive_error_rounds = consecutive_error_rounds + 1
                if self._consecutive_error_rounds >= 3:
                    msg = "Stopping: 3 consecutive rounds of tool errors. The tools may need different arguments."
                    emit(AgentEvent("error", msg))
                    final_text = msg
                    break
            else:
                self._consecutive_error_rounds = 0

        else:
            final_text = f"Warning: Max iterations ({max_turns}) reached"
            emit(AgentEvent("error", final_text))

        emit(AgentEvent("done", final_text))
        return final_text

    async def run_stream(
        self,
        user_input: str,
        system: str = "",
    ) -> AsyncIterator[AgentEvent]:
        """Async generator version of run()."""
        queue: asyncio.Queue[AgentEvent | None] = asyncio.Queue()

        def on_event(evt: AgentEvent) -> None:
            queue.put_nowait(evt)

        async def _run() -> None:
            await self.run(user_input, system, on_event)
            queue.put_nowait(None)

        task = asyncio.create_task(_run())
        while True:
            evt = await queue.get()
            if evt is None:
                break
            yield evt
        await task
