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
            scratch_dir=settings.scratch_dir,
        )

    def _get_all_tools(self) -> list[ToolDefinition]:
        """Collect all available tools: built-in + MCP (deduplicated by name)."""
        tools_dict: dict[str, ToolDefinition] = {}

        # 1. Built-in tools
        for tool in get_builtin_tools(allow_shell=self.settings.allow_shell):
            tools_dict[tool.name] = tool

        # 2. MCP tools
        if self.mcp:
            for mcp_tool in self.mcp.get_all_tools():
                tname = mcp_tool.name
                if tname in tools_dict and mcp_tool.server_name:
                    # Avoid collision with built-in
                    tname = f"{mcp_tool.server_name}__{tname}"
                tools_dict[tname] = _mcp_tool_to_definition(mcp_tool, tname)

        return list(tools_dict.values())

    async def _execute_tool(self, tool_call: ToolCall) -> str:
        """Execute a tool call, routing to built-in or MCP."""
        name = tool_call.name
        args = tool_call.arguments

        # Validate required parameters before dispatching
        for tdef in self._get_all_tools():
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

        # Per-turn state
        # Maps "tool_name:frozen_args" -> error count across iterations this turn
        _tool_error_counts: dict[str, int] = {}
        # Tracks (name, frozen_args) combos already executed this turn to dedup across iterations
        _executed_sigs: set[str] = set()

        all_tools = self._get_all_tools()
        system_prompt = system or self.settings.system_prompt
        max_turns = self.settings.agent.max_turns
        final_text = ""

        for iteration in range(max_turns):
            emit(AgentEvent("step", {"iteration": iteration + 1, "max_turns": max_turns}))

            # Check token budget
            token_est = self.ctx.token_estimate()
            budget = self.settings.agent.token_budget
            if token_est > budget * self.settings.agent.compact_threshold and self.settings.agent.auto_compact:
                msg = self.ctx.compact()
                emit(AgentEvent("thinking", f"Auto-compacted context: {msg}"))

            # Steering nudge if approaching turn limit to prevent runaway micro-tools
            active_system = system_prompt
            if iteration >= 8:
                steering = (
                    "\n[Notice: You have executed multiple tools. If you have gathered sufficient findings, "
                    "synthesize your observations into a comprehensive, structured report and present your final answer now.]"
                )
                active_system = (system_prompt + steering) if system_prompt else steering.strip()

            # Stream response
            text_parts: list[str] = []
            tool_calls: list[ToolCall] = []
            finish_reason = ""
            turn_usage = Usage()

            try:
                async for chunk in self.provider.stream(
                    messages=self.ctx.messages,
                    model=self.model,
                    system=active_system,
                    tools=all_tools if all_tools else None,
                    max_tokens=self.settings.agent.max_tokens_per_turn,
                    temperature=self.settings.temperature,
                ):
                    if chunk.text:
                        text_parts.append(chunk.text)
                        emit(AgentEvent("text", chunk.text))

                    if chunk.tool_call:
                        # Deduplicate within this streaming batch by ID
                        if not any(tc.id == chunk.tool_call.id for tc in tool_calls):
                            tool_calls.append(chunk.tool_call)

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
                        system=active_system,
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

            # Guard: empty response → nudge once
            if not assistant_text and not tool_calls:
                emit(AgentEvent("thinking", "Empty model response — retrying with nudge..."))
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

            # ── Deduplicate tool calls by (name, args) fingerprint ──────────
            def _sig(tc: ToolCall) -> str:
                try:
                    frozen = json.dumps(tc.arguments, sort_keys=True)
                except Exception:
                    frozen = str(tc.arguments)
                return f"{tc.name}:{frozen}"

            # Within this batch: dedupe by (name, args), not just ID
            seen_sigs_this_batch: set[str] = set()
            unique_calls: list[ToolCall] = []
            for tc in tool_calls:
                sig = _sig(tc)
                if sig not in seen_sigs_this_batch:
                    seen_sigs_this_batch.add(sig)
                    unique_calls.append(tc)

            # Emit tool_start only for unique, unblocked calls
            results_to_add: list[tuple[ToolCall, str]] = []
            calls_to_run: list[ToolCall] = []

            for tc in unique_calls:
                sig = _sig(tc)
                err_count = _tool_error_counts.get(sig, 0)

                emit(AgentEvent("tool_start", tc))

                if err_count >= 2:
                    # Block: this exact call already failed twice this turn
                    blocked_msg = (
                        f"Error: tool '{tc.name}' with these arguments has failed {err_count} times. "
                        f"Please try a different approach or different arguments."
                    )
                    emit(AgentEvent("tool_end", {"call": tc, "result": blocked_msg}))
                    results_to_add.append((tc, blocked_msg))
                else:
                    calls_to_run.append(tc)

            # Execute non-blocked calls
            if calls_to_run:
                if self.settings.agent.parallel_tools and len(calls_to_run) > 1:
                    raw_results = await asyncio.gather(
                        *[self._execute_tool(tc) for tc in calls_to_run],
                        return_exceptions=True,
                    )
                    for tc, result in zip(calls_to_run, raw_results):
                        if isinstance(result, Exception):
                            result = f"Error: {result}"
                        result_str = str(result)
                        sig = _sig(tc)
                        if result_str.startswith("Error:"):
                            _tool_error_counts[sig] = _tool_error_counts.get(sig, 0) + 1
                        else:
                            _tool_error_counts.pop(sig, None)  # reset on success
                        emit(AgentEvent("tool_end", {"call": tc, "result": result_str}))
                        results_to_add.append((tc, result_str))
                else:
                    for tc in calls_to_run:
                        result = await self._execute_tool(tc)
                        sig = _sig(tc)
                        if result.startswith("Error:"):
                            _tool_error_counts[sig] = _tool_error_counts.get(sig, 0) + 1
                        else:
                            _tool_error_counts.pop(sig, None)
                        emit(AgentEvent("tool_end", {"call": tc, "result": result}))
                        results_to_add.append((tc, result))

            # Commit all results to context
            for tc, result_str in results_to_add:
                self.ctx.add_tool_result(tc.id, tc.name, result_str)

            # If every scheduled tool is now blocked (all at error limit), stop
            all_blocked = all(
                _tool_error_counts.get(_sig(tc), 0) >= 2 for tc in unique_calls
            )
            if all_blocked and unique_calls:
                msg = (
                    "Stopping: all tool calls have repeatedly failed. "
                    "Please try rephrasing your request or use a different approach."
                )
                emit(AgentEvent("error", msg))
                break

        else:
            emit(AgentEvent("error", f"Reached turn limit ({max_turns} turns). Generating final synthesis..."))

        # Guarantee a comprehensive final report if loop ended without one
        if not final_text:
            try:
                emit(AgentEvent("thinking", "Synthesizing all findings into final report..."))
                summary_resp = await self.provider.complete(
                    messages=self.ctx.messages + [Message(
                        role="user",
                        content="Analyze all tool outputs, tests, and observations gathered above. Provide a thorough, structured report with all your findings, vulnerability assessments, and actionable recommendations."
                    )],
                    model=self.model,
                    system=system_prompt,
                    tools=None,
                    max_tokens=self.settings.agent.max_tokens_per_turn,
                    temperature=self.settings.temperature,
                )
                if summary_resp.content:
                    final_text = summary_resp.content
                    self.ctx.add_assistant(summary_resp.content)
                    emit(AgentEvent("final_response", final_text))
            except Exception as e:
                emit(AgentEvent("error", f"Synthesis error: {e}"))

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
