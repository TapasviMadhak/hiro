"""Tests for Hiro Agent Context and Loop logic."""
from hiro.agent.loop import AgentContext, AgentEvent
from hiro.providers.base import Message, ToolCall, Usage


def test_agent_context_basic():
    ctx = AgentContext()
    ctx.add_user("List files in directory")
    assert len(ctx.messages) == 1
    assert ctx.messages[0].role == "user"

    ctx.add_assistant(
        "I will list the files.",
        tool_calls=[ToolCall(id="tc1", name="list_dir", arguments={"path": "."})],
    )
    assert len(ctx.messages) == 2
    assert ctx.messages[1].role == "assistant"
    assert len(ctx.messages[1].tool_calls) == 1

    ctx.add_tool_result("tc1", "list_dir", "file1.py\nfile2.py")
    assert len(ctx.messages) == 3
    assert ctx.messages[2].role == "tool"
    assert ctx.messages[2].name == "list_dir"


def test_token_efficiency_and_compaction():
    ctx = AgentContext()
    # Add many messages
    for i in range(25):
        ctx.add_user(f"Message number {i}")
        ctx.add_assistant(f"Response number {i}")

    assert len(ctx.messages) == 50
    initial_tokens = ctx.token_estimate()
    assert initial_tokens > 0

    # Compact keeping recent 10 messages
    msg = ctx.compact(keep_recent=10)
    assert "Compacted" in msg
    assert len(ctx.messages) == 10
    compacted_tokens = ctx.token_estimate()
    assert compacted_tokens < initial_tokens
