"""Tests for Hiro model providers and conversion logic."""
import pytest
from hiro.providers import get_provider, clear_cache
from hiro.providers.base import Message, ToolCall, ToolDefinition, Usage
from hiro.providers.openai import _convert_messages as openai_convert, _convert_tools as openai_tools
from hiro.providers.anthropic import _convert_messages as anthropic_convert, _convert_tools as anthropic_tools


def test_provider_registry():
    clear_cache()
    p_anthropic = get_provider("anthropic", api_key="test-key")
    assert p_anthropic.name == "anthropic"

    p_openai = get_provider("openai", api_key="test-key")
    assert p_openai.name == "openai"

    p_groq = get_provider("groq", api_key="test-key")
    assert p_groq.name == "groq"

    p_deepseek = get_provider("deepseek", api_key="test-key")
    assert p_deepseek.name == "deepseek"

    p_ollama = get_provider("ollama", base_url="http://localhost:11434/v1")
    assert p_ollama.name == "ollama"


def test_openai_message_and_tool_conversion():
    msgs = [
        Message(role="user", content="Hello"),
        Message(
            role="assistant",
            content="Checking files",
            tool_calls=[ToolCall(id="call_1", name="read_file", arguments={"path": "test.py"})],
        ),
        Message(role="tool", content="print('hello')", tool_call_id="call_1", name="read_file"),
    ]
    converted = openai_convert(msgs)
    assert len(converted) == 3
    assert converted[0]["role"] == "user"
    assert converted[1]["role"] == "assistant"
    assert "tool_calls" in converted[1]
    assert converted[2]["role"] == "tool"
    assert converted[2]["tool_call_id"] == "call_1"

    tools = [
        ToolDefinition(name="read_file", description="Read file", parameters={"type": "object"}),
    ]
    conv_tools = openai_tools(tools)
    assert len(conv_tools) == 1
    assert conv_tools[0]["type"] == "function"
    assert conv_tools[0]["function"]["name"] == "read_file"


def test_anthropic_message_and_tool_conversion():
    msgs = [
        Message(role="user", content="Hello"),
        Message(
            role="assistant",
            content="Checking files",
            tool_calls=[ToolCall(id="tool_1", name="read_file", arguments={"path": "test.py"})],
        ),
        Message(role="tool", content="content", tool_call_id="tool_1", name="read_file"),
    ]
    converted = anthropic_convert(msgs)
    assert len(converted) == 3
    assert converted[0]["role"] == "user"
    assert converted[1]["role"] == "assistant"
    assert converted[2]["role"] == "user"
    assert converted[2]["content"][0]["type"] == "tool_result"

    tools = [
        ToolDefinition(name="read_file", description="Read file", parameters={"type": "object"}),
    ]
    conv_tools = anthropic_tools(tools)
    assert len(conv_tools) == 1
    assert conv_tools[0]["name"] == "read_file"
    assert "input_schema" in conv_tools[0]
