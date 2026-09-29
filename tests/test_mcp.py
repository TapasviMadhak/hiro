"""Tests for Model Context Protocol (MCP) client and manager."""
import pytest
from hiro.mcp import (
    MCPManager,
    MCPTool,
    MCPResource,
    MCPPrompt,
    MCPCallResult,
    StdioMCPClient,
    SSEMCPClient,
)
from hiro.config import MCPServerConfig


def test_mcp_data_structures():
    tool = MCPTool(
        name="burp_scan",
        description="Run BurpSuite scan",
        input_schema={"type": "object", "properties": {"url": {"type": "string"}}},
        server_name="burp",
    )
    assert tool.name == "burp_scan"
    assert tool.server_name == "burp"

    res = MCPCallResult(
        content=[{"type": "text", "text": "Scan started ID: 101"}],
        is_error=False,
    )
    assert "Scan started" in res.to_text()
    assert not res.is_error


def test_mcp_manager_empty():
    manager = MCPManager()
    assert manager.get_all_tools() == []
    assert manager.get_all_resources() == []
    assert manager.get_all_prompts() == []
    stats = manager.stats()
    assert stats["connected"] == 0
    assert stats["tools"] == 0


@pytest.mark.asyncio
async def test_mcp_manager_call_unknown_tool():
    manager = MCPManager()
    res = await manager.call_tool("nonexistent_tool", {})
    assert res.is_error
    assert "Unknown tool" in res.to_text()
