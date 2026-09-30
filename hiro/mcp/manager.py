"""
MCP Manager - manages multiple MCP server connections.
Provides a unified interface for all connected servers.
"""
from __future__ import annotations

import asyncio
from typing import Any

from hiro.config import MCPServerConfig
from .client import (
    StdioMCPClient, SSEMCPClient, MCPTool, MCPResource,
    MCPPrompt, MCPCallResult, MCPTransport
)


class MCPManager:
    """Manages multiple MCP server connections."""

    def __init__(self) -> None:
        self._clients: dict[str, StdioMCPClient | SSEMCPClient] = {}
        self._tools: dict[str, MCPTool] = {}  # tool_name -> tool
        self._resources: dict[str, MCPResource] = {}
        self._prompts: dict[str, MCPPrompt] = {}
        self._tool_server_map: dict[str, str] = {}  # tool_name -> server_name
        self._connected: set[str] = set()
        self._errors: dict[str, str] = {}

    def _remove_server_data(self, name: str) -> None:
        """Remove all tools, resources, and prompts registered by a server."""
        for tool_name in list(self._tool_server_map.keys()):
            if self._tool_server_map[tool_name] == name:
                del self._tool_server_map[tool_name]
                self._tools.pop(tool_name, None)
        for uri in list(self._resources.keys()):
            if self._resources[uri].server_name == name:
                del self._resources[uri]
        for pname in list(self._prompts.keys()):
            if self._prompts[pname].server_name == name:
                del self._prompts[pname]

    async def connect_server(self, config: MCPServerConfig) -> bool:
        """Connect to an MCP server. Returns True on success."""
        if not config.enabled:
            return False

        name = config.name
        self._remove_server_data(name)
        try:
            if config.transport == MCPTransport.STDIO or not config.url:
                cmd = config.command
                args = config.args
                client = StdioMCPClient(
                    command=cmd,
                    args=args,
                    env=config.env,
                    name=name,
                )
            else:
                client = SSEMCPClient(url=config.url, name=name)

            await client.connect()
            self._clients[name] = client
            self._connected.add(name)

            # Discover capabilities
            await self._discover(name, client)
            return True

        except Exception as e:
            self._errors[name] = str(e)
            return False

    async def _discover(self, name: str, client: Any) -> None:
        """Discover tools, resources, and prompts from a connected server."""
        # Tools
        try:
            tools = await client.list_tools()
            for tool in tools:
                tool.server_name = name
                self._tools[tool.name] = tool
                self._tool_server_map[tool.name] = name
                self._tool_server_map[f"{name}__{tool.name}"] = name
        except Exception:
            pass

        # Resources
        try:
            resources = await client.list_resources()
            for res in resources:
                self._resources[res.uri] = res
        except Exception:
            pass

        # Prompts
        try:
            prompts = await client.list_prompts()
            for prompt in prompts:
                self._prompts[prompt.name] = prompt
        except Exception:
            pass

    async def call_tool(self, tool_name: str, arguments: dict) -> MCPCallResult:
        """Call a tool on the appropriate server."""
        server_name = self._tool_server_map.get(tool_name)
        if not server_name:
            return MCPCallResult(
                content=[{"type": "text", "text": f"Unknown tool: {tool_name}"}],
                is_error=True,
            )

        client = self._clients.get(server_name)
        if not client:
            return MCPCallResult(
                content=[{"type": "text", "text": f"Server not connected: {server_name}"}],
                is_error=True,
            )

        # Resolve actual tool name (strip server prefix if present)
        actual_name = tool_name
        if tool_name.startswith(f"{server_name}__"):
            actual_name = tool_name[len(server_name) + 2:]

        return await client.call_tool(actual_name, arguments)

    async def read_resource(self, uri: str) -> MCPCallResult:
        """Read a resource from the appropriate server."""
        res = self._resources.get(uri)
        if not res:
            return MCPCallResult(
                content=[{"type": "text", "text": f"Unknown resource: {uri}"}],
                is_error=True,
            )

        client = self._clients.get(res.server_name)
        if not client:
            return MCPCallResult(
                content=[{"type": "text", "text": f"Server not connected"}],
                is_error=True,
            )

        return await client.read_resource(uri)

    async def get_prompt(self, name: str, arguments: dict = None) -> str:
        """Get a prompt from the appropriate server."""
        prompt = self._prompts.get(name)
        if not prompt:
            return f"Unknown prompt: {name}"

        client = self._clients.get(prompt.server_name)
        if not client:
            return f"Server not connected"

        return await client.get_prompt(name, arguments)

    def get_all_tools(self) -> list[MCPTool]:
        return list(self._tools.values())

    def get_tools_for_server(self, server_name: str) -> list[MCPTool]:
        return [t for t in self._tools.values() if t.server_name == server_name]

    def get_all_resources(self) -> list[MCPResource]:
        return list(self._resources.values())

    def get_all_prompts(self) -> list[MCPPrompt]:
        return list(self._prompts.values())

    def connected_servers(self) -> list[str]:
        return list(self._connected)

    def failed_servers(self) -> dict[str, str]:
        return dict(self._errors)

    def is_connected(self, name: str) -> bool:
        return name in self._connected

    async def disconnect_all(self) -> None:
        """Disconnect all servers."""
        for client in self._clients.values():
            try:
                await client.disconnect()
            except Exception:
                pass
        self._clients.clear()
        self._connected.clear()

    async def reconnect(self, name: str, config: MCPServerConfig) -> bool:
        """Reconnect to a specific server."""
        if name in self._clients:
            try:
                await self._clients[name].disconnect()
            except Exception:
                pass
            del self._clients[name]
            self._connected.discard(name)
            # Remove tools from this server
            for tool_name in list(self._tool_server_map.keys()):
                if self._tool_server_map[tool_name] == name:
                    del self._tool_server_map[tool_name]
                    self._tools.pop(tool_name, None)

        return await self.connect_server(config)

    def stats(self) -> dict:
        return {
            "connected": len(self._connected),
            "tools": len(self._tools),
            "resources": len(self._resources),
            "prompts": len(self._prompts),
            "errors": len(self._errors),
        }
