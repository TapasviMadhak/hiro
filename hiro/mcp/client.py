"""
MCP (Model Context Protocol) client implementation.
Supports stdio, SSE, and HTTP transports.
Fully async, based on the official MCP spec.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
import time
import subprocess
from dataclasses import dataclass, field
from typing import Any, AsyncIterator
from enum import Enum


class MCPTransport(str, Enum):
    STDIO = "stdio"
    SSE = "sse"
    HTTP = "http"


@dataclass
class MCPTool:
    name: str
    description: str
    input_schema: dict[str, Any]
    server_name: str = ""


@dataclass
class MCPResource:
    uri: str
    name: str
    description: str = ""
    mime_type: str = ""
    server_name: str = ""


@dataclass
class MCPPrompt:
    name: str
    description: str = ""
    arguments: list[dict] = field(default_factory=list)
    server_name: str = ""


@dataclass
class MCPCallResult:
    content: list[dict[str, Any]]
    is_error: bool = False

    def to_text(self) -> str:
        parts = []
        for item in self.content:
            if item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif item.get("type") == "image":
                parts.append(f"[Image: {item.get('mimeType', 'unknown')}]")
            elif item.get("type") == "resource":
                parts.append(f"[Resource: {item.get('uri', '')}]")
            else:
                parts.append(json.dumps(item))
        return "\n".join(parts)


class MCPError(Exception):
    def __init__(self, code: int, message: str, data: Any = None):
        super().__init__(message)
        self.code = code
        self.data = data


class StdioMCPClient:
    """MCP client over stdio transport."""

    def __init__(
        self,
        command: list[str],
        args: list[str] = None,
        env: dict[str, str] = None,
        name: str = "",
    ) -> None:
        self.command = command
        self.args = args or []
        self.env = env or {}
        self.name = name
        self._process: asyncio.subprocess.Process | None = None
        self._pending: dict[str | int, asyncio.Future] = {}
        self._reader_task: asyncio.Task | None = None
        self._next_id = 1
        self._initialized = False
        self._lock = asyncio.Lock()

    async def connect(self) -> dict:
        """Start the MCP server process and initialize."""
        full_env = {**os.environ, **self.env}
        cmd = self.command + self.args

        self._process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=full_env,
        )

        self._reader_task = asyncio.create_task(self._read_loop())
        result = await self._initialize()
        self._initialized = True
        return result

    async def _initialize(self) -> dict:
        """Send MCP initialize request."""
        result = await self._request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "roots": {"listChanged": False},
                "sampling": {},
            },
            "clientInfo": {
                "name": "hiro",
                "version": "0.1.0",
            },
        })
        # Send initialized notification
        await self._notify("notifications/initialized", {})
        return result

    async def _read_loop(self) -> None:
        """Continuously read JSON-RPC messages from stdout."""
        assert self._process and self._process.stdout
        while True:
            try:
                line = await self._process.stdout.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").strip()
                if not text:
                    continue
                try:
                    msg = json.loads(text)
                except json.JSONDecodeError:
                    continue

                msg_id = msg.get("id")
                if msg_id is not None and msg_id in self._pending:
                    fut = self._pending.pop(msg_id)
                    if "error" in msg:
                        err = msg["error"]
                        fut.set_exception(MCPError(
                            err.get("code", -1),
                            err.get("message", "Unknown error"),
                            err.get("data"),
                        ))
                    else:
                        fut.set_result(msg.get("result", {}))
            except asyncio.CancelledError:
                break
            except Exception:
                continue

    async def _send(self, payload: dict) -> None:
        """Send a JSON-RPC message to the server."""
        assert self._process and self._process.stdin
        data = json.dumps(payload) + "\n"
        self._process.stdin.write(data.encode())
        await self._process.stdin.drain()

    async def _request(self, method: str, params: dict = None) -> dict:
        """Send a JSON-RPC request and wait for response."""
        async with self._lock:
            req_id = self._next_id
            self._next_id += 1

        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[req_id] = fut

        await self._send({
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params or {},
        })

        return await asyncio.wait_for(fut, timeout=30.0)

    async def _notify(self, method: str, params: dict = None) -> None:
        """Send a JSON-RPC notification (no response expected)."""
        await self._send({
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
        })

    async def list_tools(self) -> list[MCPTool]:
        result = await self._request("tools/list", {})
        tools = []
        for t in result.get("tools", []):
            tools.append(MCPTool(
                name=t["name"],
                description=t.get("description", ""),
                input_schema=t.get("inputSchema", {}),
                server_name=self.name,
            ))
        return tools

    async def call_tool(self, name: str, arguments: dict) -> MCPCallResult:
        result = await self._request("tools/call", {
            "name": name,
            "arguments": arguments,
        })
        return MCPCallResult(
            content=result.get("content", []),
            is_error=result.get("isError", False),
        )

    async def list_resources(self) -> list[MCPResource]:
        try:
            result = await self._request("resources/list", {})
        except MCPError:
            return []
        resources = []
        for r in result.get("resources", []):
            resources.append(MCPResource(
                uri=r["uri"],
                name=r.get("name", r["uri"]),
                description=r.get("description", ""),
                mime_type=r.get("mimeType", ""),
                server_name=self.name,
            ))
        return resources

    async def read_resource(self, uri: str) -> MCPCallResult:
        result = await self._request("resources/read", {"uri": uri})
        return MCPCallResult(content=result.get("contents", []))

    async def list_prompts(self) -> list[MCPPrompt]:
        try:
            result = await self._request("prompts/list", {})
        except MCPError:
            return []
        prompts = []
        for p in result.get("prompts", []):
            prompts.append(MCPPrompt(
                name=p["name"],
                description=p.get("description", ""),
                arguments=p.get("arguments", []),
                server_name=self.name,
            ))
        return prompts

    async def get_prompt(self, name: str, arguments: dict = None) -> str:
        result = await self._request("prompts/get", {
            "name": name,
            "arguments": arguments or {},
        })
        messages = result.get("messages", [])
        parts = []
        for msg in messages:
            content = msg.get("content", {})
            if isinstance(content, dict) and content.get("type") == "text":
                parts.append(content.get("text", ""))
            elif isinstance(content, str):
                parts.append(content)
        return "\n".join(parts)

    async def disconnect(self) -> None:
        if self._reader_task:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass
        if self._process:
            try:
                self._process.stdin.close()
                await self._process.wait()
            except Exception:
                pass


class SSEMCPClient:
    """MCP client over Server-Sent Events (SSE) transport."""

    def __init__(self, url: str, name: str = "", headers: dict = None) -> None:
        self.url = url
        self.name = name
        self.headers = headers or {}
        self._session_url: str = ""
        self._pending: dict[str | int, asyncio.Future] = {}
        self._next_id = 1
        self._lock = asyncio.Lock()
        self._reader_task: asyncio.Task | None = None
        self._http_client = None
        self._endpoint_ready = asyncio.Event()

    async def connect(self) -> dict:
        import httpx
        self._http_client = httpx.AsyncClient(headers=self.headers, timeout=60.0)
        self._endpoint_ready.clear()
        self._reader_task = asyncio.create_task(self._sse_loop())
        # Wait for endpoint event from SSE stream
        try:
            await asyncio.wait_for(self._endpoint_ready.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            pass
        result = await self._initialize()
        return result

    async def _sse_loop(self) -> None:
        import urllib.parse
        assert self._http_client
        current_event = None
        try:
            async with self._http_client.stream("GET", self.url, headers={
                "Accept": "text/event-stream",
                **self.headers,
            }) as resp:
                async for raw_line in resp.aiter_lines():
                    line = raw_line.strip()
                    if not line:
                        current_event = None
                        continue
                    if line.startswith("event:"):
                        current_event = line.split(":", 1)[1].strip()
                    elif line.startswith("data:"):
                        data = line.split(":", 1)[1].strip()
                        if not data or data == "[DONE]":
                            continue
                        if current_event == "endpoint":
                            self._session_url = urllib.parse.urljoin(self.url, data)
                            self._endpoint_ready.set()
                            continue

                        try:
                            msg = json.loads(data)
                        except json.JSONDecodeError:
                            if data.startswith("http") or data.startswith("?") or data.startswith("/"):
                                self._session_url = urllib.parse.urljoin(self.url, data)
                                self._endpoint_ready.set()
                            continue

                        msg_id = msg.get("id")
                        if msg_id is not None and msg_id in self._pending:
                            fut = self._pending.pop(msg_id)
                            if "error" in msg:
                                err = msg["error"]
                                fut.set_exception(MCPError(
                                    err.get("code", -1),
                                    err.get("message", "Unknown"),
                                ))
                            else:
                                fut.set_result(msg.get("result", {}))
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

    async def _initialize(self) -> dict:
        result = await self._request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {"sampling": {}},
            "clientInfo": {"name": "hiro", "version": "0.1.0"},
        })
        await self._notify("notifications/initialized", {})
        return result

    async def _request(self, method: str, params: dict = None) -> dict:
        async with self._lock:
            req_id = self._next_id
            self._next_id += 1

        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[req_id] = fut

        target_url = self._session_url or (self.url.rstrip("/") + "/message")
        assert self._http_client
        await self._http_client.post(target_url, json={
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params or {},
        })

        return await asyncio.wait_for(fut, timeout=30.0)

    async def _notify(self, method: str, params: dict = None) -> None:
        target_url = self._session_url or (self.url.rstrip("/") + "/message")
        assert self._http_client
        await self._http_client.post(target_url, json={
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
        })

    async def list_tools(self) -> list[MCPTool]:
        result = await self._request("tools/list", {})
        return [
            MCPTool(
                name=t["name"],
                description=t.get("description", ""),
                input_schema=t.get("inputSchema", {}),
                server_name=self.name,
            )
            for t in result.get("tools", [])
        ]

    async def call_tool(self, name: str, arguments: dict) -> MCPCallResult:
        result = await self._request("tools/call", {"name": name, "arguments": arguments})
        return MCPCallResult(
            content=result.get("content", []),
            is_error=result.get("isError", False),
        )

    async def list_resources(self) -> list[MCPResource]:
        try:
            result = await self._request("resources/list", {})
            return [
                MCPResource(
                    uri=r["uri"],
                    name=r.get("name", r["uri"]),
                    description=r.get("description", ""),
                    server_name=self.name,
                )
                for r in result.get("resources", [])
            ]
        except Exception:
            return []

    async def read_resource(self, uri: str) -> MCPCallResult:
        result = await self._request("resources/read", {"uri": uri})
        return MCPCallResult(content=result.get("contents", []))

    async def list_prompts(self) -> list[MCPPrompt]:
        try:
            result = await self._request("prompts/list", {})
            return [
                MCPPrompt(
                    name=p["name"],
                    description=p.get("description", ""),
                    server_name=self.name,
                )
                for p in result.get("prompts", [])
            ]
        except Exception:
            return []

    async def get_prompt(self, name: str, arguments: dict = None) -> str:
        result = await self._request("prompts/get", {
            "name": name,
            "arguments": arguments or {},
        })
        messages = result.get("messages", [])
        parts = []
        for msg in messages:
            content = msg.get("content", {})
            if isinstance(content, dict):
                parts.append(content.get("text", ""))
            elif isinstance(content, str):
                parts.append(content)
        return "\n".join(parts)

    async def disconnect(self) -> None:
        if self._reader_task:
            self._reader_task.cancel()
        if self._http_client:
            await self._http_client.aclose()
