"""
Built-in tools for Hiro.
These are always available without MCP.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import shutil
import subprocess
from pathlib import Path
from typing import Any

from hiro.providers.base import ToolDefinition


# ─── Tool Registry ────────────────────────────────────────────────────────────

_BUILTIN_TOOLS: list[ToolDefinition] = []


def register_tool(tool: ToolDefinition) -> ToolDefinition:
    _BUILTIN_TOOLS.append(tool)
    return tool


def get_builtin_tools(allow_shell: bool = True) -> list[ToolDefinition]:
    if allow_shell:
        return list(_BUILTIN_TOOLS)
    return [t for t in _BUILTIN_TOOLS if t.name not in ("bash", "shell")]


# ─── File Tools ───────────────────────────────────────────────────────────────

READ_FILE = register_tool(ToolDefinition(
    name="read_file",
    description=(
        "Read the contents of a file. Returns the file content as text. "
        "Use offset and limit to read specific sections of large files."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Absolute or relative path to the file"},
            "offset": {"type": "integer", "description": "Line number to start reading from (1-indexed)", "default": 1},
            "limit": {"type": "integer", "description": "Maximum number of lines to read", "default": 2000},
        },
        "required": ["path"],
    },
))

WRITE_FILE = register_tool(ToolDefinition(
    name="write_file",
    description="Write content to a file. Creates parent directories if needed.",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to write to"},
            "content": {"type": "string", "description": "Content to write"},
            "append": {"type": "boolean", "description": "Append instead of overwrite", "default": False},
        },
        "required": ["path", "content"],
    },
))

EDIT_FILE = register_tool(ToolDefinition(
    name="edit_file",
    description=(
        "Make precise edits to a file by replacing specific text. "
        "Use for targeted code changes without rewriting the entire file."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to edit"},
            "old_string": {"type": "string", "description": "Exact text to find and replace"},
            "new_string": {"type": "string", "description": "Replacement text"},
        },
        "required": ["path", "old_string", "new_string"],
    },
))

LIST_DIR = register_tool(ToolDefinition(
    name="list_dir",
    description="List files and directories in a path. Shows file sizes and types.",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory path to list", "default": "."},
            "recursive": {"type": "boolean", "description": "List recursively", "default": False},
            "show_hidden": {"type": "boolean", "description": "Show hidden files", "default": False},
        },
        "required": [],
    },
))

SEARCH_FILES = register_tool(ToolDefinition(
    name="search_files",
    description=(
        "Search for text patterns in files using ripgrep. "
        "Supports regex. Essential for code exploration."
    ),
    parameters={
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Search pattern (regex supported)"},
            "path": {"type": "string", "description": "Directory to search in", "default": "."},
            "file_pattern": {"type": "string", "description": "File glob pattern, e.g. '*.py'"},
            "case_sensitive": {"type": "boolean", "default": False},
            "max_results": {"type": "integer", "default": 50},
        },
        "required": ["pattern"],
    },
))

MOVE_FILE = register_tool(ToolDefinition(
    name="move_file",
    description="Move or rename a file or directory.",
    parameters={
        "type": "object",
        "properties": {
            "source": {"type": "string"},
            "destination": {"type": "string"},
        },
        "required": ["source", "destination"],
    },
))

DELETE_FILE = register_tool(ToolDefinition(
    name="delete_file",
    description="Delete a file or empty directory.",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "recursive": {"type": "boolean", "default": False},
        },
        "required": ["path"],
    },
))

# ─── Shell Tools ──────────────────────────────────────────────────────────────

BASH = register_tool(ToolDefinition(
    name="bash",
    description=(
        "Execute a shell command and return output. "
        "Use for running tests, builds, git operations, and system commands. "
        "Commands run in the current working directory. "
        "IMPORTANT: Store temporary test dumps, downloads, and curl outputs in the temporary scratch directory "
        "($HIRO_SCRATCH_DIR or %HIRO_SCRATCH_DIR%), NOT in the project root."
    ),
    parameters={
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command to execute"},
            "timeout": {"type": "integer", "description": "Timeout in seconds", "default": 120},
            "cwd": {"type": "string", "description": "Working directory override"},
        },
        "required": ["command"],
    },
))

# ─── Web Tools ────────────────────────────────────────────────────────────────

WEB_FETCH = register_tool(ToolDefinition(
    name="web_fetch",
    description="Fetch content from a URL. Returns the response body as text.",
    parameters={
        "type": "object",
        "properties": {
            "url": {"type": "string"},
            "method": {"type": "string", "default": "GET", "enum": ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD"]},
            "headers": {"type": "object", "additionalProperties": {"type": "string"}},
            "body": {"type": "string", "description": "Request body (for POST/PUT)"},
            "follow_redirects": {"type": "boolean", "default": True},
            "proxy": {"type": "string", "description": "HTTP proxy URL, e.g. http://127.0.0.1:8080 for BurpSuite"},
        },
        "required": ["url"],
    },
))


# ─── Tool Executor ────────────────────────────────────────────────────────────

class BuiltinToolExecutor:
    """Executes built-in tools."""

    def __init__(
        self,
        cwd: str = "",
        allow_shell: bool = True,
        allow_network: bool = True,
        scratch_dir: str = "",
    ) -> None:
        self.cwd = Path(cwd or os.getcwd())
        self.allow_shell = allow_shell
        self.allow_network = allow_network
        self.scratch_dir = Path(scratch_dir) if scratch_dir else (Path.home() / ".config" / "hiro" / "scratch")
        try:
            self.scratch_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    def _resolve(self, path: str) -> Path:
        p = Path(path)
        if not p.is_absolute():
            p = self.cwd / p
        return p

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Execute a built-in tool and return string result."""
        try:
            match tool_name:
                case "read_file":
                    return await self._read_file(**arguments)
                case "write_file":
                    return await self._write_file(**arguments)
                case "edit_file":
                    return await self._edit_file(**arguments)
                case "list_dir":
                    return await self._list_dir(**arguments)
                case "search_files":
                    return await self._search_files(**arguments)
                case "move_file":
                    return await self._move_file(**arguments)
                case "delete_file":
                    return await self._delete_file(**arguments)
                case "bash":
                    if not self.allow_shell:
                        return "Error: Shell execution is disabled"
                    return await self._bash(**arguments)
                case "web_fetch":
                    if not self.allow_network:
                        return "Error: Network access is disabled"
                    return await self._web_fetch(**arguments)
                case _:
                    return f"Error: Unknown built-in tool: {tool_name}"
        except Exception as e:
            return f"Error: {type(e).__name__}: {e}"

    async def _read_file(self, path: str, offset: int = 1, limit: int = 2000, **_) -> str:
        p = self._resolve(path)
        if not p.exists():
            return f"Error: File not found: {path}"
        if p.is_dir():
            return f"Error: {path} is a directory"

        try:
            content = p.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return f"Error reading file: {e}"

        lines = content.splitlines(keepends=True)
        total = len(lines)
        start = max(0, offset - 1)
        end = start + limit
        sliced = lines[start:end]

        header = f"File: {p} | Lines {start+1}-{min(end, total)} of {total}\n"
        if start > 0:
            header += f"(... {start} lines above ...)\n"
        result = header + "".join(sliced)
        if end < total:
            result += f"\n(... {total - end} more lines, use offset={end+1} to continue ...)"
        return result

    async def _write_file(self, path: str, content: str, append: bool = False, **_) -> str:
        p = self._resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        p.write_text(content, encoding="utf-8") if not append else open(p, "a", encoding="utf-8").write(content)
        size = p.stat().st_size
        action = "Appended to" if append else "Written"
        return f"{action} {p} ({size:,} bytes)"

    async def _edit_file(self, path: str, old_string: str, new_string: str, **_) -> str:
        p = self._resolve(path)
        if not p.exists():
            return f"Error: File not found: {path}"

        content = p.read_text(encoding="utf-8", errors="replace")
        if old_string not in content:
            # Try to give a helpful diff-like error
            return f"Error: Pattern not found in {path}. Make sure the string matches exactly including whitespace."

        count = content.count(old_string)
        if count > 1:
            return f"Error: Pattern found {count} times. Please provide more context to make it unique."

        new_content = content.replace(old_string, new_string, 1)
        p.write_text(new_content, encoding="utf-8")

        # Show diff-like summary
        old_lines = old_string.count("\n") + 1
        new_lines = new_string.count("\n") + 1
        return f"Edited {p}: replaced {old_lines} line(s) with {new_lines} line(s)"

    async def _list_dir(self, path: str = ".", recursive: bool = False, show_hidden: bool = False, **_) -> str:
        p = self._resolve(path)
        if not p.exists():
            return f"Error: Path not found: {path}"
        if not p.is_dir():
            return f"Error: {path} is not a directory"

        lines = [f"Directory: {p}\n"]

        def _fmt_entry(entry: Path, indent: str = "") -> str:
            name = entry.name
            if not show_hidden and name.startswith("."):
                return ""
            if entry.is_dir():
                return f"{indent}📁 {name}/"
            else:
                try:
                    size = entry.stat().st_size
                    if size < 1024:
                        size_str = f"{size}B"
                    elif size < 1024 * 1024:
                        size_str = f"{size/1024:.1f}KB"
                    else:
                        size_str = f"{size/1024/1024:.1f}MB"
                except Exception:
                    size_str = "?"
                return f"{indent}📄 {name} ({size_str})"

        if recursive:
            for entry in sorted(p.rglob("*"))[:500]:
                rel = entry.relative_to(p)
                indent = "  " * (len(rel.parts) - 1)
                line = _fmt_entry(entry, indent)
                if line:
                    lines.append(line)
        else:
            for entry in sorted(p.iterdir()):
                line = _fmt_entry(entry)
                if line:
                    lines.append(line)

        return "\n".join(lines)

    async def _search_files(
        self,
        pattern: str,
        path: str = ".",
        file_pattern: str = "",
        case_sensitive: bool = False,
        max_results: int = 50,
        **_,
    ) -> str:
        p = self._resolve(path)

        # Try ripgrep first
        if shutil.which("rg"):
            cmd = ["rg", "--json"]
            if not case_sensitive:
                cmd.append("-i")
            if file_pattern:
                cmd.extend(["-g", file_pattern])
            cmd.extend(["--max-count", str(max_results), pattern, str(p)])

            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
                results = []
                for line in stdout.decode().splitlines():
                    try:
                        obj = json.loads(line)
                        if obj.get("type") == "match":
                            data = obj["data"]
                            file = data["path"]["text"]
                            lineno = data["line_number"]
                            text = data["lines"]["text"].rstrip()
                            results.append(f"{file}:{lineno}: {text}")
                    except Exception:
                        continue
                if not results:
                    return f"No matches found for pattern: {pattern}"
                return "\n".join(results[:max_results])
            except Exception:
                pass

        # Fallback: pure Python grep
        results = []
        flags = 0 if case_sensitive else __import__("re").IGNORECASE
        import re
        try:
            pat = re.compile(pattern, flags)
        except re.error as e:
            return f"Invalid regex: {e}"

        def matches_glob(name: str) -> bool:
            if not file_pattern:
                return True
            from pathlib import PurePath
            return PurePath(name).match(file_pattern)

        for fp in sorted(p.rglob("*"))[:2000]:
            if fp.is_file() and matches_glob(fp.name):
                try:
                    text = fp.read_text(encoding="utf-8", errors="replace")
                    for i, line in enumerate(text.splitlines(), 1):
                        if pat.search(line):
                            rel = fp.relative_to(p)
                            results.append(f"{rel}:{i}: {line.rstrip()}")
                            if len(results) >= max_results:
                                break
                except Exception:
                    continue
            if len(results) >= max_results:
                break

        if not results:
            return f"No matches found for: {pattern}"
        return "\n".join(results)

    async def _move_file(self, source: str, destination: str, **_) -> str:
        src = self._resolve(source)
        dst = self._resolve(destination)
        if not src.exists():
            return f"Error: Source not found: {source}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        return f"Moved {src} → {dst}"

    async def _delete_file(self, path: str, recursive: bool = False, **_) -> str:
        p = self._resolve(path)
        if not p.exists():
            return f"Error: Path not found: {path}"
        if p.is_dir():
            if recursive:
                shutil.rmtree(p)
                return f"Deleted directory: {p}"
            else:
                return f"Error: {path} is a directory. Use recursive=true to delete."
        else:
            p.unlink()
            return f"Deleted: {p}"

    async def _bash(self, command: str, timeout: int = 120, cwd: str = "", **_) -> str:
        work_dir = self._resolve(cwd) if cwd else self.cwd
        env = dict(os.environ)
        env["HIRO_SCRATCH_DIR"] = str(self.scratch_dir)
        env["SCRATCH_DIR"] = str(self.scratch_dir)

        try:
            if sys.platform == "win32":
                proc = await asyncio.create_subprocess_shell(
                    command,
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(work_dir),
                    env=env,
                )
            else:
                proc = await asyncio.create_subprocess_shell(
                    command,
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(work_dir),
                    executable="/bin/bash",
                    env=env,
                )

            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            out = stdout.decode("utf-8", errors="replace")
            err = stderr.decode("utf-8", errors="replace")

            parts = []
            if out:
                parts.append(out.rstrip())
            if err:
                parts.append(f"[stderr]\n{err.rstrip()}")
            if proc.returncode != 0:
                parts.append(f"[exit code: {proc.returncode}]")

            return "\n".join(parts) if parts else "(no output)"

        except asyncio.TimeoutError:
            return f"Error: Command timed out after {timeout}s"
        except (OSError, NotImplementedError):
            # Fallback for Windows invalid handle or event loop limitations
            def _sync_run():
                return subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    cwd=str(work_dir),
                    timeout=timeout,
                    encoding="utf-8",
                    errors="replace",
                    env=env,
                )
            try:
                completed = await asyncio.to_thread(_sync_run)
                parts = []
                if completed.stdout:
                    parts.append(completed.stdout.rstrip())
                if completed.stderr:
                    parts.append(f"[stderr]\n{completed.stderr.rstrip()}")
                if completed.returncode != 0:
                    parts.append(f"[exit code: {completed.returncode}]")
                return "\n".join(parts) if parts else "(no output)"
            except subprocess.TimeoutExpired:
                return f"Error: Command timed out after {timeout}s"
            except Exception as e2:
                return f"Error: {e2}"
        except Exception as e:
            return f"Error: {e}"

    async def _web_fetch(
        self,
        url: str,
        method: str = "GET",
        headers: dict = None,
        body: str = "",
        follow_redirects: bool = True,
        proxy: str = "",
        **_,
    ) -> str:
        try:
            import httpx
            proxy_url = proxy or os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
            client_kwargs: dict[str, Any] = {
                "follow_redirects": follow_redirects,
                "timeout": 30.0,
            }
            if proxy_url:
                client_kwargs["proxy"] = proxy_url
                client_kwargs["verify"] = False  # Common for BurpSuite CA certs

            async with httpx.AsyncClient(**client_kwargs) as client:
                req_kwargs: dict[str, Any] = {
                    "headers": headers or {},
                }
                if body:
                    req_kwargs["content"] = body.encode()

                resp = await client.request(method, url, **req_kwargs)
                ct = resp.headers.get("content-type", "")

                if "json" in ct:
                    try:
                        return json.dumps(resp.json(), indent=2)
                    except Exception:
                        pass

                text = resp.text
                result = f"Status: {resp.status_code}\nURL: {resp.url}\n"
                if len(text) > 10000:
                    result += f"[Truncated to 10000 chars]\n"
                    text = text[:10000]
                return result + text

        except Exception as e:
            return f"Error fetching {url}: {e}"
