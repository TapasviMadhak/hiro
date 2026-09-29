"""
Main application loop for Hiro.
Handles the REPL, interactive menus, command parsing, and orchestrating all components.
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
import threading
from pathlib import Path
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.styles import Style
from prompt_toolkit.formatted_text import HTML
from rich.text import Text
from rich.live import Live
from rich.spinner import Spinner
from rich.table import Table
from rich import box
from rich.rule import Rule
from rich.console import Console
from rich.columns import Columns

from hiro.config import (
    Settings, load_settings, save_settings,
    get_api_key, get_model_info, PROVIDER_BASE_URLS,
    CONFIG_DIR, HISTORY_FILE, ensure_dirs,
)
from hiro.providers import get_provider
from hiro.providers.base import BaseProvider
from hiro.mcp import MCPManager
from hiro.agent import AgentLoop, AgentContext, AgentEvent
from hiro.ui import HiroRenderer
from hiro.ui.interactive import (
    show_command_menu, show_model_picker,
    show_api_key_wizard, show_base_url_wizard, show_confirm,
)
from hiro.commands.registry import parse_command, get_command, get_completions, list_commands


# ─── Prompt Completer ────────────────────────────────────────────────────────

class HiroCompleter(Completer):
    """Tab completion for Hiro commands and file paths."""

    def __init__(self, app: "HiroApp") -> None:
        self.app = app

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor

        # Slash command completion
        if text.startswith("/"):
            if " " not in text:
                prefix = text[1:]
                for cmd in list_commands():
                    if cmd.name.startswith(prefix):
                        yield Completion(
                            f"/{cmd.name}",
                            start_position=-len(text),
                            display=f"/{cmd.name}",
                            display_meta=cmd.description[:40],
                        )
                return

            parts = text.split()
            cmd_name = parts[0].lstrip("/")

            if cmd_name == "model" and len(parts) == 2:
                from hiro.config import BUILTIN_MODELS
                prefix = parts[1]
                for model_id in BUILTIN_MODELS:
                    if model_id.startswith(prefix):
                        yield Completion(model_id, start_position=-len(parts[1]))

            elif cmd_name == "theme" and len(parts) == 2:
                from hiro.ui.renderer import THEMES
                prefix = parts[1]
                for theme in THEMES:
                    if theme.startswith(prefix):
                        yield Completion(theme, start_position=-len(parts[1]))

            elif cmd_name in ("file", "read", "cd", "patch") and len(parts) >= 2:
                partial = parts[-1]
                yield from self._complete_path(partial)

        elif "@" in text:
            at_pos = text.rfind("@")
            partial = text[at_pos + 1:]
            for comp in self._complete_path(partial):
                yield comp

    def _complete_path(self, partial: str):
        try:
            if partial.startswith("~"):
                partial = str(Path.home()) + partial[1:]
            p = Path(partial)
            parent = p.parent if not partial.endswith("/") else p
            prefix = p.name if not partial.endswith("/") else ""
            if not parent.exists():
                parent = Path(os.getcwd())
                prefix = partial
            for entry in sorted(parent.iterdir())[:30]:
                name = entry.name
                if name.startswith(prefix):
                    full = str(entry) + ("/" if entry.is_dir() else "")
                    yield Completion(
                        full,
                        start_position=-len(partial),
                        display=name + ("/" if entry.is_dir() else ""),
                    )
        except Exception:
            pass


# ─── Prompt Styles ────────────────────────────────────────────────────────────

PROMPT_STYLE = Style.from_dict({
    "prompt":   "#7C3AED bold",
    "username": "#06B6D4",
    "path":     "#64748B",
})

CHAT_PROMPT_STYLE = Style.from_dict({
    "prompt": "#06B6D4 bold",
    "model":  "#7C3AED",
})


def _get_prompt_tokens(model: str, cwd: str, in_chat: bool = False) -> HTML:
    short_cwd = Path(cwd).name or cwd
    short_model = model.split("/")[-1] if "/" in model else model
    short_model = short_model[:22]
    if in_chat:
        return HTML(
            f'<ansibrightblack>╭─</ansibrightblack>'
            f'<ansicyan> chat </ansicyan>'
            f'<ansibrightblack>·</ansibrightblack>'
            f'<ansimagenta> {short_model} </ansimagenta>'
            f'<ansibrightblack>·</ansibrightblack>'
            f'<ansibrightblack> {short_cwd}\n'
            f'╰─❯ </ansibrightblack>'
        )
    return HTML(
        f'<ansibrightblack>[</ansibrightblack>'
        f'<ansimagenta>{short_model}</ansimagenta>'
        f'<ansibrightblack>]</ansibrightblack> '
        f'<ansibrightblack>{short_cwd}</ansibrightblack>'
        f'<ansiblue> ❯ </ansiblue>'
    )


# ─── Live Processing Display ─────────────────────────────────────────────────

class ProcessingDisplay:
    """
    Claude Code-style live processing indicator.
    Shows a spinner, elapsed time, and current action in real-time.
    """

    def __init__(self, console: Console, model: str, theme: dict) -> None:
        self._console = console
        self._model = model.split("/")[-1][:20]
        self._t = theme
        self._start = time.monotonic()
        self._status = "Thinking"
        self._lock = threading.Lock()
        self._live: Live | None = None

    def _render(self) -> Text:
        elapsed = time.monotonic() - self._start
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)
        time_str = f"{mins}:{secs:02d}" if mins else f"{secs}s"

        txt = Text()
        txt.append("  ◆ ", style=f"bold {self._t['primary']}")
        txt.append(f"{self._model}", style=f"bold {self._t['secondary']}")
        txt.append("  ", style="")
        txt.append(f"{self._status}", style=f"italic {self._t['muted']}")
        txt.append("  ", style="")
        txt.append(f"⏱ {time_str}", style=f"dim {self._t['muted']}")
        return txt

    def update(self, status: str) -> None:
        with self._lock:
            self._status = status

    def __enter__(self) -> "ProcessingDisplay":
        self._live = Live(
            self._render(),
            console=self._console,
            refresh_per_second=4,
            transient=True,
        )
        # Patch refresh to re-render dynamically
        orig_refresh = self._live.refresh

        def _refresh():
            self._live.update(self._render())
            orig_refresh()

        self._live.refresh = _refresh
        self._live.__enter__()
        return self

    def __exit__(self, *args) -> None:
        elapsed = time.monotonic() - self._start
        if self._live:
            self._live.__exit__(*args)
        # Print final timing line
        mins = int(elapsed // 60)
        secs = elapsed % 60
        time_str = f"{mins}m {secs:.1f}s" if mins else f"{secs:.1f}s"
        self._console.print(
            Text(f"  ◆ {self._model}  completed in {time_str}", style=f"dim {self._t['muted']}")
        )


# ─── Chat Header ─────────────────────────────────────────────────────────────

def print_chat_header(console: Console, model: str, provider: str, theme: dict) -> None:
    """Print a distinct header when entering chat/query mode."""
    short_model = model.split("/")[-1][:30] if "/" in model else model[:30]
    t = theme
    console.print()
    console.print(Rule(style=f"dim {t['border']}"))
    txt = Text()
    txt.append("  ◈ CHAT  ", style=f"bold {t['primary']}")
    txt.append(f"{short_model}", style=f"bold {t['secondary']}")
    txt.append(f"  via {provider}", style=f"dim {t['muted']}")
    console.print(txt)
    console.print(Rule(style=f"dim {t['border']}"))
    console.print()


# ─── Main App ─────────────────────────────────────────────────────────────────

class HiroApp:
    """The main Hiro application."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.running = True
        self._in_chat = False  # tracks whether we've sent at least one message

        # Initialize UI
        self.ui = HiroRenderer(config=settings.ui)

        # Initialize provider
        provider_name = settings.provider
        api_key = get_api_key(provider_name, settings)
        base_url = settings.base_urls.get(provider_name, PROVIDER_BASE_URLS.get(provider_name, ""))

        self.provider: BaseProvider = get_provider(
            provider_name,
            api_key=api_key,
            base_url=base_url,
        )

        # Initialize MCP
        self.mcp: MCPManager = MCPManager()

        # Initialize agent
        self.agent = AgentLoop(
            provider=self.provider,
            model=settings.model,
            settings=settings,
            mcp_manager=self.mcp,
        )

        # Import all commands (side effect: registers them)
        import hiro.commands  # noqa

        self._session_start = time.time()

    async def initialize_mcp(self) -> None:
        """Connect to all enabled MCP servers."""
        if not self.settings.mcp_servers:
            return

        self.ui.print_info("Connecting to MCP servers…")
        tasks = [self.mcp.connect_server(srv) for srv in self.settings.mcp_servers if srv.enabled]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        connected = 0
        failed = 0
        for srv, result in zip(
            [s for s in self.settings.mcp_servers if s.enabled], results
        ):
            if isinstance(result, Exception) or result is False:
                failed += 1
                self.ui.print_warning(f"MCP '{srv.name}': failed to connect")
            else:
                connected += 1

        stats = self.mcp.stats()
        if connected > 0:
            self.ui.print_success(
                f"MCP: {connected} server(s) connected, "
                f"{stats['tools']} tools, {stats['resources']} resources"
            )
        if failed > 0:
            self.ui.print_warning(f"MCP: {failed} server(s) failed")

    async def run_query(self, user_input: str) -> None:
        """Run a user query through the agent with live processing display."""
        t = self.ui._t

        # First query: print chat mode header
        if not self._in_chat:
            print_chat_header(
                self.ui.console,
                self.settings.model,
                self.settings.provider,
                t,
            )
            self._in_chat = True

        t0 = time.monotonic()
        last_usage = None
        tool_times: dict[str, float] = {}
        text_buffer: list[str] = []
        events_pending: list[AgentEvent] = []
        display_started = False

        proc = ProcessingDisplay(self.ui.console, self.settings.model, t)

        def on_event(evt: AgentEvent) -> None:
            nonlocal last_usage, display_started

            if evt.type == "text":
                if not display_started:
                    # First text: update status to "Responding"
                    proc.update("Responding")
                text_buffer.append(evt.data)

            elif evt.type == "tool_start":
                tc = evt.data
                proc.update(f"Tool: {tc.name}")
                tool_times[tc.id] = time.monotonic()

            elif evt.type == "tool_end":
                proc.update("Thinking")
                tc = evt.data["call"]
                result = evt.data["result"]
                elapsed = (time.monotonic() - tool_times.get(tc.id, time.monotonic())) * 1000
                events_pending.append(AgentEvent("tool_end_render", {"call": tc, "result": result, "ms": elapsed}))

            elif evt.type == "error":
                proc.update(f"Error")
                events_pending.append(AgentEvent("error_render", evt.data))

            elif evt.type == "usage":
                last_usage = evt.data

            elif evt.type == "thinking":
                proc.update(evt.data[:40])

        # Run with live processing display
        with proc:
            await self.agent.run(user_input, on_event=on_event)

        # Now render everything (after spinner exits / screen is clear)
        self.ui.console.print()

        # Print assistant label + streamed text
        self.ui.print_assistant_start()
        full_text = "".join(text_buffer)
        if full_text:
            self.ui.print_text_chunk(full_text)
        self.ui.print_newline()

        # Print any tool results / errors that happened
        for evt in events_pending:
            if evt.type == "tool_end_render":
                tc = evt.data["call"]
                self.ui.print_tool_end(tc.name, evt.data["result"], evt.data["ms"])
            elif evt.type == "error_render":
                self.ui.print_error(evt.data)

        elapsed_ms = (time.monotonic() - t0) * 1000
        if last_usage:
            self.ui.print_usage(last_usage, elapsed_ms)
        elif self.settings.ui.show_timing:
            self.ui.console.print(
                Text(f"  {elapsed_ms/1000:.1f}s", style=f"dim {t['muted']}")
            )
        self.ui.console.print()

    async def handle_command(self, text: str) -> None:
        """Handle a slash command."""
        parsed = parse_command(text)
        if not parsed:
            return

        cmd_name, args = parsed
        cmd = get_command(cmd_name)

        if not cmd:
            self.ui.print_error(f"Unknown command: /{cmd_name}. Try /help")
            return

        try:
            await cmd.handler(self, args)
        except Exception as e:
            self.ui.print_error(f"Command error: {e}")

    async def handle_slash_menu(self) -> str | None:
        """
        Show interactive command menu on bare '/'.
        Returns the raw command string to run, or None.
        """
        selected = await show_command_menu()
        if not selected:
            return None

        # Special interactive handlers for model/key/url selection
        if selected == "/model":
            model = await show_model_picker(self.settings.model)
            if model:
                return f"/model {model}"
            return None

        if selected in ("/key", "/apikey"):
            updated_keys = await show_api_key_wizard(self.settings.api_keys)
            if updated_keys:
                self.settings.api_keys = updated_keys
                save_settings(self.settings)
                self.ui.print_success("API keys saved.")
            return None

        # Commands that need sub-args → return them so user can complete inline
        NEEDS_ARGS = {"/cd", "/file", "/read", "/patch", "/mcp add", "/temperature", "/theme"}
        if selected in NEEDS_ARGS:
            return selected + " "  # hand back with trailing space for user to fill

        return selected

    async def run(self) -> None:
        """Main REPL loop."""
        ensure_dirs()

        mcp_count = len(self.mcp.get_all_tools())
        self.ui.print_banner(
            model=self.settings.model,
            provider=self.settings.provider,
            mcp_count=mcp_count,
        )

        history_file = str(HISTORY_FILE)
        session: PromptSession = PromptSession(
            history=FileHistory(history_file),
            auto_suggest=AutoSuggestFromHistory(),
            completer=HiroCompleter(self),
            complete_while_typing=False,
            enable_history_search=True,
            mouse_support=False,
            style=PROMPT_STYLE,
            wrap_lines=True,
        )

        while self.running:
            try:
                cwd = os.getcwd()
                prompt = _get_prompt_tokens(self.settings.model, cwd, in_chat=self._in_chat)

                try:
                    user_input = await session.prompt_async(prompt, multiline=False)
                except (EOFError, KeyboardInterrupt):
                    self.ui.print_info("\nCtrl+D — type /exit to quit.")
                    continue

                if not user_input or not user_input.strip():
                    continue

                text = user_input.strip()

                # Multi-line input with backslash continuation
                if text.endswith("\\"):
                    lines = [text[:-1]]
                    while True:
                        try:
                            cont = await session.prompt_async("... ")
                            if cont.endswith("\\"):
                                lines.append(cont[:-1])
                            else:
                                lines.append(cont)
                                break
                        except (EOFError, KeyboardInterrupt):
                            break
                    text = "\n".join(lines)

                # Bare "/" → show interactive command menu
                if text == "/":
                    cmd_str = await self.handle_slash_menu()
                    if cmd_str:
                        if cmd_str.endswith(" "):
                            # Pre-fill prompt with command so user can append args
                            try:
                                pre = await session.prompt_async(
                                    _get_prompt_tokens(self.settings.model, cwd, self._in_chat),
                                    default=cmd_str,
                                )
                                text = pre.strip()
                                if text.startswith("/"):
                                    await self.handle_command(text)
                            except (EOFError, KeyboardInterrupt):
                                pass
                        else:
                            await self.handle_command(cmd_str)
                    continue

                # Slash command
                if text.startswith("/"):
                    await self.handle_command(text)
                    continue

                # Regular query → send to agent
                await self.run_query(text)

            except KeyboardInterrupt:
                self.ui.print_info("\n[dim]Interrupted. Type /exit to quit.[/dim]")
                continue
            except Exception as e:
                self.ui.print_error(f"Unexpected error: {e}")
                import traceback
                traceback.print_exc()

        # Cleanup
        await self.mcp.disconnect_all()
