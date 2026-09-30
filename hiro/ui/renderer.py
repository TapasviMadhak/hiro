"""
Rich-based UI renderer for Hiro.
Beautiful terminal output with syntax highlighting, panels, and animations.
"""
from __future__ import annotations

import time
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.live import Live
from rich.spinner import Spinner
from rich.rule import Rule
from rich.columns import Columns
from rich import box

from hiro.config import UIConfig


# ─── Themes ──────────────────────────────────────────────────────────────────

THEMES = {
    "dark": {
        "primary": "#7C3AED",      # Violet
        "secondary": "#06B6D4",    # Cyan
        "accent": "#F59E0B",       # Amber
        "success": "#10B981",      # Emerald
        "error": "#EF4444",        # Red
        "warning": "#F59E0B",      # Amber
        "text": "#F1F5F9",         # Slate 100
        "muted": "#64748B",        # Slate 500
        "bg": "#0F172A",           # Slate 900
        "border": "#334155",       # Slate 700
    },
    "light": {
        "primary": "#7C3AED",
        "secondary": "#0891B2",
        "accent": "#D97706",
        "success": "#059669",
        "error": "#DC2626",
        "warning": "#D97706",
        "text": "#0F172A",
        "muted": "#94A3B8",
        "bg": "#F8FAFC",
        "border": "#CBD5E1",
    },
    "hacker": {
        "primary": "#00FF41",      # Matrix green
        "secondary": "#00D4FF",    # Cyan
        "accent": "#FF6B00",       # Orange
        "success": "#00FF41",
        "error": "#FF0000",
        "warning": "#FF6B00",
        "text": "#00FF41",
        "muted": "#007A22",
        "bg": "#000000",
        "border": "#003B00",
    },
    "nord": {
        "primary": "#88C0D0",
        "secondary": "#81A1C1",
        "accent": "#EBCB8B",
        "success": "#A3BE8C",
        "error": "#BF616A",
        "warning": "#EBCB8B",
        "text": "#ECEFF4",
        "muted": "#4C566A",
        "bg": "#2E3440",
        "border": "#3B4252",
    },
}

LOGO = r"""
  ██╗  ██╗██╗██████╗  ██████╗
  ██║  ██║██║██╔══██╗██╔═══██╗
  ███████║██║██████╔╝██║   ██║
  ██╔══██║██║██╔══██╗██║   ██║
  ██║  ██║██║██║  ██║╚██████╔╝
  ╚═╝  ╚═╝╚═╝╚═╝  ╚═╝ ╚═════╝"""


class HiroRenderer:
    """The primary UI renderer."""

    def __init__(self, config: UIConfig | None = None) -> None:
        self.config = config or UIConfig()
        self._theme = THEMES.get(self.config.theme, THEMES["dark"])
        self.console = Console(
            highlight=True,
            markup=True,
            emoji=True,
            force_terminal=True,
            legacy_windows=False,
        )
        self._t = self._theme  # shorthand

    def print_banner(self, model: str, provider: str, mcp_count: int = 0) -> None:
        """Print the startup banner."""
        from rich.align import Align

        primary = self._t["primary"]
        secondary = self._t["secondary"]
        muted = self._t["muted"]

        logo_text = Text(LOGO)
        logo_text.stylize(f"bold {primary}")
        self.console.print(logo_text)
        self.console.print()

        # Status row
        cols = [
            Text(f"⚡ Model: {model}", style=f"bold {secondary}"),
            Text(f"🔌 Provider: {provider}", style=f"{secondary}"),
        ]
        if mcp_count > 0:
            cols.append(Text(f"🛠  MCP: {mcp_count} tools", style=f"bold {self._t['success']}"))

        self.console.print(Columns(cols, expand=False, padding=(0, 3)))
        self.console.print(
            Text("Type /help for commands. Tab-complete. Ctrl+C to interrupt.", style=f"italic {muted}")
        )
        self.console.print(Rule(style=f"dim {self._t['border']}"))
        self.console.print()

    def print_assistant_start(self) -> None:
        """Print the assistant label before a response block."""
        self.console.print(
            Text("● HIRO", style=f"bold {self._t['primary']}"),
        )

    def print_text_chunk(self, text: str) -> None:
        """Print response text (rendered as markdown)."""
        from rich.markdown import Markdown
        if text.strip():
            self.console.print(Markdown(text, code_theme=self.config.syntax_theme))
        else:
            self.console.print(text, end="", markup=False)

    def print_newline(self) -> None:
        self.console.print()

    def print_tool_start(self, name: str, args: dict) -> None:
        """Show a tool being called."""
        args_short = _truncate_dict(args, max_len=80)
        self.console.print(
            f"\n[bold {self._t['accent']}]⚙ Tool:[/] [dim]{name}[/] "
            f"[{self._t['muted']}]{args_short}[/]"
        )

    def print_tool_end(self, name: str, result: str, elapsed_ms: float = 0) -> None:
        """Show tool result."""
        preview = result[:200].replace("\n", "↵ ")
        time_str = f" ({elapsed_ms:.0f}ms)" if elapsed_ms else ""
        self.console.print(
            f"[{self._t['success']}]✓[/] [{self._t['muted']}]{name}{time_str}:[/] "
            f"[dim]{preview}[/]"
        )

    def print_usage(
        self,
        usage: Any,
        latency_ms: float = 0,
        ctx_tokens: int = 0,
        token_limit: int = 0,
        steps: int = 0,
        max_steps: int = 0,
    ) -> None:
        """Print token usage, context limits, usage percentage, duration, and steps."""
        if not self.config.show_token_count:
            return

        parts = []

        # 1. In / Out tokens
        if hasattr(usage, "input_tokens") and hasattr(usage, "output_tokens"):
            in_tok = usage.input_tokens
            out_tok = usage.output_tokens
            in_str = f"{in_tok/1000:.1f}k" if in_tok >= 10000 else f"{in_tok:,}"
            out_str = f"{out_tok/1000:.1f}k" if out_tok >= 10000 else f"{out_tok:,}"
            parts.append(f"↑{in_str}")
            parts.append(f"↓{out_str}")

        # 2. Cache read tokens
        if getattr(usage, "cache_read_tokens", 0):
            c_tok = usage.cache_read_tokens
            c_str = f"{c_tok/1000:.1f}k" if c_tok >= 10000 else f"{c_tok:,}"
            parts.append(f"cache↑{c_str}")

        # 3. Context limit and usage %
        if ctx_tokens > 0 and token_limit > 0:
            pct = (ctx_tokens / token_limit) * 100
            ctx_str = f"{ctx_tokens/1000:.1f}k" if ctx_tokens >= 1000 else str(ctx_tokens)
            limit_str = f"{token_limit/1000:.0f}k" if token_limit >= 1000 else str(token_limit)

            if pct >= 85:
                pct_style = f"bold {self._t['error']}"
            elif pct >= 65:
                pct_style = f"bold {self._t['warning']}"
            else:
                pct_style = f"{self._t['secondary']}"

            parts.append(f"ctx: {ctx_str}/{limit_str} ([{pct_style}]{pct:.1f}%[/])")
        elif ctx_tokens > 0:
            ctx_str = f"{ctx_tokens/1000:.1f}k" if ctx_tokens >= 1000 else str(ctx_tokens)
            parts.append(f"ctx: {ctx_str}")

        # 4. Latency / duration
        if latency_ms:
            sec = latency_ms / 1000
            if sec >= 60:
                mins = int(sec // 60)
                rem_sec = sec % 60
                time_str = f"{mins}m {rem_sec:.1f}s"
            else:
                time_str = f"{sec:.1f}s"
            parts.append(time_str)

        # 5. Steps / iterations
        if steps > 1 or (steps > 0 and max_steps > 0):
            if max_steps > 0:
                parts.append(f"{steps}/{max_steps} steps")
            else:
                parts.append(f"{steps} steps")

        self.console.print(
            f"  [dim {self._t['muted']}]{' · '.join(parts)}[/]"
        )

    def print_error(self, message: str) -> None:
        """Print an error message."""
        self.console.print(
            Panel(
                Text(message, style="white"),
                title="[bold]Error[/]",
                border_style=self._t["error"],
                padding=(0, 1),
            )
        )

    def print_warning(self, message: str) -> None:
        self.console.print(
            f"[bold {self._t['warning']}]⚠[/] [{self._t['warning']}]{message}[/]"
        )

    def print_success(self, message: str) -> None:
        self.console.print(
            f"[bold {self._t['success']}]✓[/] [{self._t['success']}]{message}[/]"
        )

    def print_info(self, message: str) -> None:
        self.console.print(
            f"[bold {self._t['secondary']}]ℹ[/] {message}"
        )

    def print_markdown(self, content: str) -> None:
        """Render markdown content."""
        self.console.print(Markdown(content, code_theme=self.config.syntax_theme))

    def print_code(self, code: str, language: str = "python") -> None:
        """Render syntax-highlighted code."""
        syntax = Syntax(
            code,
            language,
            theme=self.config.syntax_theme,
            line_numbers=True,
            word_wrap=self.config.word_wrap,
        )
        self.console.print(syntax)

    def print_table(self, headers: list[str], rows: list[list[str]], title: str = "") -> None:
        """Render a rich table."""
        table = Table(
            title=title,
            box=box.ROUNDED,
            border_style=self._t["border"],
            header_style=f"bold {self._t['primary']}",
            show_header=True,
        )
        for h in headers:
            table.add_column(h)
        for row in rows:
            table.add_row(*row)
        self.console.print(table)

    def print_panel(self, content: str, title: str = "", style: str = "") -> None:
        """Print a bordered panel."""
        self.console.print(Panel(
            content,
            title=f"[bold]{title}[/]" if title else "",
            border_style=style or self._t["border"],
            padding=(0, 1),
        ))

    def print_rule(self, text: str = "") -> None:
        self.console.print(Rule(text, style=f"dim {self._t['border']}"))

    def spinner(self, text: str = "Thinking…") -> "SpinnerContext":
        return SpinnerContext(self.console, text, self._t["primary"])

    def set_theme(self, theme_name: str) -> bool:
        if theme_name in THEMES:
            self._theme = THEMES[theme_name]
            self._t = self._theme
            self.config.theme = theme_name
            return True
        return False


class SpinnerContext:
    """Context manager for spinner animation."""

    def __init__(self, console: Console, text: str, color: str) -> None:
        self._console = console
        self._text = text
        self._color = color
        self._live: Live | None = None

    def __enter__(self) -> "SpinnerContext":
        spinner = Spinner("dots", text=Text(self._text, style=f"dim {self._color}"))
        self._live = Live(spinner, console=self._console, refresh_per_second=10)
        self._live.__enter__()
        return self

    def __exit__(self, *args) -> None:
        if self._live:
            self._live.__exit__(*args)


def _truncate_dict(d: dict, max_len: int = 80) -> str:
    """Format a dict for display, truncating long values."""
    parts = []
    for k, v in d.items():
        v_str = str(v)
        if len(v_str) > 30:
            v_str = v_str[:27] + "…"
        parts.append(f"{k}={repr(v_str)}")
    result = "{" + ", ".join(parts) + "}"
    if len(result) > max_len:
        result = result[:max_len - 3] + "…}"
    return result
