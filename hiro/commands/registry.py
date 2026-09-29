"""
Slash command registry and base class.
All /commands are registered here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Awaitable, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from hiro.app import HiroApp


@dataclass
class Command:
    name: str
    aliases: list[str]
    description: str
    usage: str
    handler: Callable[["HiroApp", str], Awaitable[None]]
    category: str = "general"
    hidden: bool = False


_registry: dict[str, Command] = {}


def command(
    name: str,
    aliases: list[str] = None,
    description: str = "",
    usage: str = "",
    category: str = "general",
    hidden: bool = False,
) -> Callable:
    """Decorator to register a slash command."""
    def decorator(fn: Callable) -> Callable:
        cmd = Command(
            name=name,
            aliases=aliases or [],
            description=description,
            usage=usage or f"/{name}",
            handler=fn,
            category=category,
            hidden=hidden,
        )
        _registry[name] = cmd
        for alias in cmd.aliases:
            _registry[alias] = cmd
        return fn
    return decorator


def get_command(name: str) -> Command | None:
    return _registry.get(name)


def list_commands(category: str = "") -> list[Command]:
    seen = set()
    cmds = []
    for cmd in _registry.values():
        if cmd.name not in seen:
            seen.add(cmd.name)
            if not cmd.hidden and (not category or cmd.category == category):
                cmds.append(cmd)
    return sorted(cmds, key=lambda c: (c.category, c.name))


def parse_command(text: str) -> tuple[str, str] | None:
    """
    Parse a slash command from user input.
    Returns (command_name, args) or None if not a command.
    """
    text = text.strip()
    if not text.startswith("/"):
        return None
    # Remove leading slash
    text = text[1:]
    # Split on first whitespace
    parts = text.split(None, 1)
    cmd_name = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""
    return cmd_name, args


def get_completions(prefix: str) -> list[str]:
    """Get command completions for a prefix."""
    prefix = prefix.lstrip("/")
    return [
        f"/{cmd.name}" for cmd in list_commands()
        if cmd.name.startswith(prefix)
    ]
