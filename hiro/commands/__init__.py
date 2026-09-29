"""
All slash commands for Hiro.
Comprehensive set: model management, MCP, file ops, security, context, and more.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import subprocess
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from .registry import command

if TYPE_CHECKING:
    from hiro.app import HiroApp


# ═══════════════════════════════════════════════════════════════════════════════
# GENERAL COMMANDS
# ═══════════════════════════════════════════════════════════════════════════════

@command(
    "help",
    aliases=["h", "?"],
    description="Show help and list all commands",
    usage="/help [command]",
    category="general",
)
async def cmd_help(app: "HiroApp", args: str) -> None:
    from hiro.commands.registry import list_commands, get_command

    if args:
        cmd = get_command(args.strip())
        if cmd:
            app.ui.print_panel(
                f"[bold]{cmd.usage}[/bold]\n\n{cmd.description}\n"
                + (f"\nAliases: {', '.join('/' + a for a in cmd.aliases)}" if cmd.aliases else ""),
                title=f"/{cmd.name}",
            )
        else:
            app.ui.print_error(f"Unknown command: /{args}")
        return

    from hiro.commands.registry import list_commands
    cmds = list_commands()
    categories: dict[str, list] = {}
    for cmd in cmds:
        categories.setdefault(cmd.category, []).append(cmd)

    cat_icons = {
        "general": "🔧",
        "model": "🤖",
        "context": "🧠",
        "file": "📁",
        "mcp": "🔌",
        "security": "🔒",
        "config": "⚙️",
        "session": "💾",
    }

    lines = []
    for cat, cmds_in_cat in sorted(categories.items()):
        icon = cat_icons.get(cat, "•")
        lines.append(f"\n[bold]{icon} {cat.upper()}[/bold]")
        for c in cmds_in_cat:
            aliases = f" ({', '.join('/' + a for a in c.aliases)})" if c.aliases else ""
            lines.append(f"  [bold cyan]/{c.name}[/bold cyan]{aliases}  {c.description}")

    lines.append(f"\n[dim]Tip: Type /help <command> for detailed help.[/dim]")
    app.ui.console.print("\n".join(lines))


@command(
    "exit",
    aliases=["quit", "q", "bye"],
    description="Exit Hiro",
    category="general",
)
async def cmd_exit(app: "HiroApp", args: str) -> None:
    app.ui.print_info("Goodbye! 👋")
    app.running = False


@command(
    "clear",
    aliases=["cls"],
    description="Clear the terminal screen",
    category="general",
)
async def cmd_clear(app: "HiroApp", args: str) -> None:
    os.system("cls" if sys.platform == "win32" else "clear")
    app.ui.print_banner(
        model=app.settings.model,
        provider=app.settings.provider,
        mcp_count=len(app.mcp.get_all_tools()) if app.mcp else 0,
    )


@command(
    "version",
    aliases=["v"],
    description="Show Hiro version information",
    category="general",
)
async def cmd_version(app: "HiroApp", args: str) -> None:
    import hiro
    app.ui.print_info(f"Hiro v{hiro.__version__}")
    app.ui.print_info(f"Python {sys.version.split()[0]}")


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL COMMANDS
# ═══════════════════════════════════════════════════════════════════════════════

@command(
    "model",
    aliases=["m"],
    description="Switch AI model. Usage: /model <name>",
    usage="/model [name]",
    category="model",
)
async def cmd_model(app: "HiroApp", args: str) -> None:
    from hiro.config import BUILTIN_MODELS, _infer_provider, get_api_key
    from hiro.providers import get_provider, clear_cache

    if not args:
        # Launch interactive model picker
        from hiro.ui.interactive import show_model_picker
        model = await show_model_picker(app.settings.model)
        if model:
            args = model  # fall through to the switch logic below
        else:
            return

    # Validate and switch
    model = args.strip()
    provider = _infer_provider(model)
    key = get_api_key(provider, app.settings)

    if not key and provider not in ("ollama",):
        app.ui.print_warning(f"No API key found for provider '{provider}'. Set with /key {provider} <key>")

    app.settings.model = model
    app.settings.provider = provider

    # Recreate provider
    from hiro.config import get_model_info, get_api_key, PROVIDER_BASE_URLS
    from hiro.providers import get_provider as gp
    base_url = app.settings.base_urls.get(provider, PROVIDER_BASE_URLS.get(provider, ""))
    clear_cache()
    app.provider = gp(provider, api_key=key, base_url=base_url)
    app.agent.provider = app.provider
    app.agent.model = model

    app.ui.print_success(f"Switched to model: [bold]{model}[/bold] (provider: {provider})")


@command(
    "key",
    aliases=["apikey"],
    description="Set an API key for a provider. Usage: /key <provider> <key>",
    usage="/key <provider> <api_key>",
    category="config",
)
async def cmd_key(app: "HiroApp", args: str) -> None:
    from hiro.config import PROVIDER_ENV_KEYS, save_settings

    parts = args.strip().split(None, 1)
    if not parts:
        # Launch interactive API key wizard
        from hiro.ui.interactive import show_api_key_wizard
        updated = await show_api_key_wizard(app.settings.api_keys)
        if updated:
            app.settings.api_keys = updated
            save_settings(app.settings)
            app.ui.print_success("API keys saved.")
        return

    if len(parts) < 2:
        app.ui.print_error("Usage: /key <provider> <api_key>")
        return

    provider, key = parts[0].lower(), parts[1]
    app.settings.api_keys[provider] = key
    save_settings(app.settings)
    app.ui.print_success(f"API key set for [bold]{provider}[/bold] (saved to config)")

    # Refresh provider if it's current
    if provider == app.settings.provider:
        from hiro.providers import get_provider as gp, clear_cache
        from hiro.config import PROVIDER_BASE_URLS
        base_url = app.settings.base_urls.get(provider, PROVIDER_BASE_URLS.get(provider, ""))
        clear_cache()
        app.provider = gp(provider, api_key=key, base_url=base_url)
        app.agent.provider = app.provider
        app.ui.print_success("Provider refreshed with new key")


@command(
    "temperature",
    aliases=["temp"],
    description="Set model temperature (0.0-2.0)",
    usage="/temperature <value>",
    category="model",
)
async def cmd_temperature(app: "HiroApp", args: str) -> None:
    if not args:
        app.ui.print_info(f"Current temperature: {app.settings.temperature}")
        return
    try:
        temp = float(args.strip())
        if not 0.0 <= temp <= 2.0:
            raise ValueError("Out of range")
        app.settings.temperature = temp
        app.ui.print_success(f"Temperature set to {temp}")
    except ValueError:
        app.ui.print_error("Temperature must be a float between 0.0 and 2.0")


@command(
    "provider",
    aliases=["prov"],
    description="List providers or set custom base URL. Usage: /provider list | /provider url <name> <url>",
    usage="/provider [list | url <provider> <base_url>]",
    category="model",
)
async def cmd_provider(app: "HiroApp", args: str) -> None:
    from hiro.config import PROVIDER_ENV_KEYS, PROVIDER_BASE_URLS

    parts = args.strip().split()
    if not parts or parts[0] == "list":
        rows = []
        for p in PROVIDER_ENV_KEYS:
            url = app.settings.base_urls.get(p, PROVIDER_BASE_URLS.get(p, "(default)"))
            rows.append([p, url])
        app.ui.print_table(["Provider", "Base URL"], rows, title="Providers")
        return

    if parts[0] == "url" and len(parts) >= 3:
        provider, url = parts[1], parts[2]
        app.settings.base_urls[provider] = url
        from hiro.config import save_settings
        save_settings(app.settings)
        app.ui.print_success(f"Base URL set for {provider}: {url}")
        return

    app.ui.print_error("Usage: /provider list | /provider url <name> <url>")


# ═══════════════════════════════════════════════════════════════════════════════
# CONTEXT COMMANDS
# ═══════════════════════════════════════════════════════════════════════════════

@command(
    "context",
    aliases=["ctx"],
    description="Show context/conversation information and token usage",
    category="context",
)
async def cmd_context(app: "HiroApp", args: str) -> None:
    ctx = app.agent.ctx
    msgs = ctx.messages
    token_est = ctx.token_estimate()
    usage = ctx.total_usage

    app.ui.print_rule("Context Status")

    rows = [
        ["Messages", str(len(msgs))],
        ["Turns", str(ctx.turn_count)],
        ["Token estimate", f"~{token_est:,}"],
        ["Input tokens (actual)", f"{usage.input_tokens:,}"],
        ["Output tokens (actual)", f"{usage.output_tokens:,}"],
        ["Cache read tokens", f"{usage.cache_read_tokens:,}"],
    ]
    app.ui.print_table(["Field", "Value"], rows, title="Context Info")

    if msgs:
        app.ui.print_rule("Message History")
        for i, msg in enumerate(msgs[-10:]):  # Show last 10
            role_style = {
                "user": "bold cyan",
                "assistant": "bold magenta",
                "tool": "dim yellow",
                "system": "dim green",
            }.get(msg.role, "white")
            content_preview = str(msg.content)[:100].replace("\n", "↵")
            app.ui.console.print(f"  [{role_style}]{msg.role}[/]: {content_preview}")
        if len(msgs) > 10:
            app.ui.console.print(f"  [dim]... {len(msgs)-10} more messages[/]")


@command(
    "compact",
    aliases=["compress"],
    description="Summarize and compact the conversation context to save tokens",
    usage="/compact [keep=10]",
    category="context",
)
async def cmd_compact(app: "HiroApp", args: str) -> None:
    keep = 10
    if args.strip().isdigit():
        keep = int(args.strip())

    before = len(app.agent.ctx.messages)
    msg = app.agent.ctx.compact(keep_recent=keep)
    after = len(app.agent.ctx.messages)

    if before == after:
        app.ui.print_info("Context is already compact (nothing to remove)")
    else:
        app.ui.print_success(f"Compacted: {before} → {after} messages ({before - after} removed)")


@command(
    "reset",
    aliases=["new", "clear_context"],
    description="Clear conversation history and start fresh",
    category="context",
)
async def cmd_reset(app: "HiroApp", args: str) -> None:
    app.agent.ctx.clear()
    app.ui.print_success("Conversation history cleared. Starting fresh!")


@command(
    "history",
    aliases=["hist"],
    description="Show conversation history",
    usage="/history [n]",
    category="context",
)
async def cmd_history(app: "HiroApp", args: str) -> None:
    ctx = app.agent.ctx
    n = int(args.strip()) if args.strip().isdigit() else len(ctx.messages)
    msgs = ctx.messages[-n:]

    if not msgs:
        app.ui.print_info("No conversation history")
        return

    app.ui.print_rule(f"History (last {len(msgs)} messages)")
    for msg in msgs:
        role_icon = {"user": "👤", "assistant": "🤖", "tool": "⚙️", "system": "⚡"}.get(msg.role, "•")
        content = str(msg.content)
        if len(content) > 200:
            content = content[:197] + "…"
        app.ui.console.print(f"\n{role_icon} [bold]{msg.role}[/]")
        app.ui.console.print(f"  {content}")


@command(
    "system",
    aliases=["sys", "sp"],
    description="Set or show system prompt. Use /system clear to remove.",
    usage="/system [<prompt> | clear | show | file <path>]",
    category="context",
)
async def cmd_system(app: "HiroApp", args: str) -> None:
    if not args or args == "show":
        if app.settings.system_prompt:
            app.ui.print_panel(app.settings.system_prompt, title="System Prompt")
        else:
            app.ui.print_info("No system prompt set (token-efficient mode)")
        return

    if args == "clear":
        app.settings.system_prompt = ""
        app.ui.print_success("System prompt cleared")
        return

    if args.startswith("file "):
        path = Path(args[5:].strip())
        if path.exists():
            app.settings.system_prompt = path.read_text()
            app.ui.print_success(f"System prompt loaded from {path}")
        else:
            app.ui.print_error(f"File not found: {path}")
        return

    app.settings.system_prompt = args
    app.ui.print_success("System prompt set")


# ═══════════════════════════════════════════════════════════════════════════════
# FILE COMMANDS
# ═══════════════════════════════════════════════════════════════════════════════

@command(
    "file",
    aliases=["f", "read"],
    description="Add a file to the conversation context",
    usage="/file <path>",
    category="file",
)
async def cmd_file(app: "HiroApp", args: str) -> None:
    if not args:
        app.ui.print_error("Usage: /file <path>")
        return

    path = Path(args.strip())
    if not path.exists():
        # Try relative to cwd
        path = Path(os.getcwd()) / args.strip()

    if not path.exists():
        app.ui.print_error(f"File not found: {args}")
        return

    try:
        content = path.read_text(encoding="utf-8", errors="replace")
        lines = len(content.splitlines())
        ext = path.suffix.lstrip(".")

        # Add to context as a user message
        file_context = f"Here is the contents of `{path}`:\n\n```{ext}\n{content}\n```"
        app.agent.ctx.add_user(file_context)

        app.ui.print_success(f"Added {path.name} to context ({lines:,} lines, {len(content):,} chars)")
    except Exception as e:
        app.ui.print_error(f"Failed to read {path}: {e}")


@command(
    "diff",
    aliases=["d"],
    description="Show git diff or diff between two files",
    usage="/diff [file1] [file2]",
    category="file",
)
async def cmd_diff(app: "HiroApp", args: str) -> None:
    parts = args.split()
    if len(parts) == 2:
        cmd = f"diff -u {parts[0]} {parts[1]}"
    else:
        cmd = "git diff" + (f" -- {parts[0]}" if parts else "")

    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, cwd=os.getcwd()
        )
        output = result.stdout or result.stderr
        if output:
            app.ui.print_code(output, "diff")
        else:
            app.ui.print_info("No diff output")
    except Exception as e:
        app.ui.print_error(str(e))


@command(
    "cd",
    description="Change working directory",
    usage="/cd <path>",
    category="file",
)
async def cmd_cd(app: "HiroApp", args: str) -> None:
    target = Path(args.strip()).expanduser() if args.strip() else Path.home()
    if not target.is_absolute():
        target = Path(os.getcwd()) / target

    if target.exists() and target.is_dir():
        os.chdir(target)
        app.agent._tool_executor.cwd = target
        app.ui.print_success(f"Changed to: {target}")
    else:
        app.ui.print_error(f"Directory not found: {target}")


@command(
    "pwd",
    description="Print working directory",
    category="file",
)
async def cmd_pwd(app: "HiroApp", args: str) -> None:
    app.ui.print_info(f"Working directory: [bold]{os.getcwd()}[/bold]")


@command(
    "ls",
    description="List files in current directory",
    usage="/ls [path]",
    category="file",
)
async def cmd_ls(app: "HiroApp", args: str) -> None:
    target = args.strip() or "."
    result = await app.agent._tool_executor.execute("list_dir", {"path": target})
    app.ui.console.print(result)


@command(
    "patch",
    description="Apply a patch from a file or stdin",
    usage="/patch <file.patch>",
    category="file",
)
async def cmd_patch(app: "HiroApp", args: str) -> None:
    if not args:
        app.ui.print_error("Usage: /patch <file.patch>")
        return
    result = subprocess.run(
        f"patch -p1 < {args}", shell=True, capture_output=True, text=True
    )
    if result.returncode == 0:
        app.ui.print_success("Patch applied successfully")
        if result.stdout:
            app.ui.console.print(result.stdout)
    else:
        app.ui.print_error(f"Patch failed:\n{result.stderr}")


# ═══════════════════════════════════════════════════════════════════════════════
# MCP COMMANDS
# ═══════════════════════════════════════════════════════════════════════════════

@command(
    "mcp",
    aliases=["tools"],
    description="Manage MCP servers and list available tools",
    usage="/mcp [list | tools | connect <name> | disconnect <name> | add | resources | prompts]",
    category="mcp",
)
async def cmd_mcp(app: "HiroApp", args: str) -> None:
    parts = args.strip().split(None, 1)
    sub = parts[0].lower() if parts else "list"
    sub_args = parts[1] if len(parts) > 1 else ""

    if sub in ("list", ""):
        # Show server status
        app.ui.print_rule("MCP Servers")
        if not app.settings.mcp_servers:
            app.ui.print_info("No MCP servers configured. Add them to ~/.config/hiro/config.toml")
            app.ui.print_info("Example:\n  [[mcp_servers]]\n  name = \"burp\"\n  command = [\"node\", \"burp-mcp.js\"]\n  transport = \"stdio\"")
            return

        rows = []
        for srv in app.settings.mcp_servers:
            connected = app.mcp.is_connected(srv.name) if app.mcp else False
            tools = len(app.mcp.get_tools_for_server(srv.name)) if app.mcp else 0
            errors = app.mcp.failed_servers().get(srv.name, "") if app.mcp else ""
            status = "✓ connected" if connected else (f"✗ {errors[:30]}" if errors else "○ disabled")
            rows.append([srv.name, srv.transport, str(tools), status])
        app.ui.print_table(["Server", "Transport", "Tools", "Status"], rows, title="MCP Servers")

    elif sub == "tools":
        # List all MCP tools
        if not app.mcp:
            app.ui.print_info("No MCP manager initialized")
            return
        tools = app.mcp.get_all_tools()
        if not tools:
            app.ui.print_info("No MCP tools available")
            return
        rows = [[t.name, t.server_name, t.description[:60]] for t in tools]
        app.ui.print_table(["Tool", "Server", "Description"], rows, title=f"MCP Tools ({len(tools)})")

    elif sub == "resources":
        if not app.mcp:
            return
        resources = app.mcp.get_all_resources()
        if not resources:
            app.ui.print_info("No MCP resources available")
            return
        rows = [[r.uri, r.server_name, r.name] for r in resources]
        app.ui.print_table(["URI", "Server", "Name"], rows, title="MCP Resources")

    elif sub == "prompts":
        if not app.mcp:
            return
        prompts = app.mcp.get_all_prompts()
        if not prompts:
            app.ui.print_info("No MCP prompts available")
            return
        rows = [[p.name, p.server_name, p.description[:60]] for p in prompts]
        app.ui.print_table(["Prompt", "Server", "Description"], rows, title="MCP Prompts")

    elif sub == "connect":
        if not sub_args:
            app.ui.print_error("Usage: /mcp connect <server_name>")
            return
        srv_name = sub_args.strip()
        srv_config = next((s for s in app.settings.mcp_servers if s.name == srv_name), None)
        if not srv_config:
            app.ui.print_error(f"Server '{srv_name}' not found in config")
            return
        app.ui.print_info(f"Connecting to {srv_name}…")
        success = await app.mcp.connect_server(srv_config)
        if success:
            tools = app.mcp.get_tools_for_server(srv_name)
            app.ui.print_success(f"Connected to {srv_name} ({len(tools)} tools)")
        else:
            err = app.mcp.failed_servers().get(srv_name, "Unknown error")
            app.ui.print_error(f"Failed to connect to {srv_name}: {err}")

    elif sub == "disconnect":
        if not sub_args:
            app.ui.print_error("Usage: /mcp disconnect <server_name>")
            return
        srv_name = sub_args.strip()
        srv_config = next((s for s in app.settings.mcp_servers if s.name == srv_name), None)
        if srv_config and app.mcp:
            await app.mcp.reconnect(srv_name, srv_config)
            app.ui.print_success(f"Disconnected from {srv_name}")

    elif sub == "call":
        # Direct tool call: /mcp call <tool_name> [json_args]
        call_parts = sub_args.split(None, 1)
        if not call_parts:
            app.ui.print_error("Usage: /mcp call <tool_name> [json_args]")
            return
        tool_name = call_parts[0]
        call_args = {}
        if len(call_parts) > 1:
            try:
                call_args = json.loads(call_parts[1])
            except json.JSONDecodeError as e:
                app.ui.print_error(f"Invalid JSON args: {e}")
                return
        app.ui.print_info(f"Calling tool: {tool_name}")
        result = await app.mcp.call_tool(tool_name, call_args)
        app.ui.print_panel(result.to_text(), title=f"Result: {tool_name}")

    elif sub == "add":
        # Interactive: add a new MCP server
        app.ui.print_info("Add an MCP server to ~/.config/hiro/config.toml:")
        app.ui.console.print("""
[bold]Example config entries:[/bold]

[dim]# BurpSuite MCP (stdio)[/dim]
[[mcp_servers]]
name = "burp"
command = ["node", "/path/to/burp-mcp-server/index.js"]
transport = "stdio"
tags = ["security", "burp"]
description = "BurpSuite Pro MCP integration"

[dim]# Nuclei scanner[/dim]
[[mcp_servers]]
name = "nuclei"
command = ["python", "-m", "nuclei_mcp"]
transport = "stdio"
tags = ["security", "scanner"]

[dim]# SSE server[/dim]
[[mcp_servers]]
name = "myserver"
url = "http://localhost:3000/sse"
transport = "sse"
""")

    else:
        app.ui.print_error(f"Unknown MCP subcommand: {sub}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECURITY COMMANDS
# ═══════════════════════════════════════════════════════════════════════════════

@command(
    "scan",
    aliases=["vuln"],
    description="Security scan a target (uses Nuclei/Nmap via MCP or direct)",
    usage="/scan <target> [--nmap | --nuclei | --nikto | --gobuster]",
    category="security",
)
async def cmd_scan(app: "HiroApp", args: str) -> None:
    if not args:
        app.ui.print_error("Usage: /scan <target> [--nmap|--nuclei|--nikto]")
        return

    parts = args.split()
    target = parts[0]
    flags = set(p.lstrip("-") for p in parts[1:])

    # If we have MCP nuclei tool, use it
    if app.mcp:
        nuclei_tool = next((t for t in app.mcp.get_all_tools() if "nuclei" in t.name.lower()), None)
        if nuclei_tool and "nmap" not in flags:
            app.ui.print_info(f"Running Nuclei scan on {target}…")
            result = await app.mcp.call_tool(nuclei_tool.name, {"target": target})
            app.ui.print_panel(result.to_text(), title=f"Nuclei Scan: {target}")
            return

    # Direct command execution
    tool_cmds = {
        "nmap": f"nmap -sV -sC -O {target}",
        "nuclei": f"nuclei -target {target} -severity medium,high,critical",
        "nikto": f"nikto -h {target}",
        "gobuster": f"gobuster dir -u {target} -w /usr/share/wordlists/dirb/common.txt",
        "ffuf": f"ffuf -w /usr/share/wordlists/dirb/common.txt -u {target}/FUZZ",
        "subfinder": f"subfinder -d {target}",
        "whatweb": f"whatweb {target}",
    }

    cmds_to_run = [v for k, v in tool_cmds.items() if not flags or k in flags]
    if not cmds_to_run:
        cmds_to_run = [tool_cmds["nmap"]]

    for cmd in cmds_to_run:
        tool_name = cmd.split()[0]
        if not shutil.which(tool_name):
            app.ui.print_warning(f"{tool_name} not found in PATH, skipping")
            continue

        app.ui.print_info(f"Running: {cmd}")
        result = await app.agent._tool_executor.execute("bash", {"command": cmd, "timeout": 300})
        app.ui.print_panel(result, title=f"{tool_name.capitalize()}: {target}")


@command(
    "burp",
    aliases=["bp"],
    description="Interact with BurpSuite via MCP or local proxy",
    usage="/burp [status | tools | proxy [url] | scan <url> | history | <tool_name>]",
    category="security",
)
async def cmd_burp(app: "HiroApp", args: str) -> None:
    burp_tools = []
    if app.mcp:
        burp_tools = [
            t for t in app.mcp.get_all_tools()
            if "burp" in t.server_name.lower() or "burp" in t.name.lower() or "portswigger" in t.server_name.lower()
        ]

    parts = args.strip().split(None, 1)
    sub = parts[0].lower() if parts else "status"
    sub_args = parts[1] if len(parts) > 1 else ""

    if sub == "status" or not args:
        import httpx
        app.ui.print_info("Checking BurpSuite status...")
        proxy_alive = False
        api_alive = False
        try:
            async with httpx.AsyncClient(timeout=1.0) as client:
                r = await client.get("http://127.0.0.1:8080")
                if r.status_code in (200, 400, 403, 500) or "burp" in r.text.lower():
                    proxy_alive = True
        except Exception:
            pass

        try:
            async with httpx.AsyncClient(timeout=1.0) as client:
                r = await client.get("http://127.0.0.1:1337/v0.1/version")
                if r.status_code == 200:
                    api_alive = True
        except Exception:
            pass

        current_proxy = os.environ.get("HTTP_PROXY", "(none)")
        mcp_status = f"[green]Connected ({len(burp_tools)} tools)[/]" if burp_tools else "[yellow]Not connected[/]"
        proxy_status = "[green]Active (127.0.0.1:8080)[/]" if proxy_alive else "[dim]Not responding[/]"
        api_status = "[green]Active (127.0.0.1:1337)[/]" if api_alive else "[dim]Not responding[/]"

        status_text = (
            f"● BurpSuite MCP Server: {mcp_status}\n"
            f"● Burp HTTP Proxy: {proxy_status}\n"
            f"● Burp REST API: {api_status}\n"
            f"● Active Environment Proxy: [cyan]{current_proxy}[/]\n\n"
        )
        if burp_tools:
            status_text += "[bold]Available BurpSuite Tools:[/bold]\n"
            for t in burp_tools:
                status_text += f"  • [cyan]{t.name}[/]: {t.description[:60]}\n"
            status_text += "\n[dim]Usage: /burp <tool_name> <arguments_json_or_url>[/dim]"
        else:
            status_text += (
                "[dim]To connect BurpSuite MCP, configure in ~/.config/hiro/config.toml:\n"
                "  [[mcp_servers]]\n"
                "  name = \"burp\"\n"
                "  command = [\"node\", \"/path/to/burp-mcp/index.js\"]\n"
                "  transport = \"stdio\"\n"
                "  env = {BURP_API_KEY = \"...\", BURP_URL = \"http://localhost:1337\"}\n\n"
                "To route CLI / web_fetch through Burp Proxy:\n"
                "  /burp proxy http://127.0.0.1:8080[/dim]"
            )
        app.ui.print_panel(status_text, title="BurpSuite Integration")
        return

    if sub == "proxy":
        proxy_target = sub_args.strip() if sub_args else "http://127.0.0.1:8080"
        if proxy_target in ("off", "none", "disable"):
            os.environ.pop("HTTP_PROXY", None)
            os.environ.pop("HTTPS_PROXY", None)
            os.environ.pop("http_proxy", None)
            os.environ.pop("https_proxy", None)
            app.ui.print_success("Burp proxy routing disabled.")
        else:
            os.environ["HTTP_PROXY"] = proxy_target
            os.environ["HTTPS_PROXY"] = proxy_target
            os.environ["http_proxy"] = proxy_target
            os.environ["https_proxy"] = proxy_target
            app.ui.print_success(f"Traffic routing configured to proxy: {proxy_target}")
        return

    if sub == "tools":
        if not burp_tools:
            app.ui.print_warning("No BurpSuite MCP tools connected.")
            return
        rows = [[t.name, t.description[:70]] for t in burp_tools]
        app.ui.print_table(["Tool", "Description"], rows, title="BurpSuite MCP Tools")
        return

    if not burp_tools:
        # Fall back to asking agent to analyze or test target
        app.ui.print_warning("BurpSuite MCP is not connected. Sending request to AI security analyzer...")
        await app.run_query(f"Using cybersecurity and pentesting methodology, perform: {args}")
        return

    # Match tool
    matching = [t for t in burp_tools if sub == t.name.lower() or sub in t.name.lower()]
    if not matching:
        tool_names = [t.name for t in burp_tools]
        app.ui.print_error(f"Unknown Burp tool '{sub}'. Available: {', '.join(tool_names)}")
        return

    tool = matching[0]
    call_args: dict = {}
    if sub_args:
        try:
            call_args = json.loads(sub_args)
        except Exception:
            call_args = {"url": sub_args, "target": sub_args}

    app.ui.print_info(f"Calling BurpSuite tool: {tool.name}")
    result = await app.mcp.call_tool(tool.name, call_args)
    app.ui.print_panel(result.to_text(), title=f"BurpSuite: {tool.name}")


@command(
    "audit",
    aliases=["pentest"],
    description="Run a comprehensive security audit on a target or codebase",
    usage="/audit <target_or_path>",
    category="security",
)
async def cmd_audit(app: "HiroApp", args: str) -> None:
    if not args:
        app.ui.print_error("Usage: /audit <target_or_path>")
        return

    target = args.strip()
    audit_prompt = f"""Perform a comprehensive security audit on: {target}

Please analyze for:
1. Common vulnerabilities (OWASP Top 10)
2. Misconfigurations
3. Exposed secrets or credentials
4. Injection vulnerabilities (SQLi, XSS, SSTI, etc.)
5. Authentication/authorization flaws
6. Dependency vulnerabilities

Use available tools (bash, search_files, web_fetch, MCP tools) to gather information.
Provide a structured report with severity ratings and remediation steps."""

    # This sends to agent, which will use all available tools
    await app.run_query(audit_prompt)


@command(
    "sqli",
    description="Test for SQL injection using available tools",
    usage="/sqli <url> [param]",
    category="security",
)
async def cmd_sqli(app: "HiroApp", args: str) -> None:
    if not args:
        app.ui.print_error("Usage: /sqli <url> [param]")
        return

    # Check for sqlmap in MCP or system
    if app.mcp:
        sqlmap_tool = next((t for t in app.mcp.get_all_tools() if "sqlmap" in t.name.lower()), None)
        if sqlmap_tool:
            result = await app.mcp.call_tool(sqlmap_tool.name, {"url": args})
            app.ui.print_panel(result.to_text(), title="SQLMap")
            return

    if shutil.which("sqlmap"):
        cmd = f"sqlmap -u {args} --batch --level=2 --risk=2"
        result = await app.agent._tool_executor.execute("bash", {"command": cmd, "timeout": 300})
        app.ui.print_panel(result, title="SQLMap Scan")
    else:
        app.ui.print_warning("sqlmap not found. Asking AI to analyze instead.")
        await app.run_query(f"Analyze this URL for SQL injection vulnerabilities: {args}")


@command(
    "xss",
    description="Test for XSS vulnerabilities",
    usage="/xss <url>",
    category="security",
)
async def cmd_xss(app: "HiroApp", args: str) -> None:
    if not args:
        app.ui.print_error("Usage: /xss <url>")
        return

    # Check for dalfox or similar
    if shutil.which("dalfox"):
        cmd = f"dalfox url {args}"
        result = await app.agent._tool_executor.execute("bash", {"command": cmd, "timeout": 120})
        app.ui.print_panel(result, title="Dalfox XSS Scan")
    else:
        await app.run_query(f"Test this URL for XSS vulnerabilities and provide PoC payloads: {args}")


@command(
    "secrets",
    description="Scan for secrets and credentials in code",
    usage="/secrets [path]",
    category="security",
)
async def cmd_secrets(app: "HiroApp", args: str) -> None:
    path = args.strip() or "."

    # Try trufflehog or gitleaks first
    for tool, cmd_template in [
        ("trufflehog", f"trufflehog filesystem {path}"),
        ("gitleaks", f"gitleaks detect --source {path}"),
        ("semgrep", f"semgrep --config p/secrets {path}"),
    ]:
        if shutil.which(tool):
            app.ui.print_info(f"Running {tool} on {path}…")
            result = await app.agent._tool_executor.execute("bash", {"command": cmd_template, "timeout": 120})
            app.ui.print_panel(result, title=f"{tool} Secret Scan")
            return

    # Fallback: AI-based scan via grep
    await app.run_query(
        f"Scan the codebase at '{path}' for secrets, API keys, passwords, tokens, and credentials. "
        f"Use search_files to look for patterns like 'password=', 'api_key', 'secret', 'token=', "
        f"'AWS_', 'private_key', etc. Report all findings with file locations."
    )


@command(
    "payload",
    aliases=["payloads"],
    description="Generate security payloads for testing",
    usage="/payload <type> [context]",
    category="security",
)
async def cmd_payload(app: "HiroApp", args: str) -> None:
    if not args:
        app.ui.console.print("""
[bold]Available payload types:[/bold]
  /payload xss [reflected|stored|dom]
  /payload sqli [error|blind|union|time]
  /payload rce [bash|python|php]
  /payload ssrf [basic|bypass]
  /payload lfi [basic|bypass|rce]
  /payload xxe [basic|oob]
  /payload ssti [jinja2|twig|freemarker]
  /payload csrf [html|json]
  /payload ssti [template]
  /payload jwt [none|weak|alg]
""")
        return

    await app.run_query(
        f"Generate security testing payloads for: {args}\n"
        f"Include:\n1. Basic payloads\n2. Bypass techniques\n3. Polyglots if applicable\n"
        f"4. Context-specific variations\n5. PoC examples\n"
        f"Format as a code block with explanations."
    )


@command(
    "decode",
    aliases=["dec"],
    description="Decode/analyze encoded strings (base64, JWT, URL, hex, etc.)",
    usage="/decode <string>",
    category="security",
)
async def cmd_decode(app: "HiroApp", args: str) -> None:
    if not args:
        app.ui.print_error("Usage: /decode <encoded_string>")
        return

    import base64
    import urllib.parse

    results = []
    text = args.strip()

    # Try base64
    try:
        decoded = base64.b64decode(text + "==").decode("utf-8", errors="replace")
        results.append(("Base64", decoded))
    except Exception:
        pass

    # URL decode
    url_decoded = urllib.parse.unquote(text)
    if url_decoded != text:
        results.append(("URL", url_decoded))

    # Hex decode
    try:
        if all(c in "0123456789abcdefABCDEF" for c in text.replace(" ", "")):
            hex_bytes = bytes.fromhex(text.replace(" ", ""))
            results.append(("Hex", hex_bytes.decode("utf-8", errors="replace")))
    except Exception:
        pass

    # JWT
    if text.count(".") == 2:
        parts = text.split(".")
        try:
            header = json.loads(base64.b64decode(parts[0] + "=="))
            payload = json.loads(base64.b64decode(parts[1] + "=="))
            results.append(("JWT Header", json.dumps(header, indent=2)))
            results.append(("JWT Payload", json.dumps(payload, indent=2)))
        except Exception:
            pass

    if results:
        for name, val in results:
            app.ui.print_panel(val, title=name)
    else:
        app.ui.print_warning("Could not auto-decode. Sending to AI for analysis.")
        await app.run_query(f"Analyze and decode this string: {args}")


@command(
    "hash",
    description="Hash a string or identify a hash",
    usage="/hash <string> [--md5|--sha1|--sha256] | /hash identify <hash>",
    category="security",
)
async def cmd_hash(app: "HiroApp", args: str) -> None:
    if not args:
        app.ui.print_error("Usage: /hash <string> | /hash identify <hash>")
        return

    import hashlib

    parts = args.split(None, 1)

    if parts[0] == "identify" and len(parts) > 1:
        h = parts[1].strip()
        await app.run_query(f"Identify this hash and suggest cracking strategies: {h}")
        return

    text = args.strip()
    data = text.encode()

    rows = [
        ["MD5", hashlib.md5(data).hexdigest()],
        ["SHA1", hashlib.sha1(data).hexdigest()],
        ["SHA256", hashlib.sha256(data).hexdigest()],
        ["SHA512", hashlib.sha512(data).hexdigest()],
    ]
    app.ui.print_table(["Algorithm", "Hash"], rows, title=f"Hashes of: {text[:30]}")


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION / CONFIG COMMANDS
# ═══════════════════════════════════════════════════════════════════════════════

@command(
    "save",
    description="Save current session to file",
    usage="/save [filename]",
    category="session",
)
async def cmd_save(app: "HiroApp", args: str) -> None:
    from hiro.config import SESSIONS_DIR, ensure_dirs
    import datetime

    ensure_dirs()
    name = args.strip() if args.strip() else f"session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if not name.endswith(".json"):
        name += ".json"

    path = SESSIONS_DIR / name
    data = {
        "model": app.settings.model,
        "provider": app.settings.provider,
        "messages": [
            {
                "role": m.role,
                "content": m.content if isinstance(m.content, str) else str(m.content),
            }
            for m in app.agent.ctx.messages
        ],
        "usage": {
            "input": app.agent.ctx.total_usage.input_tokens,
            "output": app.agent.ctx.total_usage.output_tokens,
        }
    }

    path.write_text(json.dumps(data, indent=2))
    app.ui.print_success(f"Session saved to {path}")


@command(
    "load",
    description="Load a saved session",
    usage="/load [filename]",
    category="session",
)
async def cmd_load(app: "HiroApp", args: str) -> None:
    from hiro.config import SESSIONS_DIR

    if not args:
        # List available sessions
        sessions = list(SESSIONS_DIR.glob("*.json"))
        if not sessions:
            app.ui.print_info("No saved sessions found")
            return
        rows = [[s.name, s.stat().st_mtime] for s in sorted(sessions)]
        app.ui.print_table(["Session", "Modified"], [[r[0], str(r[1])] for r in rows], title="Saved Sessions")
        return

    name = args.strip()
    if not name.endswith(".json"):
        name += ".json"

    path = SESSIONS_DIR / name
    if not path.exists():
        app.ui.print_error(f"Session not found: {path}")
        return

    try:
        data = json.loads(path.read_text())
        app.agent.ctx.clear()
        from hiro.providers.base import Message
        for msg in data.get("messages", []):
            app.agent.ctx.messages.append(Message(
                role=msg["role"],
                content=msg["content"],
            ))
        app.ui.print_success(f"Loaded session: {name} ({len(app.agent.ctx.messages)} messages)")
    except Exception as e:
        app.ui.print_error(f"Failed to load session: {e}")


@command(
    "config",
    aliases=["cfg"],
    description="Show or edit configuration",
    usage="/config [show | edit | save | reset]",
    category="config",
)
async def cmd_config(app: "HiroApp", args: str) -> None:
    from hiro.config import CONFIG_FILE, save_settings

    sub = args.strip().lower()

    if not sub or sub == "show":
        import tomli_w
        data = app.settings.model_dump(exclude_none=True)
        # Mask API keys
        if "api_keys" in data:
            data["api_keys"] = {k: v[:8] + "…" for k, v in data["api_keys"].items()}
        app.ui.print_code(tomli_w.dumps(data), "toml")
        return

    if sub == "edit":
        editor = os.environ.get("EDITOR", "notepad" if sys.platform == "win32" else "nano")
        save_settings(app.settings)
        subprocess.run([editor, str(CONFIG_FILE)])
        return

    if sub == "save":
        save_settings(app.settings)
        app.ui.print_success(f"Config saved to {CONFIG_FILE}")
        return

    if sub == "reset":
        from hiro.config import Settings
        app.settings = Settings()
        app.ui.print_success("Config reset to defaults")
        return

    app.ui.print_error(f"Unknown config subcommand: {sub}")


@command(
    "theme",
    description="Change color theme",
    usage="/theme [dark|light|hacker|nord]",
    category="config",
)
async def cmd_theme(app: "HiroApp", args: str) -> None:
    from hiro.ui.renderer import THEMES

    if not args:
        app.ui.print_info(f"Current theme: {app.settings.ui.theme}")
        app.ui.print_info(f"Available themes: {', '.join(THEMES.keys())}")
        return

    if app.ui.set_theme(args.strip()):
        app.settings.ui.theme = args.strip()
        app.ui.print_success(f"Theme changed to: {args.strip()}")
    else:
        app.ui.print_error(f"Unknown theme: {args.strip()}. Use: {', '.join(THEMES.keys())}")


@command(
    "tokens",
    aliases=["usage"],
    description="Show token usage statistics for this session",
    category="context",
)
async def cmd_tokens(app: "HiroApp", args: str) -> None:
    usage = app.agent.ctx.total_usage
    rows = [
        ["Input tokens", f"{usage.input_tokens:,}"],
        ["Output tokens", f"{usage.output_tokens:,}"],
        ["Cache read tokens", f"{usage.cache_read_tokens:,}"],
        ["Cache write tokens", f"{usage.cache_write_tokens:,}"],
        ["Total tokens", f"{usage.total_tokens:,}"],
    ]
    app.ui.print_table(["Metric", "Count"], rows, title="Token Usage (this session)")


@command(
    "run",
    aliases=["exec", "!"],
    description="Execute a shell command directly",
    usage="/run <command>",
    category="general",
)
async def cmd_run(app: "HiroApp", args: str) -> None:
    if not args:
        app.ui.print_error("Usage: /run <command>")
        return
    result = await app.agent._tool_executor.execute("bash", {"command": args})
    app.ui.console.print(result)


@command(
    "shell",
    aliases=["sh"],
    description="Open an interactive shell",
    category="general",
)
async def cmd_shell(app: "HiroApp", args: str) -> None:
    shell = os.environ.get("SHELL", "powershell" if sys.platform == "win32" else "/bin/bash")
    app.ui.print_info(f"Launching {shell}... (type 'exit' to return to Hiro)")
    subprocess.run([shell])
    app.ui.print_info("Returned to Hiro")


@command(
    "inspect",
    description="Inspect a tool or MCP capability in detail",
    usage="/inspect <tool_name>",
    category="mcp",
)
async def cmd_inspect(app: "HiroApp", args: str) -> None:
    if not args:
        app.ui.print_error("Usage: /inspect <tool_name>")
        return

    tool_name = args.strip()

    # Check built-in tools
    from hiro.tools import get_builtin_tools
    for t in get_builtin_tools():
        if t.name == tool_name:
            app.ui.print_panel(
                f"[bold]Name:[/bold] {t.name}\n"
                f"[bold]Description:[/bold] {t.description}\n\n"
                f"[bold]Schema:[/bold]\n{json.dumps(t.parameters, indent=2)}",
                title=f"Built-in Tool: {tool_name}",
            )
            return

    # Check MCP tools
    if app.mcp:
        for t in app.mcp.get_all_tools():
            if t.name == tool_name:
                app.ui.print_panel(
                    f"[bold]Name:[/bold] {t.name}\n"
                    f"[bold]Server:[/bold] {t.server_name}\n"
                    f"[bold]Description:[/bold] {t.description}\n\n"
                    f"[bold]Schema:[/bold]\n{json.dumps(t.input_schema, indent=2)}",
                    title=f"MCP Tool: {tool_name}",
                )
                return

    app.ui.print_error(f"Tool not found: {tool_name}")


@command(
    "import",
    description="Import a file or URL into context",
    usage="/import <path_or_url>",
    category="context",
)
async def cmd_import(app: "HiroApp", args: str) -> None:
    if not args:
        app.ui.print_error("Usage: /import <path_or_url>")
        return

    target = args.strip()
    if target.startswith(("http://", "https://")):
        result = await app.agent._tool_executor.execute("web_fetch", {"url": target})
        app.agent.ctx.add_user(f"Content from {target}:\n\n{result}")
        app.ui.print_success(f"Fetched and added to context: {target}")
    else:
        await cmd_file(app, target)


@command(
    "diff",
    description="Show git diff or compare two files",
    usage="/diff [file1 file2 | branch/commit]",
    category="file",
)
async def cmd_diff(app: "HiroApp", args: str) -> None:
    parts = args.strip().split()
    if len(parts) == 2 and Path(parts[0]).exists() and Path(parts[1]).exists():
        import difflib
        p1, p2 = Path(parts[0]), Path(parts[1])
        t1 = p1.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
        t2 = p2.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
        diff = "".join(difflib.unified_diff(t1, t2, fromfile=str(p1), tofile=str(p2)))
        if not diff:
            app.ui.print_info("Files are identical.")
        else:
            app.ui.print_code(diff, "diff")
        return

    # Fall back to git diff
    cmd = f"git diff {args}".strip()
    result = await app.agent._tool_executor.execute("bash", {"command": cmd})
    if not result or result == "(no output)":
        app.ui.print_info("No git changes detected.")
    else:
        app.ui.print_code(result, "diff")


@command(
    "patch",
    description="Apply unified diff patch to a file",
    usage="/patch <file> <patch_content_or_file>",
    category="file",
)
async def cmd_patch(app: "HiroApp", args: str) -> None:
    if not args:
        app.ui.print_error("Usage: /patch <target_file> [patch_file]")
        return
    parts = args.strip().split(None, 1)
    target_path = Path(parts[0])
    if not target_path.exists():
        app.ui.print_error(f"File not found: {target_path}")
        return

    if len(parts) > 1 and Path(parts[1]).exists():
        patch_file = parts[1]
        cmd = f"git apply {patch_file}"
        res = await app.agent._tool_executor.execute("bash", {"command": cmd})
        app.ui.print_panel(res, title=f"Patch Applied to {target_path}")
    else:
        app.ui.print_info("Send patch instructions to agent...")
        await app.run_query(f"Review and apply patch to {target_path}: {args}")


@command(
    "headers",
    description="Analyze security response headers of a URL",
    usage="/headers <url>",
    category="security",
)
async def cmd_headers(app: "HiroApp", args: str) -> None:
    if not args:
        app.ui.print_error("Usage: /headers <url>")
        return

    url = args.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    import httpx
    try:
        app.ui.print_info(f"Auditing security headers for {url}...")
        async with httpx.AsyncClient(follow_redirects=True, timeout=10.0, verify=False) as client:
            resp = await client.get(url)

        headers = resp.headers
        expected = [
            ("Strict-Transport-Security", "Enforces HTTPS connections"),
            ("Content-Security-Policy", "Mitigates XSS and data injection"),
            ("X-Frame-Options", "Prevents clickjacking"),
            ("X-Content-Type-Options", "Prevents MIME-type sniffing (nosniff)"),
            ("Referrer-Policy", "Controls referrer leakage"),
            ("Permissions-Policy", "Restricts browser API access"),
        ]

        rows = []
        for h, purpose in expected:
            val = headers.get(h)
            status = f"[green]Present[/]" if val else f"[red]Missing[/]"
            preview = f": {val[:40]}…" if val and len(val) > 40 else (f": {val}" if val else "")
            rows.append([h, status + preview, purpose])

        app.ui.print_table(["Header", "Status", "Security Purpose"], rows, title=f"Security Headers: {url}")

        cookies = resp.headers.get_list("set-cookie")
        if cookies:
            cookie_rows = []
            for c in cookies:
                c_name = c.split("=")[0]
                is_http_only = "httponly" in c.lower()
                is_secure = "secure" in c.lower()
                same_site = "samesite" in c.lower()
                cookie_rows.append([
                    c_name,
                    "[green]Yes[/]" if is_http_only else "[red]No[/]",
                    "[green]Yes[/]" if is_secure else "[red]No[/]",
                    "[green]Yes[/]" if same_site else "[yellow]Missing[/]",
                ])
            app.ui.print_table(["Cookie", "HttpOnly", "Secure", "SameSite"], cookie_rows, title="Cookie Flags")

    except Exception as e:
        app.ui.print_error(f"Failed to inspect headers: {e}")


@command(
    "cors",
    description="Test target URL for CORS misconfigurations",
    usage="/cors <url>",
    category="security",
)
async def cmd_cors(app: "HiroApp", args: str) -> None:
    if not args:
        app.ui.print_error("Usage: /cors <url>")
        return

    url = args.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    import httpx
    evil_origins = ["https://evil.com", "null", f"{url}.attacker.com"]
    findings = []

    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=10.0, verify=False) as client:
            for origin in evil_origins:
                resp = await client.get(url, headers={"Origin": origin})
                acao = resp.headers.get("access-control-allow-origin")
                acac = resp.headers.get("access-control-allow-credentials", "false").lower()

                if acao == origin:
                    if acac == "true":
                        findings.append(f"[bold red]CRITICAL:[/] Origin '{origin}' reflected with Access-Control-Allow-Credentials: true!")
                    else:
                        findings.append(f"[yellow]WARNING:[/] Origin '{origin}' reflected without credentials.")
                elif acao == "*":
                    findings.append(f"[dim]Wildcard '*' allowed (public resource, no credentials).[/]")

        if findings:
            app.ui.print_panel("\n".join(findings), title=f"CORS Misconfiguration Test: {url}")
        else:
            app.ui.print_success(f"No CORS misconfigurations detected for {url}.")

    except Exception as e:
        app.ui.print_error(f"CORS test failed: {e}")
