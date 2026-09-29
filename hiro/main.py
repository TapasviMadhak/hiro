"""
Hiro CLI entry point.
Handles argument parsing and application startup.
"""
from __future__ import annotations

import asyncio
import sys
import os
from pathlib import Path

import click

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("prompt", nargs=-1, required=False)
@click.option("-m", "--model", default="", help="AI model to use")
@click.option("-p", "--provider", default="", help="Provider (anthropic/openai/google/groq/ollama/…)")
@click.option("-k", "--api-key", default="", help="API key for the provider")
@click.option("-s", "--system", default="", help="System prompt (empty = no pre-prompt)")
@click.option("-c", "--config", default="", help="Path to config file")
@click.option("--no-tools", is_flag=True, help="Disable built-in tools")
@click.option("--no-shell", is_flag=True, help="Disable shell execution")
@click.option("--no-mcp", is_flag=True, help="Disable MCP server connections")
@click.option("--theme", default="", help="Color theme (dark/light/hacker/nord)")
@click.option("--temperature", "-t", default=-1.0, help="Model temperature (0.0-2.0)")
@click.option("--max-tokens", default=0, help="Max tokens per response")
@click.option("--version", "-v", is_flag=True, help="Show version and exit")
@click.option("--list-models", is_flag=True, help="List all available models")
@click.option("--mcp-server", multiple=True, help="Extra MCP server (name:command)")
@click.option("--url", default="", help="Custom base URL for provider")
@click.option("--stream/--no-stream", default=True, help="Enable/disable streaming")
def main(
    prompt: tuple,
    model: str,
    provider: str,
    api_key: str,
    system: str,
    config: str,
    no_tools: bool,
    no_shell: bool,
    no_mcp: bool,
    theme: str,
    temperature: float,
    max_tokens: int,
    version: bool,
    list_models: bool,
    mcp_server: tuple,
    url: str,
    stream: bool,
) -> None:
    """
    Hiro — Terminal AI coding assistant.

    \b
    Examples:
      hiro                           Start interactive session
      hiro "Fix the bug in main.py"  One-shot query
      hiro -m claude-opus-4          Use specific model
      hiro -m gpt-4o -k sk-...       OpenAI with key
      hiro -m ollama/llama3.2        Use local Ollama model
      hiro --list-models             Show all available models
    """
    import hiro as _hiro

    if version:
        click.echo(f"Hiro v{_hiro.__version__}")
        sys.exit(0)

    if list_models:
        from hiro.config import list_models as lm, get_api_key
        settings = _load_settings_cli(config)
        _print_models(settings)
        sys.exit(0)

    asyncio.run(_async_main(
        prompt=prompt,
        model=model,
        provider=provider,
        api_key=api_key,
        system=system,
        config_path=config,
        no_tools=no_tools,
        no_shell=no_shell,
        no_mcp=no_mcp,
        theme=theme,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_mcp_servers=mcp_server,
        base_url=url,
        stream=stream,
    ))


async def _async_main(
    prompt: tuple,
    model: str,
    provider: str,
    api_key: str,
    system: str,
    config_path: str,
    no_tools: bool,
    no_shell: bool,
    no_mcp: bool,
    theme: str,
    temperature: float,
    max_tokens: int,
    extra_mcp_servers: tuple,
    base_url: str,
    stream: bool,
) -> None:
    """Async entry point."""
    from hiro.config import load_settings, MCPServerConfig
    from hiro.app import HiroApp

    # Load settings
    settings = _load_settings_cli(config_path)

    # Apply CLI overrides
    if model:
        settings.model = model
        from hiro.config import _infer_provider
        if not provider:
            settings.provider = _infer_provider(model)

    if provider:
        settings.provider = provider

    if api_key:
        settings.api_keys[settings.provider] = api_key

    if base_url:
        settings.base_urls[settings.provider] = base_url

    if not settings.provider:
        from hiro.config import _infer_provider
        settings.provider = _infer_provider(settings.model)

    if theme:
        settings.ui.theme = theme

    if temperature >= 0:
        settings.temperature = temperature

    if max_tokens > 0:
        settings.agent.max_tokens_per_turn = max_tokens

    if system:
        settings.system_prompt = system

    if no_shell:
        settings.allow_shell = False

    if no_mcp:
        # Clear MCP servers
        settings.mcp_servers = []

    if no_tools:
        settings.allow_shell = False
        settings.allow_network = False

    # Add extra MCP servers from CLI
    for srv_spec in extra_mcp_servers:
        parts = srv_spec.split(":", 1)
        if len(parts) == 2:
            name, cmd = parts
            settings.mcp_servers.append(MCPServerConfig(
                name=name,
                command=cmd.split(),
                transport="stdio",
            ))

    settings.ui.stream = stream

    # Create and run app
    app = HiroApp(settings)

    # Initialize MCP if not disabled
    if not no_mcp and settings.mcp_servers:
        await app.initialize_mcp()

    # One-shot mode: single query then exit
    if prompt:
        user_input = " ".join(prompt)
        await app.run_query(user_input)
        await app.mcp.disconnect_all()
        return

    # Interactive mode
    await app.run()


def _load_settings_cli(config_path: str):
    from hiro.config import load_settings
    from pathlib import Path
    path = Path(config_path) if config_path else None
    return load_settings(path)


def _print_models(settings) -> None:
    """Print all available models."""
    from hiro.config import BUILTIN_MODELS, get_api_key
    from rich.console import Console
    from rich.table import Table
    from rich import box

    console = Console(legacy_windows=False)
    table = Table(box=box.ROUNDED, title="Available Models", show_header=True)
    table.add_column("Model ID", style="cyan")
    table.add_column("Display Name")
    table.add_column("Provider", style="green")
    table.add_column("Context")
    table.add_column("Key", style="yellow")

    for model_id, info in BUILTIN_MODELS.items():
        key = get_api_key(info["provider"], settings)
        has_key = "✓" if key else "✗"
        ctx = f"{info['ctx']//1000}K"
        table.add_row(model_id, info["display"], info["provider"], ctx, has_key)

    console.print(table)


if __name__ == "__main__":
    main()
