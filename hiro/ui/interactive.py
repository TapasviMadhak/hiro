"""
Interactive TUI components for Hiro.
Provides arrow-key menus for commands, model selection, and API key setup.
Compatible with prompt_toolkit 3.x API.
"""
from __future__ import annotations

import asyncio

from prompt_toolkit.formatted_text import HTML, AnyFormattedText
from prompt_toolkit.styles import Style
from prompt_toolkit.shortcuts import radiolist_dialog, input_dialog, yes_no_dialog


# ─── Shared Style ────────────────────────────────────────────────────────────

DIALOG_STYLE = Style.from_dict({
    "dialog":              "bg:#1E1B4B",
    "dialog.body":         "bg:#0F172A fg:#E2E8F0",
    "dialog.shadow":       "bg:#000000",
    "frame.label":         "fg:#7C3AED bold",
    "button":              "bg:#334155 fg:#E2E8F0",
    "button.focused":      "bg:#7C3AED fg:#FFFFFF bold",
    "radio-list":          "bg:#0F172A fg:#CBD5E1",
    "radio-list focused":  "bg:#1E1B4B fg:#FFFFFF",
    "radio":               "fg:#7C3AED",
    "radio-checked":       "fg:#06B6D4 bold",
    "label":               "fg:#94A3B8",
    "text-area":           "bg:#1E293B fg:#F1F5F9",
    "text-area focused":   "bg:#1E293B fg:#FFFFFF",
})


# ─── Command Menu ─────────────────────────────────────────────────────────────

CATEGORY_ICONS = {
    "general":  "🔧",
    "model":    "🤖",
    "context":  "🧠",
    "file":     "📁",
    "mcp":      "🔌",
    "security": "🔒",
    "session":  "💾",
    "dev":      "⚙",
    "config":   "⚙",
}


async def show_command_menu() -> str | None:
    """
    Show an interactive arrow-key menu of all slash commands.
    Returns the selected command string (e.g. '/help') or None if cancelled.
    """
    from hiro.commands.registry import list_commands
    cmds = list_commands()

    categories: dict[str, list] = {}
    for cmd in cmds:
        categories.setdefault(cmd.category, []).append(cmd)

    values: list[tuple[str, AnyFormattedText]] = []
    for cat, cat_cmds in sorted(categories.items()):
        icon = CATEGORY_ICONS.get(cat, "▸")
        for cmd in sorted(cat_cmds, key=lambda c: c.name):
            label = HTML(
                f"<ansibrightblack>{icon}</ansibrightblack>  "
                f"<ansicyan>/{cmd.name:<20}</ansicyan>"
                f"<ansibrightblack>  {cmd.description[:55]}</ansibrightblack>"
            )
            values.append((f"/{cmd.name}", label))

    result = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: radiolist_dialog(
            title=HTML("<ansimagenta><b> ⚡ HIRO Commands </b></ansimagenta>"),
            text=HTML("<ansibrightblack>↑↓ navigate   Space select   Enter confirm   Ctrl+C cancel</ansibrightblack>"),
            values=values,
            ok_text="  Run  ",
            cancel_text=" Cancel ",
            style=DIALOG_STYLE,
        ).run()
    )
    return result


# ─── Model Picker ─────────────────────────────────────────────────────────────

PROVIDER_ICONS = {
    "anthropic": "🟠",
    "openai":    "🟢",
    "google":    "🔵",
    "groq":      "⚡",
    "deepseek":  "🟣",
    "mistral":   "🌊",
    "nvidia":    "🟩",
    "ollama":    "🖥",
}


async def show_model_picker(current_model: str = "") -> str | None:
    """
    Show an interactive model selection menu grouped by provider.
    Returns the selected model ID or None if cancelled.
    """
    from hiro.config import BUILTIN_MODELS

    by_provider: dict[str, list] = {}
    for model_id, info in BUILTIN_MODELS.items():
        prov = info.get("provider", "other")
        by_provider.setdefault(prov, []).append((model_id, info))

    values: list[tuple[str, AnyFormattedText]] = []
    for prov in sorted(by_provider.keys()):
        icon = PROVIDER_ICONS.get(prov, "◆")
        prov_label = prov.upper()
        for model_id, info in sorted(by_provider[prov], key=lambda x: x[0]):
            ctx_k = info.get("ctx", 0) // 1000
            ctx_str = f"{ctx_k}K" if ctx_k < 1000 else f"{ctx_k//1000}M"
            display = info.get("display", model_id)
            marker = " ●" if model_id == current_model else "  "
            label = HTML(
                f"<ansibrightblack>{icon} {prov_label:<10}</ansibrightblack>"
                f"<ansicyan>{display:<32}</ansicyan>"
                f"<ansibrightblack>{ctx_str:>5} ctx{marker}</ansibrightblack>"
            )
            values.append((model_id, label))

    result = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: radiolist_dialog(
            title=HTML("<ansimagenta><b> 🤖 Select Model </b></ansimagenta>"),
            text=HTML(
                f"<ansibrightblack>Current: </ansibrightblack>"
                f"<ansicyan>{current_model}</ansicyan>"
                f"<ansibrightblack>  ·  ↑↓ navigate   Space select   Enter confirm</ansibrightblack>"
            ),
            values=values,
            ok_text=" Select ",
            cancel_text=" Cancel ",
            style=DIALOG_STYLE,
        ).run()
    )
    return result


# ─── API Key Wizard ───────────────────────────────────────────────────────────

PROVIDER_DOCS = {
    "anthropic":  "https://console.anthropic.com/keys",
    "openai":     "https://platform.openai.com/api-keys",
    "google":     "https://aistudio.google.com/app/apikey",
    "groq":       "https://console.groq.com/keys",
    "deepseek":   "https://platform.deepseek.com/api_keys",
    "mistral":    "https://console.mistral.ai/api-keys/",
    "openrouter": "https://openrouter.ai/keys",
    "together":   "https://api.together.xyz/settings/api-keys",
    "fireworks":  "https://fireworks.ai/account/api-keys",
    "nvidia":     "https://build.nvidia.com/settings/api-key",
}

PROVIDER_NAMES = {
    "anthropic":  "Anthropic  (Claude)",
    "openai":     "OpenAI     (GPT / o-series)",
    "google":     "Google     (Gemini)",
    "groq":       "Groq       (fast Llama / Mixtral)",
    "deepseek":   "DeepSeek",
    "mistral":    "Mistral AI",
    "openrouter": "OpenRouter (multi-provider)",
    "together":   "Together AI",
    "fireworks":  "Fireworks AI",
    "nvidia":     "NVIDIA NIM",
}


async def show_api_key_wizard(current_keys: dict[str, str]) -> dict[str, str] | None:
    """
    Interactive wizard to add/update API keys.
    Returns updated keys dict or None if cancelled.
    """
    # Step 1: pick provider
    values: list[tuple[str, AnyFormattedText]] = []
    for provider, name in PROVIDER_NAMES.items():
        has_key = "✓" if current_keys.get(provider) else " "
        url = PROVIDER_DOCS.get(provider, "")
        label = HTML(
            f"<ansicyan>[{has_key}]</ansicyan> "
            f"<ansiwhite>{name:<38}</ansiwhite>"
            f"<ansibrightblack>{url}</ansibrightblack>"
        )
        values.append((provider, label))

    provider = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: radiolist_dialog(
            title=HTML("<ansimagenta><b> 🔑 API Key Setup </b></ansimagenta>"),
            text=HTML(
                "<ansibrightblack>[✓] = key already set  ·  select a provider to update its key</ansibrightblack>"
            ),
            values=values,
            ok_text=" Select ",
            cancel_text=" Cancel ",
            style=DIALOG_STYLE,
        ).run()
    )

    if not provider:
        return None

    # Step 2: enter key
    existing = current_keys.get(provider, "")
    hint = f"Current: {'*' * min(len(existing), 8)}..." if existing else "No key set yet"
    url = PROVIDER_DOCS.get(provider, "")

    key_value = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: input_dialog(
            title=HTML(f"<ansimagenta><b> 🔑 {PROVIDER_NAMES.get(provider, provider)} </b></ansimagenta>"),
            text=HTML(
                f"<ansibrightblack>{hint}\nGet your key at: {url}\n\nPaste API key below:</ansibrightblack>"
            ),
            password=True,
            ok_text=" Save ",
            cancel_text=" Cancel ",
            style=DIALOG_STYLE,
        ).run()
    )

    if key_value is None:
        return None

    updated = dict(current_keys)
    updated[provider] = key_value.strip()
    return updated


# ─── Provider URL Wizard ───────────────────────────────────────────────────────

async def show_base_url_wizard(provider: str, current_url: str = "") -> str | None:
    """Prompt for a custom base URL for a provider."""
    result = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: input_dialog(
            title=HTML(f"<ansimagenta><b> 🌐 Custom Base URL — {provider} </b></ansimagenta>"),
            text=HTML(
                f"<ansibrightblack>Current: {current_url or '(default)'}\n\n"
                "Enter the new base URL (e.g. http://localhost:1234/v1):</ansibrightblack>"
            ),
            default=current_url,
            ok_text=" Save ",
            cancel_text=" Cancel ",
            style=DIALOG_STYLE,
        ).run()
    )
    return result.strip() if result else None


# ─── Confirm Dialog ────────────────────────────────────────────────────────────

async def show_confirm(title: str, text: str) -> bool:
    result = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: yes_no_dialog(
            title=HTML(f"<ansimagenta><b> {title} </b></ansimagenta>"),
            text=HTML(f"<ansibrightblack>{text}</ansibrightblack>"),
            yes_text=" Yes ",
            no_text="  No ",
            style=DIALOG_STYLE,
        ).run()
    )
    return bool(result)
