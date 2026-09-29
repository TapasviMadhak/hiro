"""
Configuration management for Hiro.
Uses TOML config file + environment variables.
Priority: env vars > config file > defaults
"""
from __future__ import annotations

import os
try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore
import tomli_w
from pathlib import Path
from typing import Any
from pydantic import BaseModel, Field


# ─── Paths ───────────────────────────────────────────────────────────────────

CONFIG_DIR = Path.home() / ".config" / "hiro"
CONFIG_FILE = CONFIG_DIR / "config.toml"
HISTORY_FILE = CONFIG_DIR / "history"
SESSIONS_DIR = CONFIG_DIR / "sessions"


def ensure_dirs() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


# ─── Model Registry ──────────────────────────────────────────────────────────

BUILTIN_MODELS: dict[str, dict[str, Any]] = {
    # Anthropic
    "claude-opus-4-5": {"provider": "anthropic", "ctx": 200000, "display": "Claude Opus 4.5"},
    "claude-sonnet-4-5": {"provider": "anthropic", "ctx": 200000, "display": "Claude Sonnet 4.5"},
    "claude-haiku-3-5": {"provider": "anthropic", "ctx": 200000, "display": "Claude Haiku 3.5"},
    "claude-opus-4": {"provider": "anthropic", "ctx": 200000, "display": "Claude Opus 4"},
    "claude-sonnet-4": {"provider": "anthropic", "ctx": 200000, "display": "Claude Sonnet 4"},
    "claude-3-5-sonnet-latest": {"provider": "anthropic", "ctx": 200000, "display": "Claude 3.5 Sonnet"},
    "claude-3-5-haiku-latest": {"provider": "anthropic", "ctx": 200000, "display": "Claude 3.5 Haiku"},
    "claude-3-opus-latest": {"provider": "anthropic", "ctx": 200000, "display": "Claude 3 Opus"},
    # OpenAI
    "gpt-4o": {"provider": "openai", "ctx": 128000, "display": "GPT-4o"},
    "gpt-4o-mini": {"provider": "openai", "ctx": 128000, "display": "GPT-4o Mini"},
    "gpt-4-turbo": {"provider": "openai", "ctx": 128000, "display": "GPT-4 Turbo"},
    "o1": {"provider": "openai", "ctx": 200000, "display": "o1"},
    "o1-mini": {"provider": "openai", "ctx": 128000, "display": "o1-mini"},
    "o3": {"provider": "openai", "ctx": 200000, "display": "o3"},
    "o3-mini": {"provider": "openai", "ctx": 128000, "display": "o3-mini"},
    "o4-mini": {"provider": "openai", "ctx": 200000, "display": "o4-mini"},
    # Google
    "gemini-2.5-pro": {"provider": "google", "ctx": 1000000, "display": "Gemini 2.5 Pro"},
    "gemini-2.5-flash": {"provider": "google", "ctx": 1000000, "display": "Gemini 2.5 Flash"},
    "gemini-2.0-flash": {"provider": "google", "ctx": 1000000, "display": "Gemini 2.0 Flash"},
    "gemini-1.5-pro": {"provider": "google", "ctx": 2000000, "display": "Gemini 1.5 Pro"},
    "gemini-1.5-flash": {"provider": "google", "ctx": 1000000, "display": "Gemini 1.5 Flash"},
    # Groq
    "llama-3.3-70b-versatile": {"provider": "groq", "ctx": 128000, "display": "Llama 3.3 70B"},
    "llama-3.1-8b-instant": {"provider": "groq", "ctx": 128000, "display": "Llama 3.1 8B"},
    "mixtral-8x7b-32768": {"provider": "groq", "ctx": 32768, "display": "Mixtral 8x7B"},
    "gemma2-9b-it": {"provider": "groq", "ctx": 8192, "display": "Gemma 2 9B"},
    # DeepSeek
    "deepseek-chat": {"provider": "deepseek", "ctx": 64000, "display": "DeepSeek Chat"},
    "deepseek-reasoner": {"provider": "deepseek", "ctx": 64000, "display": "DeepSeek Reasoner"},
    # Mistral
    "mistral-large-latest": {"provider": "mistral", "ctx": 128000, "display": "Mistral Large"},
    "mistral-small-latest": {"provider": "mistral", "ctx": 32000, "display": "Mistral Small"},
    "codestral-latest": {"provider": "mistral", "ctx": 256000, "display": "Codestral"},
    # Ollama (local)
    "ollama/llama3.2": {"provider": "ollama", "ctx": 131072, "display": "Ollama Llama 3.2"},
    "ollama/codellama": {"provider": "ollama", "ctx": 100000, "display": "Ollama CodeLlama"},
    "ollama/qwen2.5-coder": {"provider": "ollama", "ctx": 128000, "display": "Ollama Qwen2.5 Coder"},
    "ollama/deepseek-coder-v2": {"provider": "ollama", "ctx": 128000, "display": "Ollama DeepSeek Coder V2"},
    # NVIDIA NIM
    "meta/llama-3.3-70b-instruct": {"provider": "nvidia", "ctx": 128000, "display": "NVIDIA Llama 3.3 70B"},
    "meta/llama-3.1-405b-instruct": {"provider": "nvidia", "ctx": 128000, "display": "NVIDIA Llama 3.1 405B"},
    "deepseek-ai/deepseek-r1": {"provider": "nvidia", "ctx": 128000, "display": "NVIDIA DeepSeek R1"},
    "nvidia/llama-3.1-nemotron-70b-instruct": {"provider": "nvidia", "ctx": 128000, "display": "NVIDIA Nemotron 70B"},
    "qwen/qwen2.5-coder-32b-instruct": {"provider": "nvidia", "ctx": 32768, "display": "NVIDIA Qwen 2.5 Coder 32B"},
    "deepseek-ai/deepseek-v4.1-flash": {"provider": "nvidia", "ctx": 128000, "display": "NVIDIA DeepSeek V4.1 Flash"},
    "moonshotai/kimi-k3": {"provider": "nvidia", "ctx": 128000, "display": "NVIDIA Kimi K3"},
    "z-ai/glm-5.3": {"provider": "nvidia", "ctx": 128000, "display": "NVIDIA GLM 5.3"},
    "z-ai/glm-5.3-flash": {"provider": "nvidia", "ctx": 128000, "display": "NVIDIA GLM 5.3 Flash"},
}

PROVIDER_ENV_KEYS: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "groq": "GROQ_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "together": "TOGETHER_API_KEY",
    "fireworks": "FIREWORKS_API_KEY",
    "nvidia": "NVIDIA_API_KEY",
}

PROVIDER_BASE_URLS: dict[str, str] = {
    "deepseek": "https://api.deepseek.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "together": "https://api.together.xyz/v1",
    "fireworks": "https://api.fireworks.ai/inference/v1",
    "mistral": "https://api.mistral.ai/v1",
    "ollama": "http://localhost:11434/v1",
    "nvidia": "https://integrate.api.nvidia.com/v1",
}


# ─── Config Schema ───────────────────────────────────────────────────────────

class MCPServerConfig(BaseModel):
    name: str
    command: list[str] = Field(default_factory=list)
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    url: str = ""  # for SSE/HTTP transport
    transport: str = "stdio"  # stdio | sse | http
    enabled: bool = True
    description: str = ""
    tags: list[str] = Field(default_factory=list)  # e.g. ["security", "burp"]


class AgentConfig(BaseModel):
    max_turns: int = 50
    max_tokens_per_turn: int = 8192
    token_budget: int = 200000
    auto_compact: bool = True
    compact_threshold: float = 0.80  # compact at 80% context usage
    parallel_tools: bool = True


class UIConfig(BaseModel):
    theme: str = "dark"
    syntax_theme: str = "monokai"
    show_token_count: bool = True
    show_timing: bool = True
    stream: bool = True
    word_wrap: bool = True
    max_output_height: int = 40


class Settings(BaseModel):
    # Core
    model: str = "claude-sonnet-4-5"
    provider: str = ""  # auto-detect from model if empty
    temperature: float = 0.0

    # API Keys (can also be in env)
    api_keys: dict[str, str] = Field(default_factory=dict)

    # Custom base URLs
    base_urls: dict[str, str] = Field(default_factory=dict)

    # System prompt (empty = no pre-prompt, token-efficient)
    system_prompt: str = ""
    system_prompt_file: str = ""

    # MCP Servers
    mcp_servers: list[MCPServerConfig] = Field(default_factory=list)

    # Agent behavior
    agent: AgentConfig = Field(default_factory=AgentConfig)

    # UI
    ui: UIConfig = Field(default_factory=UIConfig)

    # Shell
    shell: str = ""  # auto-detect
    working_directory: str = ""  # empty = cwd

    # Security
    allow_shell: bool = True
    allow_network: bool = True
    confirm_shell: bool = False  # ask before running shell commands


# ─── Config Loader ───────────────────────────────────────────────────────────

_settings: Settings | None = None


def load_settings(config_path: Path | None = None) -> Settings:
    """Load settings from config file and environment variables."""
    global _settings

    path = config_path or CONFIG_FILE
    raw: dict[str, Any] = {}

    # Load from .env if exists
    env_file = Path.cwd() / ".env"
    if env_file.exists():
        from dotenv import load_dotenv
        load_dotenv(env_file, override=False)

    # Load TOML config
    if path.exists():
        with open(path, "rb") as f:
            raw = tomllib.load(f)

    settings = Settings(**raw) if raw else Settings()

    # Apply env var overrides
    for provider, env_key in PROVIDER_ENV_KEYS.items():
        val = os.environ.get(env_key, "")
        if val and provider not in settings.api_keys:
            settings.api_keys[provider] = val

    # Model env var
    if model_env := os.environ.get("HIRO_MODEL"):
        settings.model = model_env

    # Resolve provider from model
    if not settings.provider:
        settings.provider = _infer_provider(settings.model)

    _settings = settings
    return settings


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = load_settings()
    return _settings


def save_settings(settings: Settings, path: Path | None = None) -> None:
    """Persist settings to TOML config."""
    ensure_dirs()
    target = path or CONFIG_FILE
    data = settings.model_dump(exclude_none=True)
    with open(target, "wb") as f:
        tomli_w.dump(data, f)


def _infer_provider(model: str) -> str:
    """Infer provider from model name."""
    if model in BUILTIN_MODELS:
        return BUILTIN_MODELS[model]["provider"]
    if model.startswith("claude"):
        return "anthropic"
    if model.startswith(("gpt-", "o1", "o3", "o4")):
        return "openai"
    if model.startswith("gemini"):
        return "google"
    if model.startswith("llama") or model.startswith("mixtral") or model.startswith("gemma"):
        return "groq"
    if model.startswith("deepseek"):
        return "deepseek"
    if model.startswith("mistral") or model.startswith("codestral"):
        return "mistral"
    if model.startswith("ollama/"):
        return "ollama"
    if model.startswith(("nvidia", "meta/", "deepseek-ai/", "moonshotai/", "z-ai/", "qwen/")):
        return "nvidia"
    return "openai"  # fallback to openai-compatible


def get_api_key(provider: str, settings: Settings | None = None) -> str:
    """Get API key for a provider."""
    s = settings or get_settings()
    # Check config
    if key := s.api_keys.get(provider):
        return key
    # Check env
    env_key = PROVIDER_ENV_KEYS.get(provider, f"{provider.upper()}_API_KEY")
    return os.environ.get(env_key, "")


def get_model_info(model: str) -> dict[str, Any]:
    """Get model metadata."""
    if model in BUILTIN_MODELS:
        return BUILTIN_MODELS[model]
    # Unknown model - return defaults
    provider = _infer_provider(model)
    return {"provider": provider, "ctx": 128000, "display": model}


def list_models(provider: str | None = None) -> list[tuple[str, dict]]:
    """List available models, optionally filtered by provider."""
    models = list(BUILTIN_MODELS.items())
    if provider:
        models = [(k, v) for k, v in models if v["provider"] == provider]
    return models
