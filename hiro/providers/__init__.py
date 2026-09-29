"""
Provider factory and registry.
Instantiates the correct provider based on provider name.
"""
from __future__ import annotations

from typing import Any
from .base import BaseProvider


_provider_cache: dict[str, BaseProvider] = {}


def get_provider(
    provider: str,
    api_key: str = "",
    base_url: str = "",
    **kwargs: Any,
) -> BaseProvider:
    """Get or create a provider instance."""
    cache_key = f"{provider}:{api_key[:8] if api_key else 'nokey'}:{base_url}"
    if cache_key in _provider_cache:
        return _provider_cache[cache_key]

    instance = _create_provider(provider, api_key, base_url, **kwargs)
    _provider_cache[cache_key] = instance
    return instance


def _create_provider(
    provider: str,
    api_key: str,
    base_url: str,
    **kwargs: Any,
) -> BaseProvider:
    """Create a new provider instance."""
    p = provider.lower()

    if p == "anthropic":
        from .anthropic import AnthropicProvider
        return AnthropicProvider(api_key=api_key, base_url=base_url, **kwargs)

    elif p == "google":
        from .google import GoogleProvider
        return GoogleProvider(api_key=api_key, base_url=base_url, **kwargs)

    elif p in ("openai", "deepseek", "mistral", "openrouter", "together", "fireworks", "ollama", "groq", "nvidia"):
        from .openai import OpenAIProvider
        from hiro.config import PROVIDER_BASE_URLS

        # Use configured or default base URL
        if not base_url and p in PROVIDER_BASE_URLS:
            base_url = PROVIDER_BASE_URLS[p]

        prov = OpenAIProvider(api_key=api_key, base_url=base_url, **kwargs)
        prov.name = p
        return prov

    else:
        # Fallback: treat as OpenAI-compatible with custom base_url
        from .openai import OpenAIProvider
        prov = OpenAIProvider(api_key=api_key, base_url=base_url, **kwargs)
        prov.name = p
        return prov


def clear_cache() -> None:
    _provider_cache.clear()
