"""Tests for Hiro configuration and token efficiency settings."""
from pathlib import Path
from hiro.config import (
    Settings,
    load_settings,
    save_settings,
    get_api_key,
    get_model_info,
    BUILTIN_MODELS,
    PROVIDER_BASE_URLS,
)


def test_default_settings_no_preprompt():
    """Verify that by default Hiro has zero pre-prompts to prevent eating tokens."""
    settings = Settings()
    assert settings.system_prompt == ""
    assert settings.system_prompt_file == ""
    assert settings.agent.token_budget == 200000
    assert settings.agent.auto_compact is True


def test_builtin_models_coverage():
    """Verify built-in models include all major providers."""
    providers = {info["provider"] for info in BUILTIN_MODELS.values()}
    assert "anthropic" in providers
    assert "openai" in providers
    assert "google" in providers
    assert "groq" in providers
    assert "deepseek" in providers
    assert "mistral" in providers
    assert "ollama" in providers


def test_get_model_info():
    info = get_model_info("claude-opus-4-5")
    assert info is not None
    assert info["provider"] == "anthropic"

    ollama_info = get_model_info("ollama/my-custom-model")
    assert ollama_info is not None
    assert ollama_info["provider"] == "ollama"


def test_api_key_resolution(monkeypatch):
    settings = Settings()
    settings.api_keys["anthropic"] = "sk-ant-config-key"
    assert get_api_key("anthropic", settings) == "sk-ant-config-key"

    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-openai-key")
    assert get_api_key("openai", settings) == "sk-env-openai-key"


def test_settings_save_and_load(tmp_path):
    config_file = tmp_path / "config.toml"
    s = Settings(model="gpt-4o", provider="openai")
    s.api_keys["openai"] = "test-key"
    save_settings(s, config_file)

    loaded = load_settings(config_file)
    assert loaded.model == "gpt-4o"
    assert loaded.provider == "openai"
    assert loaded.api_keys.get("openai") == "test-key"
