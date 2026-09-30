"""
Security utilities for Hiro.

Provides:
- OS Keyring integration for storing API keys securely
- Secret masking (redacts keys from output/logs)
- Audit helpers
"""
from __future__ import annotations

import re
import os
import logging
from typing import Any

logger = logging.getLogger(__name__)

# --- Keyring Service Name ---

_KEYRING_SERVICE = "hiro-ai"

# --- Keyring Operations ---

def _keyring_available() -> bool:
    """Check whether a working keyring backend is available."""
    try:
        import keyring
        backend = keyring.get_keyring()
        cls_name = type(backend).__name__
        return "Fail" not in cls_name and "Null" not in cls_name
    except Exception:
        return False


def keyring_set(provider: str, key: str) -> bool:
    """Store an API key in the OS keyring. Returns True on success."""
    try:
        import keyring
        keyring.set_password(_KEYRING_SERVICE, provider, key)
        return True
    except Exception as e:
        logger.warning(f"Keyring write failed for {provider}: {e}")
        return False


def keyring_get(provider: str) -> str:
    """Retrieve an API key from the OS keyring. Returns empty string on failure."""
    try:
        import keyring
        return keyring.get_password(_KEYRING_SERVICE, provider) or ""
    except Exception as e:
        logger.debug(f"Keyring read failed for {provider}: {e}")
        return ""


def keyring_delete(provider: str) -> bool:
    """Remove an API key from the OS keyring."""
    try:
        import keyring
        keyring.delete_password(_KEYRING_SERVICE, provider)
        return True
    except Exception:
        return False


def list_keyring_providers() -> list[str]:
    """Return all provider names stored in the keyring (best-effort)."""
    from hiro.config import PROVIDER_ENV_KEYS
    found = []
    for provider in PROVIDER_ENV_KEYS:
        if keyring_get(provider):
            found.append(provider)
    return found


# --- Secret Masking ---

_SECRET_PATTERNS: list[re.Pattern] = [
    re.compile(r"nvapi-[A-Za-z0-9_\-]{20,60}"),
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,100}"),
    re.compile(r"sk-[A-Za-z0-9_\-]{20,100}"),
    re.compile(r"(?i)(api[_-]?key|token|secret)[=:]\s*['\"]?([A-Za-z0-9_\-\.]{16,})['\"]?"),
    re.compile(r"(?i)bearer\s+([A-Za-z0-9_\-\.]{16,})"),
]

_PLACEHOLDER = "[REDACTED]"


def mask_secrets(text: str, extra_secrets: list[str] | None = None) -> str:
    """Replace known secret patterns in text with [REDACTED]."""
    result = text
    if extra_secrets:
        for secret in extra_secrets:
            if secret and len(secret) >= 8:
                result = result.replace(secret, _PLACEHOLDER)
    for pattern in _SECRET_PATTERNS:
        result = pattern.sub(_PLACEHOLDER, result)
    return result


def get_all_active_keys() -> list[str]:
    """Collect all currently-loaded API keys for use with mask_secrets."""
    keys: list[str] = []
    try:
        from hiro.config import get_settings, PROVIDER_ENV_KEYS
        s = get_settings()
        keys.extend(v for v in s.api_keys.values() if v)
        for env_var in PROVIDER_ENV_KEYS.values():
            val = os.environ.get(env_var, "")
            if val:
                keys.append(val)
    except Exception:
        pass
    return keys


class SecretRedactingHandler(logging.Handler):
    """A logging handler wrapper that redacts secrets before emitting."""

    def __init__(self, inner: logging.Handler) -> None:
        super().__init__()
        self._inner = inner

    def emit(self, record: logging.LogRecord) -> None:
        original = record.getMessage()
        redacted = mask_secrets(original, get_all_active_keys())
        record.msg = redacted
        record.args = ()
        self._inner.emit(record)


def install_log_redaction() -> None:
    """Wrap the root logger handlers so secrets are never logged."""
    root = logging.getLogger()
    new_handlers = []
    for h in root.handlers:
        if not isinstance(h, SecretRedactingHandler):
            new_handlers.append(SecretRedactingHandler(h))
        else:
            new_handlers.append(h)
    root.handlers = new_handlers


# --- Pre-commit Hook ---

def install_precommit_hook(repo_root: str = ".") -> bool:
    """Write a pre-commit hook into .git/hooks/. Returns True on success."""
    import stat
    from pathlib import Path
    hook_path = Path(repo_root) / ".git" / "hooks" / "pre-commit"
    hook_script = r'''#!/usr/bin/env python3
"""Pre-commit hook: scan staged files for leaked API keys."""
import subprocess, sys, re

PATTERNS = [
    r"nvapi-[A-Za-z0-9_\-]{20,}",
    r"sk-ant-[A-Za-z0-9_\-]{20,}",
    r"sk-[A-Za-z0-9_\-]{20,}",
    r"AIza[A-Za-z0-9_\-]{35}",
]

result = subprocess.run(
    ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
    capture_output=True, text=True,
)
staged = [f for f in result.stdout.splitlines()
          if any(f.endswith(ext) for ext in (".py", ".toml", ".json", ".env", ".yaml", ".yml"))]

found = []
for filepath in staged:
    try:
        with open(filepath, encoding="utf-8", errors="ignore") as fh:
            content = fh.read()
        for pat in PATTERNS:
            for m in re.finditer(pat, content):
                val = m.group(0)
                if len(val) >= 24:
                    found.append((filepath, val[:12] + "..."))
    except Exception:
        pass

if found:
    print("\033[31m[hiro pre-commit] POTENTIAL SECRET LEAK DETECTED:\033[0m")
    for f, v in found:
        print(f"  {f}: {v}")
    print("\nIf false positive, commit with --no-verify")
    sys.exit(1)
'''
    try:
        hook_path.parent.mkdir(parents=True, exist_ok=True)
        hook_path.write_text(hook_script, encoding="utf-8")
        hook_path.chmod(hook_path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        return True
    except Exception as e:
        logger.warning(f"Could not install pre-commit hook: {e}")
        return False
