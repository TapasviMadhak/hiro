"""Tests for Hiro cybersecurity commands and utilities."""
import pytest
import base64
import json
import hashlib
import os
from hiro.app import HiroApp
from hiro.config import Settings


@pytest.mark.asyncio
async def test_burp_status_and_proxy_toggle():
    settings = Settings()
    app = HiroApp(settings)

    # Test /burp status
    await app.handle_command("/burp status")

    # Test /burp proxy toggle
    await app.handle_command("/burp proxy http://127.0.0.1:8080")
    assert os.environ.get("HTTP_PROXY") == "http://127.0.0.1:8080"
    assert os.environ.get("HTTPS_PROXY") == "http://127.0.0.1:8080"

    await app.handle_command("/burp proxy off")
    assert "HTTP_PROXY" not in os.environ
    assert "HTTPS_PROXY" not in os.environ


@pytest.mark.asyncio
async def test_decode_command():
    settings = Settings()
    app = HiroApp(settings)

    # Base64 test
    payload = "admin' OR 1=1--"
    b64 = base64.b64encode(payload.encode()).decode()
    await app.handle_command(f"/decode {b64}")

    # JWT test
    h = base64.b64encode(b'{"alg":"none","typ":"JWT"}').decode().rstrip("=")
    p = base64.b64encode(b'{"user":"admin","role":"root"}').decode().rstrip("=")
    jwt = f"{h}.{p}."
    await app.handle_command(f"/decode {jwt}")


@pytest.mark.asyncio
async def test_hash_command():
    settings = Settings()
    app = HiroApp(settings)
    await app.handle_command("/hash password123")


@pytest.mark.asyncio
async def test_diff_command(tmp_path):
    settings = Settings()
    app = HiroApp(settings)

    f1 = tmp_path / "v1.py"
    f2 = tmp_path / "v2.py"
    f1.write_text("def test(): return 1\n")
    f2.write_text("def test(): return 2\n")

    await app.handle_command(f"/diff {f1} {f2}")
