"""Tests for Hiro slash commands."""
import pytest
import base64
import json
from hiro.commands.registry import parse_command, get_command, list_commands
import hiro.commands  # Register all commands


def test_command_parsing():
    assert parse_command("/model gpt-4o") == ("model", "gpt-4o")
    assert parse_command("/burp status") == ("burp", "status")
    assert parse_command("/help") == ("help", "")
    assert parse_command("not a command") is None


def test_registered_command_lookup():
    cmd = get_command("burp")
    assert cmd is not None
    assert cmd.name == "burp"
    assert "security" in cmd.category

    bp = get_command("bp")
    assert bp is not None
    assert bp.name == "burp"


def test_security_commands_registered():
    cmds = {c.name for c in list_commands()}
    expected = {
        "burp", "audit", "scan", "sqli", "xss", "secrets",
        "payload", "decode", "hash", "headers", "cors",
    }
    for exp in expected:
        assert exp in cmds, f"Expected security command {exp} not registered"
