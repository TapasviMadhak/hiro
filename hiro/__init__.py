"""
Hiro - Claude Code-grade terminal AI coding assistant.

Multi-provider LLM support, MCP integration, cybersecurity tooling.
Token-efficient by design. No bloated pre-prompts.
"""
import sys

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

__version__ = "0.1.0"
__author__ = "Hiro"
