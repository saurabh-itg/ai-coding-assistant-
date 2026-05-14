"""Tests for configuration loading."""
from __future__ import annotations

from app.config import Settings


def test_terminal_allowlist_set_parses():
    s = Settings(terminal_allowlist="git, python ,pytest ,, ")
    assert s.terminal_allowlist_set == {"git", "python", "pytest"}


def test_cors_origins_wildcard():
    s = Settings(cors_origins="*")
    assert s.cors_origins_list == ["*"]


def test_cors_origins_list():
    s = Settings(cors_origins="http://a.com,http://b.com")
    assert s.cors_origins_list == ["http://a.com", "http://b.com"]
