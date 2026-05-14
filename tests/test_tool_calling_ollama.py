"""Unit tests for content-to-tool-call recovery."""
from __future__ import annotations

import json

from langchain_core.messages import AIMessage

from app.agent.tool_calling_ollama import (
    _extract_tool_call,
    _extract_tool_calls,
    _rewrite,
)


def test_extracts_plain_json():
    tc = _extract_tool_call('{"name": "list_files", "arguments": {"path": "."}}')
    assert tc is not None
    assert tc["name"] == "list_files"
    assert tc["args"] == {"path": "."}
    assert tc["type"] == "tool_call"


def test_extracts_fenced_json():
    fenced = "```json\n" + json.dumps({"name": "read_file", "arguments": {"path": "x"}}) + "\n```"
    tc = _extract_tool_call(fenced)
    assert tc is not None
    assert tc["name"] == "read_file"


def test_returns_none_for_prose():
    assert _extract_tool_call("Sure, I'll list files for you.") is None


def test_returns_none_for_invalid_json():
    assert _extract_tool_call("{not real json}") is None


def test_returns_none_when_no_arguments():
    assert _extract_tool_call('{"name": "x"}') is None


def test_rewrite_passthrough_when_already_has_tool_calls():
    msg = AIMessage(
        content="",
        tool_calls=[{"name": "x", "args": {}, "id": "id-1", "type": "tool_call"}],
    )
    out = _rewrite(msg)
    assert out is msg


def test_rewrite_extracts_from_content():
    msg = AIMessage(content='{"name": "list_files", "arguments": {"path": "."}}')
    out = _rewrite(msg)
    assert out.tool_calls and out.tool_calls[0]["name"] == "list_files"
    assert out.content == ""


def test_rewrite_leaves_prose_alone():
    msg = AIMessage(content="Here is your answer.")
    out = _rewrite(msg)
    assert out is msg


def test_extracts_multiple_calls_in_one_blob():
    blob = (
        '{"name": "recent_commits", "arguments": {"limit": 1}}\n\n'
        '{"name": "current_branch", "arguments": {}}'
    )
    calls = _extract_tool_calls(blob)
    assert [c["name"] for c in calls] == ["recent_commits", "current_branch"]


def test_rewrite_multi_calls():
    msg = AIMessage(
        content=(
            '{"name": "recent_commits", "arguments": {"limit": 1}}\n'
            '{"name": "current_branch", "arguments": {}}'
        )
    )
    out = _rewrite(msg)
    assert len(out.tool_calls) == 2
    assert out.content == ""
