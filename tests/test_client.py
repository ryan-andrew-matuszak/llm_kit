"""Offline tests for llm_kit — no network, no keys.

Pure translate/parse helpers are tested directly; the end-to-end path is driven
through an httpx MockTransport so we can assert the exact request each provider
receives and how its canned reply parses into a ChatTurn.
"""
import json

import httpx
import pytest

import llm_kit.client as c
from llm_kit import (
    ChatTurn,
    LLMError,
    ToolCall,
    chat_with_tools,
    estimate_cost,
    tool_calls_message,
    tool_result_message,
)

TOOLS = [{"name": "turn_off", "description": "turn a device off",
          "parameters": {"type": "object", "properties": {"id": {"type": "string"}},
                         "required": ["id"]}}]


# --- pure translation --------------------------------------------------------

def test_anthropic_tools_shape():
    out = c._anthropic_tools(TOOLS)
    assert out[0]["name"] == "turn_off"
    assert out[0]["input_schema"]["required"] == ["id"]


def test_openai_tools_shape():
    out = c._openai_tools(TOOLS)
    assert out[0]["type"] == "function"
    assert out[0]["function"]["name"] == "turn_off"


def test_anthropic_messages_extracts_system_and_merges_tool_results():
    msgs = [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "off please"},
        tool_calls_message([ToolCall("t1", "turn_off", {"id": "tv"})]),
        tool_result_message("t1", "ok"),
        tool_result_message("t2", "ok2"),  # consecutive -> one user turn
    ]
    system, out = c._anthropic_messages(msgs)
    assert system == "be terse"
    # the two tool results collapse into a single user message with two blocks
    tool_users = [m for m in out if m["role"] == "user" and isinstance(m["content"], list)]
    assert len(tool_users) == 1
    assert [b["type"] for b in tool_users[0]["content"]] == ["tool_result", "tool_result"]
    # the assistant tool-call turn round-trips into a tool_use block
    assistant = [m for m in out if m["role"] == "assistant"][0]
    assert assistant["content"][0]["type"] == "tool_use"
    assert assistant["content"][0]["input"] == {"id": "tv"}


def test_openai_messages_serialises_tool_call_args_as_json_string():
    msgs = [tool_calls_message([ToolCall("t1", "turn_off", {"id": "tv"})])]
    out = c._openai_messages(msgs)
    args = out[0]["tool_calls"][0]["function"]["arguments"]
    assert json.loads(args) == {"id": "tv"}


# --- response parsing --------------------------------------------------------

def test_anthropic_parse_tool_use():
    data = {
        "content": [
            {"type": "text", "text": "sure"},
            {"type": "tool_use", "id": "abc", "name": "turn_off", "input": {"id": "tv"}},
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 12, "output_tokens": 7},
    }
    turn = c._anthropic_parse(data)
    assert turn.text == "sure"
    assert turn.tool_calls[0].name == "turn_off"
    assert turn.tool_calls[0].arguments == {"id": "tv"}
    assert turn.usage.input_tokens == 12 and turn.usage.output_tokens == 7


def test_openai_parse_tool_calls():
    data = {
        "choices": [{"message": {"content": None, "tool_calls": [
            {"id": "abc", "type": "function",
             "function": {"name": "turn_off", "arguments": '{"id": "tv"}'}},
        ]}, "finish_reason": "tool_calls"}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 3},
    }
    turn = c._openai_parse(data)
    assert turn.tool_calls[0].arguments == {"id": "tv"}
    assert turn.usage.input_tokens == 20 and turn.usage.output_tokens == 3


def test_openai_parse_bad_json_args_degrades_to_empty():
    data = {"choices": [{"message": {"tool_calls": [
        {"id": "x", "function": {"name": "f", "arguments": "not json"}}]}}]}
    assert c._openai_parse(data).tool_calls[0].arguments == {}


# --- cost --------------------------------------------------------------------

def test_estimate_cost_known_and_unknown():
    assert estimate_cost(c.Usage(1_000_000, 1_000_000), "claude-haiku-4-5") == 6.0
    assert estimate_cost(c.Usage(1_000_000, 0), "mystery-model") == 0.0


# --- end to end via mock transport -------------------------------------------

def _mock_client(monkeypatch, capture: dict, response_json: dict):
    def handler(request: httpx.Request) -> httpx.Response:
        capture["url"] = str(request.url)
        capture["headers"] = dict(request.headers)
        capture["body"] = json.loads(request.content)
        return httpx.Response(200, json=response_json)

    real_client = httpx.Client

    def fake_client(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(c.httpx, "Client", fake_client)


def test_end_to_end_anthropic(monkeypatch):
    cap: dict = {}
    _mock_client(monkeypatch, cap, {
        "content": [{"type": "tool_use", "id": "1", "name": "turn_off", "input": {"id": "tv"}}],
        "stop_reason": "tool_use", "usage": {"input_tokens": 5, "output_tokens": 2},
    })
    turn = chat_with_tools([{"role": "user", "content": "tv off"}], TOOLS,
                           provider="anthropic", api_key="k")
    assert cap["url"].endswith("/v1/messages")
    assert cap["headers"]["x-api-key"] == "k"
    assert cap["headers"]["anthropic-version"] == c.ANTHROPIC_VERSION
    assert cap["body"]["model"] == "claude-haiku-4-5"  # provider default
    assert turn.tool_calls[0].name == "turn_off"


def test_end_to_end_xai_uses_openai_shape(monkeypatch):
    cap: dict = {}
    _mock_client(monkeypatch, cap, {
        "choices": [{"message": {"content": "done"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 4, "completion_tokens": 1},
    })
    turn = chat_with_tools([{"role": "user", "content": "hi"}], TOOLS,
                           provider="xai", model="grok-3-mini", api_key="k")
    assert "api.x.ai" in cap["url"] and cap["url"].endswith("/chat/completions")
    assert cap["headers"]["authorization"] == "Bearer k"
    assert cap["body"]["tools"][0]["type"] == "function"
    assert turn.text == "done"


def test_missing_key_raises(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(LLMError):
        chat_with_tools([{"role": "user", "content": "x"}], provider="openai", api_key=None)


def test_unknown_provider_raises():
    with pytest.raises(LLMError):
        chat_with_tools([{"role": "user", "content": "x"}], provider="nope", api_key="k")


# --- server tools (web_search) + pause_turn ----------------------------------

def test_server_tool_passthrough_and_openai_skip():
    server = {"type": "web_search_20250305", "name": "web_search", "max_uses": 5}
    a = c._anthropic_tools([server] + TOOLS)
    assert a[0] == server                       # passed through untouched
    assert a[1]["name"] == "turn_off"           # neutral tool still converted
    o = c._openai_tools([server] + TOOLS)
    assert [t["function"]["name"] for t in o] == ["turn_off"]   # server tool dropped


@pytest.mark.asyncio
async def test_anthropic_pause_turn_continues(monkeypatch):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(200, json={
                "content": [{"type": "text", "text": "searching…"}],
                "stop_reason": "pause_turn", "usage": {"input_tokens": 10, "output_tokens": 2}})
        return httpx.Response(200, json={
            "content": [{"type": "text", "text": "The Padres won 5-3."}],
            "stop_reason": "end_turn", "usage": {"input_tokens": 8, "output_tokens": 6}})

    real = httpx.AsyncClient

    def fake(*a, **k):
        k["transport"] = httpx.MockTransport(handler)
        return real(*a, **k)

    monkeypatch.setattr(c.httpx, "AsyncClient", fake)
    turn = await c.achat_with_tools(
        [{"role": "user", "content": "who won the game"}],
        [{"type": "web_search_20250305", "name": "web_search"}],
        provider="anthropic", api_key="k")
    assert calls["n"] == 2                       # paused, then continued
    assert turn.text == "The Padres won 5-3." and turn.stop_reason == "end_turn"
    assert turn.usage.input_tokens == 18 and turn.usage.output_tokens == 8   # summed
