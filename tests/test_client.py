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


# --- astream_text ------------------------------------------------------------

import asyncio as _asyncio

import httpx as _httpx

from llm_kit import astream_text


def _sse(lines):
    body = "".join(f"data: {l}\n\n" for l in lines).encode()
    return _httpx.MockTransport(lambda req: _httpx.Response(200, content=body,
                                headers={"content-type": "text/event-stream"}))


async def _collect(gen):
    return [c async for c in gen]


def test_astream_text_anthropic():
    t = _sse(['{"type":"message_start"}',
              '{"type":"content_block_delta","delta":{"type":"text_delta","text":"Hel"}}',
              '{"type":"content_block_delta","delta":{"type":"text_delta","text":"lo"}}',
              '{"type":"message_stop"}'])
    out = _asyncio.run(_collect(astream_text([{"role": "user", "content": "hi"}],
                                             provider="anthropic", api_key="k", transport=t)))
    assert out == ["Hel", "lo"]


def test_astream_text_openai_family():
    t = _sse(['{"choices":[{"delta":{"role":"assistant"}}]}',
              '{"choices":[{"delta":{"content":"a"}}]}',
              '{"choices":[{"delta":{"content":"b"}}]}', "[DONE]"])
    out = _asyncio.run(_collect(astream_text([{"role": "user", "content": "hi"}],
                                             provider="xai", api_key="k", transport=t)))
    assert out == ["a", "b"]


def test_astream_text_http_error():
    import pytest
    from llm_kit import LLMError
    t = _httpx.MockTransport(lambda req: _httpx.Response(401, text="nope"))
    with pytest.raises(LLMError):
        _asyncio.run(_collect(astream_text([{"role": "user", "content": "hi"}],
                                           provider="openai", api_key="k", transport=t)))


def test_astream_text_reports_usage_both_families():
    from llm_kit import Usage
    t = _sse(['{"type":"message_start","message":{"usage":{"input_tokens":42}}}',
              '{"type":"content_block_delta","delta":{"type":"text_delta","text":"x"}}',
              '{"type":"message_delta","usage":{"output_tokens":7}}'])
    u = Usage()
    _asyncio.run(_collect(astream_text([{"role": "user", "content": "hi"}], provider="anthropic",
                                       api_key="k", transport=t, usage=u)))
    assert (u.input_tokens, u.output_tokens) == (42, 7)

    seen = {}
    def handler(req):
        import json as _json
        seen["body"] = _json.loads(req.content)
        body = ("data: " + '{"choices":[{"delta":{"content":"y"}}]}' + "\n\n"
                "data: " + '{"choices":[],"usage":{"prompt_tokens":11,"completion_tokens":3}}' + "\n\n"
                "data: [DONE]\n\n").encode()
        return _httpx.Response(200, content=body)
    u = Usage()
    _asyncio.run(_collect(astream_text([{"role": "user", "content": "hi"}], provider="xai",
                                       api_key="k", transport=_httpx.MockTransport(handler), usage=u)))
    assert (u.input_tokens, u.output_tokens) == (11, 3)
    assert seen["body"]["stream_options"] == {"include_usage": True}


def test_ledger_roundtrip_has_no_app_field(tmp_path):
    from llm_kit import Usage, read_usage, record_usage
    path = tmp_path / "usage.jsonl"
    row = record_usage("xai", "grok-4.7", Usage(100, 20), path=path)
    record_usage("anthropic", "claude-haiku-4-5", Usage(1000, 500), path=path)
    rows = read_usage(path=path)
    assert [r["model"] for r in rows] == ["grok-4.7", "claude-haiku-4-5"]
    assert set(row) == {"ts", "provider", "model", "in", "out", "cost"}   # no app name, by design
    assert rows[1]["cost"] == round((1000 * 1.0 + 500 * 5.0) / 1_000_000, 6)
    assert read_usage(path=tmp_path / "missing.jsonl") == []


# --- vision input: canonical image parts -------------------------------------

import base64  # noqa: E402

from llm_kit import image_part, text_part  # noqa: E402

PNG = b"\x89PNG\r\n\x1a\n-not-really-a-png-"
B64 = base64.b64encode(PNG).decode()
VISION = [{"role": "user",
           "content": [image_part(PNG, "image/png"), text_part("describe this")]}]


def test_image_part_encodes_bytes_and_passes_base64_through():
    assert image_part(PNG, "image/png") == {"type": "image", "media_type": "image/png", "data": B64}
    assert image_part(B64, "image/jpeg")["data"] == B64


def test_anthropic_translates_image_parts_to_base64_blocks():
    _, out = c._anthropic_messages(VISION)
    assert out == [{"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": B64}},
        {"type": "text", "text": "describe this"},
    ]}]


def test_openai_translates_image_parts_to_image_url_data_uri():
    out = c._openai_messages(VISION)
    assert out == [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{B64}"}},
        {"type": "text", "text": "describe this"},
    ]}]


def test_plain_string_content_is_unchanged():
    msgs = [{"role": "user", "content": "hi"}]
    assert c._openai_messages(msgs) == [{"role": "user", "content": "hi"}]
    assert c._anthropic_messages(msgs)[1] == [{"role": "user", "content": "hi"}]


def test_system_and_assistant_parts_flatten_to_text():
    msgs = [{"role": "system", "content": [text_part("be terse"), text_part("be kind")]},
            {"role": "assistant", "content": [text_part("ok")]}]
    system, out = c._anthropic_messages(msgs)
    assert system == "be terse\nbe kind"
    assert c._openai_messages(msgs) == [
        {"role": "system", "content": "be terse\nbe kind"},
        {"role": "assistant", "content": "ok"},
    ]


@pytest.mark.parametrize("bad", [
    [{"type": "image", "media_type": "image/tiff", "data": B64}],   # unsupported type
    [{"type": "image", "media_type": "image/png", "data": ""}],     # no data
    [{"type": "audio", "data": "x"}],                               # unknown part
    ["just a string in the list"],                                  # not a part dict
])
def test_bad_parts_raise_in_both_families(bad):
    msgs = [{"role": "user", "content": bad}]
    for fn in (c._anthropic_messages, c._openai_messages):
        with pytest.raises(LLMError):
            fn(msgs)


def test_images_outside_user_turns_raise():
    msgs = [{"role": "assistant", "content": [image_part(PNG, "image/png")]}]
    for fn in (c._anthropic_messages, c._openai_messages):
        with pytest.raises(LLMError, match="user turns"):
            fn(msgs)


def test_end_to_end_xai_sends_image_and_reads_usage(monkeypatch):
    cap: dict = {}
    _mock_client(monkeypatch, cap, {
        "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 900, "completion_tokens": 40},
    })
    turn = chat_with_tools(VISION, provider="xai", model="grok-4.7", api_key="k")
    content = cap["body"]["messages"][0]["content"]
    assert content[0]["image_url"]["url"] == f"data:image/png;base64,{B64}"
    assert content[1] == {"type": "text", "text": "describe this"}
    assert (turn.usage.input_tokens, turn.usage.output_tokens) == (900, 40)


def test_end_to_end_anthropic_sends_image_block(monkeypatch):
    cap: dict = {}
    _mock_client(monkeypatch, cap, {"content": [{"type": "text", "text": "a cat"}],
                                    "usage": {"input_tokens": 3, "output_tokens": 1}})
    turn = chat_with_tools(VISION, provider="anthropic", api_key="k")
    assert cap["body"]["messages"][0]["content"][0]["source"]["data"] == B64
    assert turn.text == "a cat"


@pytest.mark.asyncio
async def test_achat_sends_image_too(monkeypatch):
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    real = httpx.AsyncClient
    monkeypatch.setattr(c.httpx, "AsyncClient",
                        lambda *a, **kw: real(*a, **{**kw, "transport": httpx.MockTransport(handler)}))
    turn = await c.achat_with_tools(VISION, provider="xai", api_key="k")
    assert seen["body"]["messages"][0]["content"][0]["type"] == "image_url"
    assert turn.text == "ok"


# --- JSON output ---------------------------------------------------------------

def test_json_output_sets_response_format_for_openai_family_only():
    _, _, xai = c._build_request("xai", "grok-4.7", "k", VISION, None, 256, 0.0, json_output=True)
    assert xai["response_format"] == {"type": "json_object"}
    _, _, anth = c._build_request("anthropic", "claude-haiku-4-5", "k", VISION, None, 256, 0.0,
                                  json_output=True)
    assert "response_format" not in anth            # no such switch on Messages
    _, _, plain = c._build_request("xai", "grok-4.7", "k", VISION, None, 256, 0.0)
    assert "response_format" not in plain           # opt-in; old positional callers unaffected


def test_end_to_end_json_output_flag_reaches_the_wire(monkeypatch):
    cap: dict = {}
    _mock_client(monkeypatch, cap, {"choices": [{"message": {"content": "{}"}}]})
    chat_with_tools(VISION, provider="xai", api_key="k", json_output=True)
    assert cap["body"]["response_format"] == {"type": "json_object"}
