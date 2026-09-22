"""Provider-agnostic *chat-with-tools* client for hosted LLMs.

One tiny seam — ``chat_with_tools(messages, tools)`` — over three cloud
providers (Anthropic, OpenAI, xAI), so the model behind an app is a config flag,
not a rewrite. Deliberately **raw REST over httpx**, no vendor SDKs: the whole
kit depends only on ``httpx``, which keeps it light enough to install on a
Raspberry Pi (and anywhere else). Adding a provider is one entry in ``PROVIDERS``
plus, if its wire format is new, one pair of translate/parse helpers.

Design:
  * **Canonical messages** (provider-neutral) flow through the caller's tool
    loop; each adapter translates them to that provider's wire shape and parses
    the reply back into a `ChatTurn`. So the loop never branches on provider.
  * Two wire families cover all three providers: Anthropic Messages
    (``/v1/messages``, tool-use blocks) and OpenAI Chat Completions
    (``/chat/completions``, function tools) — xAI is OpenAI-compatible.
  * Both **sync** (`chat_with_tools`) and **async** (`achat_with_tools`) entry
    points share the same pure request-builders and response-parsers, so tests
    can exercise the translation/parse logic with no network at all.

Canonical message shapes (plain dicts):
  * text turn:       ``{"role": "system"|"user"|"assistant", "content": str}``
  * tool-call turn:  ``{"role": "assistant", "content": str|None,
                        "tool_calls": [{"id", "name", "arguments": dict}, ...]}``
  * tool result:     ``{"role": "tool", "tool_call_id": str, "name": str,
                        "content": str}``
Use `tool_calls_message()` / `tool_result_message()` to build the last two.

A user turn's ``content`` may also be a **list of parts** (vision input):
  * ``{"type": "text", "text": str}``
  * ``{"type": "image", "media_type": "image/jpeg", "data": <base64 str>}``
`image_part(raw_bytes, media_type)` builds the second. Each adapter translates
them: Anthropic ``image`` blocks, OpenAI/xAI ``image_url`` data URIs. Images are
user-turn only; the model must support vision (that's the caller's choice).

Config resolves from args first, then env: ``LLM_PROVIDER``, ``LLM_MODEL``, and
the provider key (``ANTHROPIC_API_KEY`` / ``OPENAI_API_KEY`` / ``XAI_API_KEY``).
`estimate_cost()` turns reported token usage into dollars against `PRICES` so a
caller can log spend and hold a budget.
"""
from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass, field
from typing import Any

import httpx

ANTHROPIC_VERSION = "2023-06-01"

#: provider -> transport defaults. `family` picks the wire adapter. `default_model`
#: is a sane, cheap starting point; the caller should still pin one via config.
PROVIDERS: dict[str, dict[str, str]] = {
    "anthropic": {
        "family": "anthropic",
        "base_url": "https://api.anthropic.com",
        "key_env": "ANTHROPIC_API_KEY",
        "default_model": "claude-haiku-4-5",
    },
    "openai": {
        "family": "openai",
        "base_url": "https://api.openai.com/v1",
        "key_env": "OPENAI_API_KEY",
        "default_model": "gpt-4o-mini",
    },
    "xai": {
        "family": "openai",
        "base_url": "https://api.x.ai/v1",
        "key_env": "XAI_API_KEY",
        "default_model": "grok-3-mini",
    },
}

#: model id -> (input $/1M tokens, output $/1M tokens). Missing models cost $0
#: (estimate degrades to free rather than lying); extend as needed.
PRICES: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-opus-5": (5.00, 25.00),
    "gpt-4o-mini": (0.15, 0.60),
    "grok-3-mini": (0.30, 0.50),
}


class LLMError(RuntimeError):
    """Raised on a transport failure, HTTP error, or missing credentials."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class ChatTurn:
    """One assistant turn. `tool_calls` non-empty ⇒ the model wants tools run;
    else `text` is the final answer. `raw` is the untouched provider JSON."""
    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    stop_reason: str | None = None
    raw: dict = field(default_factory=dict)


# --- canonical message builders ----------------------------------------------


def tool_calls_message(tool_calls: list[ToolCall], text: str | None = None) -> dict:
    """Canonical assistant turn that requested tools (echo it back before the
    matching tool results, so the model sees its own call)."""
    return {
        "role": "assistant",
        "content": text,
        "tool_calls": [
            {"id": c.id, "name": c.name, "arguments": c.arguments} for c in tool_calls
        ],
    }


def tool_result_message(tool_call_id: str, content: str, name: str = "") -> dict:
    """Canonical result of running one tool the model asked for."""
    return {"role": "tool", "tool_call_id": tool_call_id, "name": name, "content": content}


# --- multimodal content parts --------------------------------------------------

#: Image types the wire families accept in common. A given model may take fewer
#: (xAI's vision models take only jpeg/png) — the caller knows its model.
IMAGE_TYPES = ("image/jpeg", "image/png", "image/gif", "image/webp")


def text_part(text: str) -> dict:
    """Canonical text part of a multimodal ``content`` list."""
    return {"type": "text", "text": text}


def image_part(data: bytes | str, media_type: str) -> dict:
    """Canonical image part. `data` is raw bytes (base64-encoded here) or an
    already-base64 string."""
    if isinstance(data, (bytes, bytearray)):
        data = base64.b64encode(bytes(data)).decode("ascii")
    return {"type": "image", "media_type": media_type, "data": data}


def _parts(content: Any, role: str) -> list[dict]:
    """Validate a message's ``content`` into a list of canonical parts. A plain
    string (or None) is one text part. Raises `LLMError` on a malformed part or on
    an image anywhere but a user turn."""
    if content is None or isinstance(content, str):
        return [text_part(content or "")]
    if not isinstance(content, list):
        raise LLMError(f"message content must be a string or a list of parts, got {type(content).__name__}")
    out = []
    for part in content:
        kind = part.get("type") if isinstance(part, dict) else None
        if kind == "text" and isinstance(part.get("text"), str):
            out.append(part)
        elif kind == "image":
            if role != "user":
                raise LLMError("images are only supported in user turns")
            if part.get("media_type") not in IMAGE_TYPES:
                raise LLMError(f"unsupported image type {part.get('media_type')!r}; "
                               f"known: {', '.join(IMAGE_TYPES)}")
            if not part.get("data") or not isinstance(part["data"], str):
                raise LLMError("image part has no base64 data")
            out.append(part)
        else:
            raise LLMError(f"unknown content part: {part!r:.80}")
    return out


def _text_of(content: Any, role: str) -> str:
    """Just the text of a message's content (parts joined) — for turns whose wire
    slot is a plain string (system, assistant)."""
    return "\n".join(p["text"] for p in _parts(content, role) if p["type"] == "text")


# --- provider resolution -----------------------------------------------------


def resolve_provider(provider: str | None) -> str:
    prov = (provider or os.environ.get("LLM_PROVIDER") or "anthropic").strip().lower()
    if prov not in PROVIDERS:
        raise LLMError(f"unknown provider {prov!r}; known: {', '.join(PROVIDERS)}")
    return prov


def resolve_model(provider: str, model: str | None) -> str:
    return (model or os.environ.get("LLM_MODEL") or PROVIDERS[provider]["default_model"]).strip()


def resolve_key(provider: str, api_key: str | None) -> str:
    key = api_key or os.environ.get(PROVIDERS[provider]["key_env"], "")
    if not key:
        raise LLMError(
            f"no API key for provider {provider!r}: pass api_key= or set "
            f"${PROVIDERS[provider]['key_env']}"
        )
    return key


def price_for(model: str) -> tuple[float, float]:
    """(input, output) $/1M for a model. Falls back to the longest PRICES key the
    model id starts with, so a dated id (claude-haiku-4-5-20251001) still matches
    its base entry (claude-haiku-4-5). Unknown ⇒ (0.0, 0.0)."""
    if model in PRICES:
        return PRICES[model]
    hits = [k for k in PRICES if model.startswith(k)]
    return PRICES[max(hits, key=len)] if hits else (0.0, 0.0)


def estimate_cost(usage: Usage, model: str) -> float:
    """Dollar cost of a turn's tokens against `PRICES` (0.0 for unknown models)."""
    in_rate, out_rate = price_for(model)
    return (usage.input_tokens * in_rate + usage.output_tokens * out_rate) / 1_000_000


# --- Anthropic (Messages API) wire adapter -----------------------------------


def _anthropic_tools(tools: list[dict] | None) -> list[dict]:
    """Provider-neutral ``{name, description, parameters}`` -> Anthropic tools.
    A tool dict carrying a ``type`` (e.g. a server tool like
    ``{"type": "web_search_20250305", "name": "web_search"}``) is passed through
    untouched — Anthropic runs it server-side."""
    out = []
    for t in tools or []:
        if t.get("type"):
            out.append(dict(t))
        else:
            out.append({
                "name": t["name"],
                "description": t.get("description", ""),
                "input_schema": t.get("parameters") or {"type": "object", "properties": {}},
            })
    return out


def _anthropic_block(part: dict) -> dict:
    if part["type"] == "image":
        return {"type": "image", "source": {"type": "base64",
                                            "media_type": part["media_type"], "data": part["data"]}}
    return {"type": "text", "text": part["text"]}


def _anthropic_messages(messages: list[dict]) -> tuple[str | None, list[dict]]:
    """Translate canonical messages to (system, messages) for /v1/messages.
    Consecutive tool results are merged into one user message (Anthropic wants
    each ``tool_result`` in a user turn)."""
    system_parts: list[str] = []
    out: list[dict] = []
    for m in messages:
        role = m["role"]
        if role == "system":
            text = _text_of(m.get("content"), role)
            if text:
                system_parts.append(text)
        elif role == "tool":
            block = {"type": "tool_result", "tool_use_id": m["tool_call_id"],
                     "content": m.get("content", "")}
            if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
        elif role == "assistant" and m.get("tool_calls"):
            content: list[dict] = []
            if m.get("content"):
                content.append({"type": "text", "text": m["content"]})
            for c in m["tool_calls"]:
                content.append({"type": "tool_use", "id": c["id"], "name": c["name"],
                                "input": c.get("arguments") or {}})
            out.append({"role": "assistant", "content": content})
        elif isinstance(m.get("content"), list):  # multimodal parts
            out.append({"role": role, "content": [_anthropic_block(p) for p in _parts(m["content"], role)]})
        else:  # plain user/assistant text
            out.append({"role": role, "content": m.get("content") or ""})
    system = "\n\n".join(system_parts) if system_parts else None
    return system, out


def _anthropic_payload(messages, tools, model, max_tokens, temperature) -> dict:
    system, msgs = _anthropic_messages(messages)
    payload: dict[str, Any] = {"model": model, "max_tokens": max_tokens,
                               "temperature": temperature, "messages": msgs}
    if system:
        payload["system"] = system
    tl = _anthropic_tools(tools)
    if tl:
        payload["tools"] = tl
    return payload


def _anthropic_parse(data: dict) -> ChatTurn:
    text_parts, calls = [], []
    for block in data.get("content", []) or []:
        if block.get("type") == "text":
            text_parts.append(block.get("text", ""))
        elif block.get("type") == "tool_use":
            calls.append(ToolCall(id=block.get("id", ""), name=block.get("name", ""),
                                  arguments=block.get("input") or {}))
    u = data.get("usage") or {}
    return ChatTurn(
        text="".join(text_parts) or None,
        tool_calls=calls,
        usage=Usage(int(u.get("input_tokens", 0)), int(u.get("output_tokens", 0))),
        stop_reason=data.get("stop_reason"),
        raw=data,
    )


# --- OpenAI / xAI (Chat Completions) wire adapter ----------------------------


def _openai_tools(tools: list[dict] | None) -> list[dict]:
    out = []
    for t in tools or []:
        if t.get("type"):
            continue  # a provider-native server tool (e.g. Anthropic web_search) — not for OpenAI
        out.append({"type": "function", "function": {
            "name": t["name"],
            "description": t.get("description", ""),
            "parameters": t.get("parameters") or {"type": "object", "properties": {}},
        }})
    return out


def _openai_part(part: dict) -> dict:
    if part["type"] == "image":
        return {"type": "image_url",
                "image_url": {"url": f"data:{part['media_type']};base64,{part['data']}"}}
    return {"type": "text", "text": part["text"]}


def _openai_messages(messages: list[dict]) -> list[dict]:
    out: list[dict] = []
    for m in messages:
        role = m["role"]
        if role == "tool":
            out.append({"role": "tool", "tool_call_id": m["tool_call_id"],
                        "content": m.get("content", "")})
        elif role == "assistant" and m.get("tool_calls"):
            out.append({
                "role": "assistant",
                "content": m.get("content"),
                "tool_calls": [{
                    "id": c["id"], "type": "function",
                    "function": {"name": c["name"],
                                 "arguments": json.dumps(c.get("arguments") or {})},
                } for c in m["tool_calls"]],
            })
        elif role == "user" and isinstance(m.get("content"), list):  # multimodal parts
            out.append({"role": role, "content": [_openai_part(p) for p in _parts(m["content"], role)]})
        elif isinstance(m.get("content"), list):                     # system/assistant: text only
            out.append({"role": role, "content": _text_of(m["content"], role)})
        else:
            out.append({"role": role, "content": m.get("content") or ""})
    return out


def _openai_payload(messages, tools, model, max_tokens, temperature, json_output=False) -> dict:
    payload: dict[str, Any] = {"model": model, "max_tokens": max_tokens,
                               "temperature": temperature,
                               "messages": _openai_messages(messages)}
    if json_output:
        payload["response_format"] = {"type": "json_object"}
    tl = _openai_tools(tools)
    if tl:
        payload["tools"] = tl
    return payload


def _openai_parse(data: dict) -> ChatTurn:
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    calls = []
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except (json.JSONDecodeError, TypeError):
            args = {}
        calls.append(ToolCall(id=tc.get("id", ""), name=fn.get("name", ""),
                              arguments=args if isinstance(args, dict) else {}))
    u = data.get("usage") or {}
    return ChatTurn(
        text=msg.get("content") or None,
        tool_calls=calls,
        usage=Usage(int(u.get("prompt_tokens", 0)), int(u.get("completion_tokens", 0))),
        stop_reason=choice.get("finish_reason"),
        raw=data,
    )


# --- request assembly (shared by sync + async) -------------------------------


def _build_request(provider: str, model: str, key: str, messages, tools,
                   max_tokens: int, temperature: float,
                   json_output: bool = False) -> tuple[str, dict, dict]:
    """Return (url, headers, json_payload) for the resolved provider. With
    `json_output`, the OpenAI/xAI family asks for a JSON-object reply
    (``response_format``); Anthropic's Messages API has no such switch, so there the
    prompt must ask for JSON and the caller must validate."""
    spec = PROVIDERS[provider]
    if spec["family"] == "anthropic":
        url = f"{spec['base_url']}/v1/messages"
        headers = {"x-api-key": key, "anthropic-version": ANTHROPIC_VERSION,
                   "content-type": "application/json"}
        payload = _anthropic_payload(messages, tools, model, max_tokens, temperature)
    else:
        url = f"{spec['base_url']}/chat/completions"
        headers = {"authorization": f"Bearer {key}", "content-type": "application/json"}
        payload = _openai_payload(messages, tools, model, max_tokens, temperature, json_output)
    return url, headers, payload


def _parse(provider: str, data: dict) -> ChatTurn:
    return (_anthropic_parse if PROVIDERS[provider]["family"] == "anthropic"
            else _openai_parse)(data)


# --- public entry points -----------------------------------------------------


def chat_with_tools(
    messages: list[dict],
    tools: list[dict] | None = None,
    *,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    timeout: float = 30.0,
    json_output: bool = False,
) -> ChatTurn:
    """One synchronous provider call. `messages` are canonical (see module docs);
    `tools` are provider-neutral ``{name, description, parameters(JSON Schema)}``.
    Returns a `ChatTurn`; raises `LLMError` on transport/HTTP/credential failure."""
    prov = resolve_provider(provider)
    mdl = resolve_model(prov, model)
    key = resolve_key(prov, api_key)
    url, headers, payload = _build_request(prov, mdl, key, messages, tools,
                                           max_tokens, temperature, json_output)
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as exc:
        raise LLMError(f"{prov} HTTP {exc.response.status_code}: {exc.response.text[:400]}") from exc
    except httpx.HTTPError as exc:
        raise LLMError(f"{prov} request failed: {exc}") from exc
    return _parse(prov, data)


async def achat_with_tools(
    messages: list[dict],
    tools: list[dict] | None = None,
    *,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    timeout: float = 30.0,
    json_output: bool = False,
) -> ChatTurn:
    """Async twin of `chat_with_tools` — use inside an event loop (e.g. FastAPI)."""
    prov = resolve_provider(provider)
    mdl = resolve_model(prov, model)
    key = resolve_key(prov, api_key)
    url, headers, payload = _build_request(prov, mdl, key, messages, tools,
                                           max_tokens, temperature, json_output)
    anthropic = PROVIDERS[prov]["family"] == "anthropic"
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            turn = _parse(prov, data)
            # Server tools (e.g. web_search) can make Anthropic pause mid-turn; it
            # asks us to continue by echoing its content back. Loop until it's done.
            guard = 0
            while anthropic and turn.stop_reason == "pause_turn" and guard < 4:
                guard += 1
                payload["messages"].append({"role": "assistant", "content": data.get("content", [])})
                resp = await client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
                nxt = _parse(prov, data)
                turn = ChatTurn(
                    text=nxt.text, tool_calls=nxt.tool_calls, stop_reason=nxt.stop_reason,
                    usage=Usage(turn.usage.input_tokens + nxt.usage.input_tokens,
                                turn.usage.output_tokens + nxt.usage.output_tokens),
                    raw=nxt.raw)
    except httpx.HTTPStatusError as exc:
        raise LLMError(f"{prov} HTTP {exc.response.status_code}: {exc.response.text[:400]}") from exc
    except httpx.HTTPError as exc:
        raise LLMError(f"{prov} request failed: {exc}") from exc
    return turn


# --- streaming text (no tools) -----------------------------------------------


def _stream_delta(family: str, event: dict) -> str:
    """Pull the text delta out of one parsed SSE event ('' when it carries none)."""
    if family == "anthropic":
        if event.get("type") == "content_block_delta":
            delta = event.get("delta") or {}
            if delta.get("type") == "text_delta":
                return delta.get("text", "")
        return ""
    choices = event.get("choices") or []
    if choices:
        return (choices[0].get("delta") or {}).get("content") or ""
    return ""


def _stream_usage(family: str, event: dict, usage: Usage) -> None:
    """Accumulate token counts from one SSE event into `usage`."""
    if family == "anthropic":
        if event.get("type") == "message_start":
            u = (event.get("message") or {}).get("usage") or {}
            usage.input_tokens = int(u.get("input_tokens") or 0)
        elif event.get("type") == "message_delta":
            u = event.get("usage") or {}
            usage.output_tokens = int(u.get("output_tokens") or usage.output_tokens)
        return
    u = event.get("usage")
    if u:
        usage.input_tokens = int(u.get("prompt_tokens") or 0)
        usage.output_tokens = int(u.get("completion_tokens") or 0)


async def astream_text(
    messages: list[dict],
    *,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    max_tokens: int = 1024,
    temperature: float = 0.8,
    timeout: float = 120.0,
    transport: httpx.AsyncBaseTransport | None = None,
    usage: Usage | None = None,
):
    """Async generator of text deltas for a plain (tool-free) chat turn.

    Same canonical messages and config resolution as `achat_with_tools`; uses each
    wire family's SSE stream (``stream: true``). For typing-effect UIs.
    Pass a `Usage()` as `usage` to have it filled with the turn's token counts
    once the stream ends (an async generator can't return a value).
    Raises `LLMError` on transport/HTTP/credential failure."""
    prov = resolve_provider(provider)
    mdl = resolve_model(prov, model)
    key = resolve_key(prov, api_key)
    url, headers, payload = _build_request(prov, mdl, key, messages, None,
                                           max_tokens, temperature)
    payload["stream"] = True
    family = PROVIDERS[prov]["family"]
    if family == "openai":
        payload["stream_options"] = {"include_usage": True}   # final chunk carries usage
    try:
        async with httpx.AsyncClient(timeout=timeout, transport=transport) as client:
            async with client.stream("POST", url, headers=headers, json=payload) as resp:
                if resp.status_code >= 400:
                    body = (await resp.aread()).decode(errors="replace")
                    raise LLMError(f"{prov} HTTP {resp.status_code}: {body[:400]}")
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data or data == "[DONE]":
                        continue
                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if family == "anthropic" and event.get("type") == "error":
                        raise LLMError(f"{prov} stream error: {event.get('error')}")
                    if usage is not None:
                        _stream_usage(family, event, usage)
                    text = _stream_delta(family, event)
                    if text:
                        yield text
    except httpx.HTTPError as exc:
        raise LLMError(f"{prov} request failed: {exc}") from exc
