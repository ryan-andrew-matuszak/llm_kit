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

Config resolves from args first, then env: ``LLM_PROVIDER``, ``LLM_MODEL``, and
the provider key (``ANTHROPIC_API_KEY`` / ``OPENAI_API_KEY`` / ``XAI_API_KEY``).
`estimate_cost()` turns reported token usage into dollars against `PRICES` so a
caller can log spend and hold a budget.
"""
from __future__ import annotations

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
    """Provider-neutral ``{name, description, parameters}`` -> Anthropic tools."""
    out = []
    for t in tools or []:
        out.append({
            "name": t["name"],
            "description": t.get("description", ""),
            "input_schema": t.get("parameters") or {"type": "object", "properties": {}},
        })
    return out


def _anthropic_messages(messages: list[dict]) -> tuple[str | None, list[dict]]:
    """Translate canonical messages to (system, messages) for /v1/messages.
    Consecutive tool results are merged into one user message (Anthropic wants
    each ``tool_result`` in a user turn)."""
    system_parts: list[str] = []
    out: list[dict] = []
    for m in messages:
        role = m["role"]
        if role == "system":
            if m.get("content"):
                system_parts.append(m["content"])
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
        out.append({"type": "function", "function": {
            "name": t["name"],
            "description": t.get("description", ""),
            "parameters": t.get("parameters") or {"type": "object", "properties": {}},
        }})
    return out


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
        else:
            out.append({"role": role, "content": m.get("content") or ""})
    return out


def _openai_payload(messages, tools, model, max_tokens, temperature) -> dict:
    payload: dict[str, Any] = {"model": model, "max_tokens": max_tokens,
                               "temperature": temperature,
                               "messages": _openai_messages(messages)}
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
                   max_tokens: int, temperature: float) -> tuple[str, dict, dict]:
    """Return (url, headers, json_payload) for the resolved provider."""
    spec = PROVIDERS[provider]
    if spec["family"] == "anthropic":
        url = f"{spec['base_url']}/v1/messages"
        headers = {"x-api-key": key, "anthropic-version": ANTHROPIC_VERSION,
                   "content-type": "application/json"}
        payload = _anthropic_payload(messages, tools, model, max_tokens, temperature)
    else:
        url = f"{spec['base_url']}/chat/completions"
        headers = {"authorization": f"Bearer {key}", "content-type": "application/json"}
        payload = _openai_payload(messages, tools, model, max_tokens, temperature)
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
) -> ChatTurn:
    """One synchronous provider call. `messages` are canonical (see module docs);
    `tools` are provider-neutral ``{name, description, parameters(JSON Schema)}``.
    Returns a `ChatTurn`; raises `LLMError` on transport/HTTP/credential failure."""
    prov = resolve_provider(provider)
    mdl = resolve_model(prov, model)
    key = resolve_key(prov, api_key)
    url, headers, payload = _build_request(prov, mdl, key, messages, tools,
                                           max_tokens, temperature)
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
) -> ChatTurn:
    """Async twin of `chat_with_tools` — use inside an event loop (e.g. FastAPI)."""
    prov = resolve_provider(provider)
    mdl = resolve_model(prov, model)
    key = resolve_key(prov, api_key)
    url, headers, payload = _build_request(prov, mdl, key, messages, tools,
                                           max_tokens, temperature)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as exc:
        raise LLMError(f"{prov} HTTP {exc.response.status_code}: {exc.response.text[:400]}") from exc
    except httpx.HTTPError as exc:
        raise LLMError(f"{prov} request failed: {exc}") from exc
    return _parse(prov, data)
