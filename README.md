# llm_kit

A small, provider-agnostic **chat-with-tools** client for hosted LLMs —
Anthropic, OpenAI, and xAI — behind one function. Swapping the model (or the
whole provider) is a config flag, not a rewrite. Deliberately **raw REST over
`httpx`, no vendor SDKs**, so the only dependency is `httpx` and it installs
cleanly anywhere, including a Raspberry Pi.

Sibling of [`ollama_kit`](../ollama_kit) (local models) and
[`prompt_kit`](../prompt_kit) (prompt assembly). `ollama_kit` is for local
generation; `llm_kit` is for hosted tool-calling. A future `ollama` provider here
would let a local model serve the same seam.

## Install

```bash
pip install -e ~/Development/shared/llm_kit
```

## Use

```python
from llm_kit import chat_with_tools, achat_with_tools, tool_result_message

tools = [{
    "name": "set_light",
    "description": "Turn a room's lights on or off.",
    "parameters": {"type": "object",
                   "properties": {"room": {"type": "string"},
                                  "on": {"type": "boolean"}},
                   "required": ["room", "on"]},
}]

turn = chat_with_tools(
    [{"role": "user", "content": "turn off the living room"}],
    tools,
    provider="anthropic",     # or "openai" / "xai"; else $LLM_PROVIDER
    model="claude-haiku-4-5",  # else $LLM_MODEL, else the provider default
)                              # key from arg or $ANTHROPIC_API_KEY / $OPENAI_API_KEY / $XAI_API_KEY

if turn.tool_calls:
    for call in turn.tool_calls:
        result = run(call.name, call.arguments)          # your dispatch
        # feed the result back and loop until turn.tool_calls is empty
else:
    print(turn.text)          # the final answer
```

`achat_with_tools(...)` is the async twin for use inside an event loop.

## The tool loop (provider-neutral)

Messages are canonical dicts that every adapter understands, so your loop never
branches on provider:

- text: `{"role": "system"|"user"|"assistant", "content": str}`
- the model's tool request — echo it back with `tool_calls_message(turn.tool_calls, turn.text)`
- each result — `tool_result_message(call.id, result_string)`

Append the assistant tool-call turn, then one result message per call, then call
again. Stop when a turn comes back with no `tool_calls`.

## Streaming

`astream_text(messages, provider=..., model=..., usage=Usage())` is an async
generator of text deltas (no tools) over each family's SSE stream. Pass a
`Usage()` to get the turn's token counts filled in when the stream ends.

## Shared usage ledger

`record_usage(provider, model, usage)` appends one JSON line to
`$LLM_USAGE_LEDGER` (default `~/.local/share/llm_kit/usage.jsonl`);
`read_usage(since=...)` reads it back. Every app on a machine can write to the
same ledger so one dashboard shows total spend. **There is deliberately no app
field** — the ledger says what was spent on which model, never which app spent
it, so a dashboard built on it can't reveal what's installed.

## Budget

`turn.usage` carries `input_tokens` / `output_tokens`; `estimate_cost(usage,
model)` converts them to dollars against the `PRICES` table (unknown models cost
`0.0` rather than lying). Log both to keep spend under a cap.

## Providers

| provider    | wire family | base URL                     | key env             | default model      |
|-------------|-------------|------------------------------|---------------------|--------------------|
| `anthropic` | Messages    | `api.anthropic.com`          | `ANTHROPIC_API_KEY` | `claude-haiku-4-5` |
| `openai`    | Chat        | `api.openai.com/v1`          | `OPENAI_API_KEY`    | `gpt-4o-mini`      |
| `xai`       | Chat        | `api.x.ai/v1`                | `XAI_API_KEY`       | `grok-3-mini`      |

xAI is OpenAI-compatible, so it reuses the Chat Completions adapter. Add a
provider by adding one `PROVIDERS` entry (and a translate/parse pair only if its
wire format is genuinely new).

## Tests

Fully offline (`httpx.MockTransport`, no keys, no network):

```bash
python -m pytest ~/Development/shared/llm_kit/tests -q
```
