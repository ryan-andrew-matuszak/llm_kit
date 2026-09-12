# llm_kit — working notes

Provider-agnostic `chat_with_tools` over Anthropic / OpenAI / xAI. One job:
give an app a swappable hosted-LLM tool-caller behind a single seam.

## Non-negotiables
- **Raw REST over `httpx` only — never add a vendor SDK.** The one-dependency
  rule is why this installs on a Pi. `anthropic` / `openai` pull heavy trees;
  don't.
- **Two wire families, not three adapters:** Anthropic Messages and OpenAI Chat
  Completions. xAI rides the OpenAI adapter. A new provider is a `PROVIDERS`
  row; only a genuinely new wire format earns a new translate/parse pair.
- **Pure builders/parsers stay side-effect-free** (`_anthropic_*`, `_openai_*`)
  so tests exercise them with no network. Keep the sync/async entry points thin
  wrappers over the same `_build_request` / `_parse`.
- **Canonical messages are the contract.** Callers speak the neutral dict shapes
  (see the client docstring); adapters translate. Don't leak provider shapes to
  the caller.

## Gotchas
- Anthropic wants each `tool_result` inside a **user** turn; consecutive results
  are merged into one user message (`_anthropic_messages`). OpenAI takes each as
  a `role:"tool"` message.
- OpenAI tool-call arguments are a **JSON string** on the wire — always
  `json.loads` (done in `_openai_parse`; bad JSON degrades to `{}`).
- Model ids are caller-supplied; only Anthropic's `claude-haiku-4-5` default is
  pinned from first-party docs. Keep `PRICES` current; unknown model ⇒ `$0`.

## Tests
`python -m pytest tests -q` — offline via `httpx.MockTransport`. Cover any new
provider with a shape test + an end-to-end mock, like the existing two.
