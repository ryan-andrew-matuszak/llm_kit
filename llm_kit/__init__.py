"""llm_kit — a provider-agnostic chat-with-tools client (Anthropic / OpenAI / xAI).

See client.py for the full contract. Public surface:
    chat_with_tools / achat_with_tools   - sync & async provider call
    ChatTurn, ToolCall, Usage            - the return shapes
    tool_calls_message / tool_result_message - canonical message builders
    estimate_cost                        - token usage -> dollars
    PROVIDERS, PRICES                    - the registries
    LLMError                             - raised on transport/HTTP/credential failure
"""
from .client import (
    PRICES,
    PROVIDERS,
    ChatTurn,
    LLMError,
    ToolCall,
    Usage,
    achat_with_tools,
    astream_text,
    chat_with_tools,
    estimate_cost,
    resolve_model,
    resolve_provider,
    tool_calls_message,
    tool_result_message,
)

__all__ = [
    "chat_with_tools",
    "achat_with_tools",
    "astream_text",
    "ChatTurn",
    "ToolCall",
    "Usage",
    "tool_calls_message",
    "tool_result_message",
    "estimate_cost",
    "resolve_provider",
    "resolve_model",
    "PROVIDERS",
    "PRICES",
    "LLMError",
]

__version__ = "0.2.0"
