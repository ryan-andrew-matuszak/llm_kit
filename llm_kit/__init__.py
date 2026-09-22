"""llm_kit — a provider-agnostic chat-with-tools client (Anthropic / OpenAI / xAI).

See client.py for the full contract. Public surface:
    chat_with_tools / achat_with_tools   - sync & async provider call
    ChatTurn, ToolCall, Usage            - the return shapes
    tool_calls_message / tool_result_message - canonical message builders
    image_part / text_part               - multimodal (vision) content parts
    estimate_cost                        - token usage -> dollars
    PROVIDERS, PRICES                    - the registries
    LLMError                             - raised on transport/HTTP/credential failure
"""
from .client import (
    IMAGE_TYPES,
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
    image_part,
    resolve_model,
    resolve_provider,
    text_part,
    tool_calls_message,
    tool_result_message,
)
from .ledger import ledger_path, read_usage, record_usage

__all__ = [
    "chat_with_tools",
    "achat_with_tools",
    "astream_text",
    "ChatTurn",
    "ToolCall",
    "Usage",
    "tool_calls_message",
    "tool_result_message",
    "image_part",
    "text_part",
    "IMAGE_TYPES",
    "estimate_cost",
    "resolve_provider",
    "resolve_model",
    "PROVIDERS",
    "PRICES",
    "LLMError",
    "record_usage",
    "read_usage",
    "ledger_path",
]

__version__ = "0.4.0"
