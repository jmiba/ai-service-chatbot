"""Classify OpenAI SDK errors before choosing a UI recovery path."""

from enum import Enum

from openai import APIConnectionError, APIError, RateLimitError


class OpenAIErrorAction(str, Enum):
    CONNECTION = "connection"
    RATE_LIMIT = "rate_limit"
    MCP_FALLBACK = "mcp_fallback"
    API_ERROR = "api_error"


def classify_openai_error(
    error: APIError,
    *,
    is_mcp_tool_error: bool,
) -> OpenAIErrorAction:
    """Return the one recovery action appropriate for an SDK error."""
    if isinstance(error, APIConnectionError):
        return OpenAIErrorAction.CONNECTION
    if isinstance(error, RateLimitError):
        return OpenAIErrorAction.RATE_LIMIT
    if is_mcp_tool_error:
        return OpenAIErrorAction.MCP_FALLBACK
    return OpenAIErrorAction.API_ERROR
