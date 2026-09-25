import unittest

import httpx
from openai import APIConnectionError, APIError, RateLimitError

from utils.openai_errors import OpenAIErrorAction, classify_openai_error


class OpenAIErrorRoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.request = httpx.Request("POST", "https://api.openai.com/v1/responses")

    def test_connection_error_never_uses_fallback(self) -> None:
        error = APIConnectionError(request=self.request)

        action = classify_openai_error(error, is_mcp_tool_error=True)

        self.assertEqual(action, OpenAIErrorAction.CONNECTION)

    def test_rate_limit_never_uses_fallback(self) -> None:
        response = httpx.Response(429, request=self.request)
        error = RateLimitError("rate limited", response=response, body=None)

        action = classify_openai_error(error, is_mcp_tool_error=True)

        self.assertEqual(action, OpenAIErrorAction.RATE_LIMIT)

    def test_only_confirmed_mcp_error_uses_fallback(self) -> None:
        error = APIError("tool list failed", self.request, body=None)

        self.assertEqual(
            classify_openai_error(error, is_mcp_tool_error=True),
            OpenAIErrorAction.MCP_FALLBACK,
        )
        self.assertEqual(
            classify_openai_error(error, is_mcp_tool_error=False),
            OpenAIErrorAction.API_ERROR,
        )


if __name__ == "__main__":
    unittest.main()
