"""
Unit tests for services/llm_gateway.py (reusable LLM Gateway).
Tests dynamic model routing, text condensation, JSON sanitization, and multi-provider failover.
"""

import unittest
from unittest.mock import MagicMock, patch

from services.llm_gateway import (
    LLMGateway,
    get_llm_gateway,
    safe_extract_json,
    condense_head_tail,
)


class TestLLMGatewaySanitization(unittest.TestCase):
    """Tests safe_extract_json and condense_head_tail helpers."""

    def test_safe_extract_json_clean(self):
        result = safe_extract_json('{"status": "ok", "score": 95}')
        self.assertEqual(result, {"status": "ok", "score": 95})

    def test_safe_extract_json_markdown_block(self):
        raw = """Here is the audit result:
```json
{
  "overall_score": 88,
  "status": "PASS"
}
```
Hope this helps!"""
        result = safe_extract_json(raw)
        self.assertEqual(result["overall_score"], 88)
        self.assertEqual(result["status"], "PASS")

    def test_safe_extract_json_with_think_tags(self):
        raw = """<think>
Internal chain of thought reasoning...
Checking Kenyan DPA principles...
</think>
{
  "overall_score": 90,
  "compliant": true
}"""
        result = safe_extract_json(raw)
        self.assertEqual(result["overall_score"], 90)
        self.assertTrue(result["compliant"])

    def test_safe_extract_json_trailing_commas(self):
        raw = '{"name": "Kenya Corp", "items": [1, 2, 3, ], "nested": {"a": 1, }, }'
        result = safe_extract_json(raw)
        self.assertEqual(result["name"], "Kenya Corp")
        self.assertEqual(result["items"], [1, 2, 3])

    def test_safe_extract_json_empty_or_invalid(self):
        with self.assertRaises(ValueError):
            safe_extract_json("")
        with self.assertRaises(ValueError):
            safe_extract_json("not valid json at all")

    def test_condense_head_tail_short(self):
        short = "A" * 5000
        self.assertEqual(condense_head_tail(short, 4500, 4000), short)

    def test_condense_head_tail_long(self):
        text = "START_" + ("M" * 10000) + "_END"
        condensed = condense_head_tail(text, head_chars=200, tail_chars=200)
        self.assertTrue(condensed.startswith("START_"))
        self.assertTrue(condensed.endswith("_END"))
        self.assertIn("CONDENSED FOR TOKEN WINDOW", condensed)
        self.assertTrue(len(condensed) < len(text))


class TestLLMGatewayRouting(unittest.TestCase):
    """Tests model routing and candidate chain building."""

    def setUp(self):
        self.gateway = LLMGateway(
            groq_api_key="test-groq-key",
            gemini_api_key="test-gemini-key",
            groq_base_url="https://api.groq.com/openai/v1",
            gemini_base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )

    def test_candidate_chain_short_input(self):
        """Short documents (<= 9000 chars) should prioritize Groq gpt-oss models first."""
        chain = self.gateway.get_candidate_chain(input_length=5000)
        self.assertTrue(len(chain) >= 2)
        self.assertEqual(chain[0]["provider"], "groq/openai")
        self.assertIn("gpt-oss", chain[0]["model"])
        # Gemini appears later in chain as fallback
        providers = [t["provider"] for t in chain]
        self.assertIn("gemini", providers)

    def test_candidate_chain_long_input(self):
        """Long documents (> 9000 chars) should prioritize Gemini 1M-window models first."""
        chain = self.gateway.get_candidate_chain(input_length=25000)
        self.assertTrue(len(chain) >= 2)
        self.assertEqual(chain[0]["provider"], "gemini")
        self.assertEqual(chain[0]["model"], "gemini-3.5-flash")
        # Groq appears as secondary fallback
        providers = [t["provider"] for t in chain]
        self.assertIn("groq/openai", providers)


class TestLLMGatewayExecution(unittest.TestCase):
    """Tests generate_json and generate_text with mocks."""

    def setUp(self):
        self.gateway = LLMGateway(
            groq_api_key="test-groq-key",
            gemini_api_key="test-gemini-key",
        )

    @patch("openai.OpenAI")
    def test_generate_json_gemini_native_json_mode(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content='{"status": "audited", "score": 95}'))
        ]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_cls.return_value = mock_client

        # Long text triggers Gemini first
        parsed, model = self.gateway.generate_json(
            system_prompt="Auditor system prompt",
            user_content="Audit this: {document_text}",
            document_text="Detailed policy text " * 500
        )

        self.assertEqual(parsed, {"status": "audited", "score": 95})
        self.assertEqual(model, "gemini-3.5-flash")

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        self.assertEqual(call_kwargs["response_format"], {"type": "json_object"})
        self.assertEqual(call_kwargs["max_tokens"], 8192)

    @patch("openai.OpenAI")
    def test_generate_text_markdown(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="# Bespoke Privacy Policy\n\nSection 1..."))
        ]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_cls.return_value = mock_client

        text, model = self.gateway.generate_text(
            system_prompt="Legal drafter",
            user_content="Redraft policy: {document_text}",
            document_text="Original text"
        )

        self.assertIn("# Bespoke Privacy Policy", text)
        self.assertTrue(mock_client.chat.completions.create.called)


if __name__ == "__main__":
    unittest.main()
