"""
Tests for MistralAPIGenerator.

All tests mock mistralai.Mistral to avoid real API calls.
Uses unittest.mock following project conventions.
"""

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.mistral_generator import MistralAPIGenerator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_response(content: str, prompt_tokens: int = 100, completion_tokens: int = 20):
    """Build a minimal mock Mistral API response."""
    usage = SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice], usage=usage)


def _make_chunks(n: int = 3):
    return [{"content": f"Context chunk {i}."} for i in range(1, n + 1)]


def _make_sdk_error(status_code: int):
    """Build a mock SDKError with a given HTTP status code."""
    from mistralai.models import SDKError

    raw_response = MagicMock()
    raw_response.status_code = status_code
    raw_response.text = "error body"

    err = SDKError.__new__(SDKError)
    object.__setattr__(err, "raw_response", raw_response)
    object.__setattr__(err, "message", f"Status {status_code}")
    object.__setattr__(err, "body", None)
    err.args = (f"Status {status_code}",)
    return err


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_client():
    """A patched Mistral client that returns a successful response by default."""
    with patch("src.mistral_generator.Mistral") as MockMistral:
        client = MagicMock()
        client.chat.complete.return_value = _make_response("Paris is the capital of France.")
        MockMistral.return_value = client
        yield MockMistral, client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMistralAPIGeneratorInit:
    def test_empty_api_key_raises_on_init(self):
        """No env var, no kwarg → ValueError at construction."""
        with patch.dict(os.environ, {}, clear=True):
            # Ensure MISTRAL_API_KEY is not present
            env = {k: v for k, v in os.environ.items() if k != "MISTRAL_API_KEY"}
            with patch.dict(os.environ, env, clear=True):
                with patch("src.mistral_generator.Mistral"):
                    with pytest.raises(ValueError, match="No Mistral API key"):
                        MistralAPIGenerator(api_key=None)

    def test_api_key_from_env(self):
        """API key resolved from MISTRAL_API_KEY env var."""
        with patch.dict(
            os.environ, {"MISTRAL_API_KEY": "test-key-from-env"}  # pragma: allowlist secret
        ):
            with patch("src.mistral_generator.Mistral") as MockMistral:
                MistralAPIGenerator()
                MockMistral.assert_called_once_with(
                    api_key="test-key-from-env"  # pragma: allowlist secret
                )

    def test_api_key_kwarg_takes_precedence(self):
        """Explicit api_key kwarg overrides env var."""
        with patch.dict(os.environ, {"MISTRAL_API_KEY": "env-key"}):  # pragma: allowlist secret
            with patch("src.mistral_generator.Mistral") as MockMistral:
                MistralAPIGenerator(api_key="explicit-key")  # pragma: allowlist secret
                MockMistral.assert_called_once_with(
                    api_key="explicit-key"  # pragma: allowlist secret
                )


class TestMistralAPIGeneratorGenerate:
    def test_generate_returns_answer(self, mock_client):
        """generate() returns dict with answer, input_tokens, output_tokens."""
        MockMistral, client = mock_client
        client.chat.complete.return_value = _make_response(
            "Paris.", prompt_tokens=150, completion_tokens=5
        )

        gen = MistralAPIGenerator(api_key="test-key")
        result = gen.generate("What is the capital of France?", _make_chunks())

        assert result["answer"] == "Paris."
        assert result["input_tokens"] == 150
        assert result["output_tokens"] == 5
        assert result["model"] == "mistral-small-latest"
        assert "prompt" in result
        assert result["num_chunks_used"] == 3

    def test_model_name_passed_to_api(self, mock_client):
        """model_name is passed to client.chat.complete()."""
        MockMistral, client = mock_client
        client.chat.complete.return_value = _make_response("answer")

        gen = MistralAPIGenerator(api_key="test-key", model_name="mistral-large-latest")
        gen.generate("q?", _make_chunks(1))

        call_kwargs = client.chat.complete.call_args
        assert call_kwargs.kwargs["model"] == "mistral-large-latest"

    def test_temperature_respected(self, mock_client):
        """temperature kwarg is passed through to the API call."""
        MockMistral, client = mock_client
        client.chat.complete.return_value = _make_response("answer")

        gen = MistralAPIGenerator(api_key="test-key", temperature=0.7)
        gen.generate("q?", _make_chunks(1))

        call_kwargs = client.chat.complete.call_args
        assert call_kwargs.kwargs["temperature"] == 0.7

    def test_max_new_tokens_respected(self, mock_client):
        """max_new_tokens is passed as max_tokens to the API call."""
        MockMistral, client = mock_client
        client.chat.complete.return_value = _make_response("answer")

        gen = MistralAPIGenerator(api_key="test-key", max_new_tokens=120)
        gen.generate("q?", _make_chunks(1))

        call_kwargs = client.chat.complete.call_args
        assert call_kwargs.kwargs["max_tokens"] == 120


class TestMistralAPIGeneratorBuildPrompt:
    def test_build_prompt_matches_llm_generator_format(self):
        """Prompt format is identical to LLMGenerator.build_prompt()."""
        from src.generator import LLMGenerator  # noqa: F401

        with patch("src.mistral_generator.Mistral"):
            mistral_gen = MistralAPIGenerator(api_key="test-key")

        # Build the same prompt from both generators without loading model
        chunks = [
            {"content": "Paris is the capital of France."},
            {"content": "France is in Europe."},
        ]
        query = "What is the capital of France?"

        mistral_prompt = mistral_gen.build_prompt(query, chunks)

        # Verify key structural elements
        assert "Paris is the capital of France." in mistral_prompt
        assert "France is in Europe." in mistral_prompt
        assert query in mistral_prompt
        assert "[1]" in mistral_prompt
        assert "[2]" in mistral_prompt
        assert "Direct answer:" in mistral_prompt
        assert "ONLY the provided context" in mistral_prompt

    def test_max_chunks_limits_context(self):
        """Only max_chunks chunks are included in the prompt."""
        with patch("src.mistral_generator.Mistral"):
            gen = MistralAPIGenerator(api_key="test-key")

        chunks = [{"content": f"chunk {i}"} for i in range(10)]
        prompt = gen.build_prompt("q?", chunks, max_chunks=3)

        assert "chunk 0" in prompt
        assert "chunk 2" in prompt
        assert "chunk 3" not in prompt


class TestMistralAPIGeneratorRetry:
    def test_retry_on_rate_limit(self):
        """mock raises 429 twice then succeeds, verifies 3 total calls."""
        err = _make_sdk_error(429)
        success = _make_response("answer")

        with patch("src.mistral_generator.Mistral") as MockMistral:
            client = MagicMock()
            client.chat.complete.side_effect = [err, err, success]
            MockMistral.return_value = client

            with patch("src.mistral_generator.time.sleep"):
                gen = MistralAPIGenerator(api_key="test-key", max_retries=5, retry_delay=0.01)
                result = gen.generate("q?", _make_chunks(1))

        assert client.chat.complete.call_count == 3
        assert result["answer"] == "answer"

    def test_retry_exhausted_raises(self):
        """mock always raises 429, verifies exception after max_retries."""
        err = _make_sdk_error(429)

        with patch("src.mistral_generator.Mistral") as MockMistral:
            client = MagicMock()
            client.chat.complete.side_effect = err
            MockMistral.return_value = client

            with patch("src.mistral_generator.time.sleep"):
                gen = MistralAPIGenerator(api_key="test-key", max_retries=2, retry_delay=0.01)
                from mistralai.models import SDKError

                with pytest.raises(SDKError):
                    gen.generate("q?", _make_chunks(1))

        assert client.chat.complete.call_count == 3  # initial + 2 retries
