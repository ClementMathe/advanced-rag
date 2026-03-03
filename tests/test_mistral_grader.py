"""
Tests for MistralDocumentGrader, MistralAnswerGrader, MistralQueryRewriter.

All tests mock mistralai.Mistral to return pre-built tool call responses.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.graders import AnswerGrader, DocumentGrader, QueryRewriter
from src.mistral_grader import (
    MistralAnswerGrader,
    MistralDocumentGrader,
    MistralQueryRewriter,
    RateLimiter,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tool_call_response(args: dict):
    """Build a mock Mistral API response with a single tool call."""
    func = SimpleNamespace(arguments=json.dumps(args))
    tool_call = SimpleNamespace(function=func)
    message = SimpleNamespace(tool_calls=[tool_call])
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice], usage=None)


def _make_client(args: dict):
    """Return a mock Mistral client whose chat.complete always returns args."""
    client = MagicMock()
    client.chat.complete.return_value = _tool_call_response(args)
    return client


# ---------------------------------------------------------------------------
# Inheritance checks
# ---------------------------------------------------------------------------


class TestInheritance:
    def test_doc_grader_is_document_grader(self):
        client = _make_client({"relevant": True, "confidence": 0.9})
        grader = MistralDocumentGrader(client=client)
        assert isinstance(grader, DocumentGrader)

    def test_answer_grader_is_answer_grader(self):
        client = _make_client({"acceptable": True, "confidence": 0.9, "issue": "none"})
        grader = MistralAnswerGrader(client=client)
        assert isinstance(grader, AnswerGrader)

    def test_query_rewriter_is_query_rewriter(self):
        client = _make_client({"rewritten_query": "q", "strategy": "rephrase"})
        rewriter = MistralQueryRewriter(client=client)
        assert isinstance(rewriter, QueryRewriter)


# ---------------------------------------------------------------------------
# MistralDocumentGrader tests
# ---------------------------------------------------------------------------


class TestMistralDocumentGrader:
    def test_doc_grade_single_relevant(self):
        """Returns True when relevant=True and confidence >= threshold."""
        client = _make_client({"relevant": True, "confidence": 0.9})
        grader = MistralDocumentGrader(client=client)
        assert grader.grade_single("q", "doc text") is True

    def test_doc_grade_single_irrelevant(self):
        """Returns False when relevant=False."""
        client = _make_client({"relevant": False, "confidence": 0.8})
        grader = MistralDocumentGrader(client=client)
        assert grader.grade_single("q", "doc text") is False

    def test_doc_grade_low_confidence_boundary(self):
        """Returns False when confidence < threshold even if relevant=True."""
        client = _make_client({"relevant": True, "confidence": 0.55})
        grader = MistralDocumentGrader(client=client, confidence_threshold=0.6)
        assert grader.grade_single("q", "doc text") is False

    def test_doc_grade_batch_returns_list(self):
        """grade_batch returns a list of booleans, one per document."""
        responses = [
            _tool_call_response({"relevant": True, "confidence": 0.9}),
            _tool_call_response({"relevant": False, "confidence": 0.8}),
            _tool_call_response({"relevant": True, "confidence": 0.7}),
            _tool_call_response({"relevant": False, "confidence": 0.95}),
            _tool_call_response({"relevant": True, "confidence": 0.85}),
        ]
        client = MagicMock()
        client.chat.complete.side_effect = responses

        grader = MistralDocumentGrader(client=client)
        docs = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        results = grader.grade_batch("q", docs)

        assert results == [True, False, True, False, True]
        assert len(results) == 5

    def test_grade_batch_with_scores_returns_dicts(self):
        """grade_batch_with_scores returns full dict with confidence and reason."""
        client = _make_client(
            {"relevant": True, "confidence": 0.9, "reason": "Directly mentions the topic."}
        )
        grader = MistralDocumentGrader(client=client)
        results = grader.grade_batch_with_scores("q", ["doc text"])

        assert len(results) == 1
        assert results[0]["relevant"] is True
        assert results[0]["confidence"] == 0.9
        assert "reason" in results[0]

    def test_grade_batch_empty_returns_empty(self):
        """grade_batch with empty list returns empty list without API call."""
        client = MagicMock()
        grader = MistralDocumentGrader(client=client)
        assert grader.grade_batch("q", []) == []
        client.chat.complete.assert_not_called()

    def test_doc_grader_reuses_single_grading_prompt(self):
        """grade_single uses SINGLE_GRADING_PROMPT (inherited from DocumentGrader)."""
        client = _make_client({"relevant": True, "confidence": 0.9})
        grader = MistralDocumentGrader(client=client)
        grader.grade_single("test query", "test document")

        call_args = client.chat.complete.call_args
        user_content = call_args.kwargs["messages"][0]["content"]
        assert "test query" in user_content
        assert "test document" in user_content


# ---------------------------------------------------------------------------
# MistralAnswerGrader tests
# ---------------------------------------------------------------------------


class TestMistralAnswerGrader:
    def test_answer_grade_acceptable(self):
        """Returns True when acceptable=True."""
        client = _make_client({"acceptable": True, "confidence": 0.9, "issue": "none"})
        grader = MistralAnswerGrader(client=client)
        assert grader.grade("q", "good answer", ["doc1"]) is True

    def test_answer_grade_wrong_answer(self):
        """Returns False when acceptable=False."""
        client = _make_client({"acceptable": False, "confidence": 0.85, "issue": "wrong_answer"})
        grader = MistralAnswerGrader(client=client)
        assert grader.grade("q", "bad answer", ["doc1"]) is False

    def test_grade_with_details_returns_dict(self):
        """grade_with_details returns full structured output."""
        client = _make_client({"acceptable": True, "confidence": 0.9, "issue": "none"})
        grader = MistralAnswerGrader(client=client)
        result = grader.grade_with_details("q", "answer", ["doc1"])

        assert result["acceptable"] is True
        assert result["confidence"] == 0.9
        assert result["issue"] == "none"

    def test_empty_answer_returns_false(self):
        """Empty answer returns False immediately, without API call."""
        client = MagicMock()
        grader = MistralAnswerGrader(client=client)
        assert grader.grade("q", "", ["doc1"]) is False
        client.chat.complete.assert_not_called()

    def test_answer_grader_reuses_grading_prompt(self):
        """grade() uses GRADING_PROMPT (inherited from AnswerGrader)."""
        client = _make_client({"acceptable": True, "confidence": 0.9, "issue": "none"})
        grader = MistralAnswerGrader(client=client)
        grader.grade("my question", "my answer", ["doc text"])

        call_args = client.chat.complete.call_args
        user_content = call_args.kwargs["messages"][0]["content"]
        assert "my question" in user_content
        assert "my answer" in user_content


# ---------------------------------------------------------------------------
# MistralQueryRewriter tests
# ---------------------------------------------------------------------------


class TestMistralQueryRewriter:
    def test_query_rewrite_returns_string(self):
        """rewrite() returns the rewritten_query string."""
        client = _make_client(
            {"rewritten_query": "Beyonce Grammy wins 2015", "strategy": "add_context"}
        )
        rewriter = MistralQueryRewriter(client=client)
        result = rewriter.rewrite("q", num_total=5, num_relevant=1)
        assert result == "Beyonce Grammy wins 2015"

    def test_query_rewrite_logs_strategy(self):
        """rewrite() logs the rewriting strategy via loguru."""
        from loguru import logger

        log_messages = []
        handler_id = logger.add(lambda msg: log_messages.append(msg), level="INFO")

        try:
            client = _make_client(
                {"rewritten_query": "Beyonce Grammy wins 2015", "strategy": "add_context"}
            )
            rewriter = MistralQueryRewriter(client=client)
            rewriter.rewrite("Grammy wins?", num_total=5, num_relevant=1)
        finally:
            logger.remove(handler_id)

        combined = "".join(log_messages)
        assert "add_context" in combined

    def test_query_rewrite_fallback_on_empty(self):
        """Falls back to original query if rewritten_query is empty."""
        client = _make_client({"rewritten_query": "", "strategy": "rephrase"})
        rewriter = MistralQueryRewriter(client=client)
        result = rewriter.rewrite("original query", num_total=5, num_relevant=0)
        assert result == "original query"

    def test_query_rewriter_reuses_rewrite_prompt(self):
        """rewrite() uses REWRITE_PROMPT (inherited from QueryRewriter)."""
        client = _make_client({"rewritten_query": "improved query", "strategy": "rephrase"})
        rewriter = MistralQueryRewriter(client=client)
        rewriter.rewrite("original", num_total=10, num_relevant=2)

        call_args = client.chat.complete.call_args
        user_content = call_args.kwargs["messages"][0]["content"]
        assert "original" in user_content
        assert "10" in user_content
        assert "2" in user_content


# ---------------------------------------------------------------------------
# Retry tests
# ---------------------------------------------------------------------------


class TestMistralGraderRetry:
    def test_retry_on_rate_limit(self):
        """Grader retries on HTTP 429 and eventually succeeds."""
        from mistralai.models import SDKError

        raw_response = MagicMock()
        raw_response.status_code = 429
        raw_response.text = "rate limited"

        err = SDKError.__new__(SDKError)
        object.__setattr__(err, "raw_response", raw_response)
        object.__setattr__(err, "message", "Status 429")
        object.__setattr__(err, "body", None)
        err.args = ("Status 429",)

        success = _tool_call_response({"relevant": True, "confidence": 0.9})

        client = MagicMock()
        client.chat.complete.side_effect = [err, err, success]

        with patch("src.mistral_grader.time.sleep"):
            grader = MistralDocumentGrader(client=client, max_retries=5, retry_delay=0.01)
            result = grader.grade_single("q", "doc")

        assert client.chat.complete.call_count == 3
        assert result is True


# ---------------------------------------------------------------------------
# RateLimiter tests
# ---------------------------------------------------------------------------


class TestRateLimiter:
    def test_rate_limiter_waits_between_calls(self):
        """RateLimiter sleeps when calls are too fast."""
        limiter = RateLimiter(calls_per_second=1.0)
        limiter._last_call = 1e18  # far in the future → forces sleep

        with patch("src.mistral_grader.time.sleep") as mock_sleep:
            limiter.wait()
            mock_sleep.assert_called_once()
