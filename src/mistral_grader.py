"""
Mistral API-based grading and query rewriting for Agentic RAG.

Subclasses of DocumentGrader, AnswerGrader, and QueryRewriter from graders.py
that replace the brittle text-parsing approach with Mistral function calling.

Key difference from base classes:
- Base classes call generator._generate_text() then parse ambiguous text output
  through a multi-strategy fallback chain.
- Mistral subclasses use function calling: the API guarantees JSON-matching the
  tool schema, so all _parse_* methods are replaced by a single json.loads().

isinstance(MistralDocumentGrader(), DocumentGrader) is True — AgenticRAGPipeline
accepts them without modification.
"""

import json
import time
from typing import Dict, List, Optional

from loguru import logger
from mistralai import Mistral
from mistralai.models import SDKError

from src.graders import AnswerGrader, DocumentGrader, QueryRewriter

# ---------------------------------------------------------------------------
# Tool schemas for Mistral function calling
# ---------------------------------------------------------------------------

GRADE_DOCUMENT_TOOL = {
    "type": "function",
    "function": {
        "name": "grade_document",
        "description": "Grade whether a document is relevant to a query",
        "parameters": {
            "type": "object",
            "properties": {
                "relevant": {
                    "type": "boolean",
                    "description": (
                        "True if the document contains information useful for "
                        "answering the query"
                    ),
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": (
                        "Confidence in the relevance judgment " "(0=uncertain, 1=certain)"
                    ),
                },
                "reason": {
                    "type": "string",
                    "maxLength": 120,
                    "description": "One-sentence explanation of the relevance decision",
                },
            },
            "required": ["relevant", "confidence"],
        },
    },
}

GRADE_ANSWER_TOOL = {
    "type": "function",
    "function": {
        "name": "grade_answer",
        "description": "Grade whether a generated answer correctly addresses the question",
        "parameters": {
            "type": "object",
            "properties": {
                "acceptable": {
                    "type": "boolean",
                    "description": (
                        "True if the answer correctly addresses the question " "based on context"
                    ),
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "issue": {
                    "type": "string",
                    "enum": [
                        "none",
                        "wrong_answer",
                        "hallucination",
                        "incomplete",
                        "off_topic",
                    ],
                    "description": ("Type of issue if not acceptable, 'none' if acceptable"),
                },
            },
            "required": ["acceptable", "confidence", "issue"],
        },
    },
}

REWRITE_QUERY_TOOL = {
    "type": "function",
    "function": {
        "name": "rewrite_query",
        "description": "Rewrite a search query to improve retrieval quality",
        "parameters": {
            "type": "object",
            "properties": {
                "rewritten_query": {
                    "type": "string",
                    "maxLength": 150,
                    "description": (
                        "The improved query. Resolve pronouns, add specific " "names and context."
                    ),
                },
                "strategy": {
                    "type": "string",
                    "enum": ["expand_entity", "add_context", "decompose", "rephrase"],
                    "description": "What rewriting strategy was applied",
                },
            },
            "required": ["rewritten_query", "strategy"],
        },
    },
}


# ---------------------------------------------------------------------------
# Shared rate limiter
# ---------------------------------------------------------------------------


class RateLimiter:
    """Token-bucket rate limiter to stay within free-tier API limits (1 req/s)."""

    def __init__(self, calls_per_second: float = 0.5):
        self._min_interval = 1.0 / calls_per_second
        self._last_call = 0.0

    def wait(self):
        elapsed = time.monotonic() - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call = time.monotonic()


# ---------------------------------------------------------------------------
# Base mixin: retry + function calling helper
# ---------------------------------------------------------------------------


class _MistralGraderBase:
    """Shared retry and function-call logic for all Mistral grader subclasses."""

    def __init__(
        self,
        client: Mistral,
        model_name: str,
        max_retries: int = 5,
        retry_delay: float = 1.2,
        rate_limiter: Optional[RateLimiter] = None,
    ):
        self.client = client
        self.model_name = model_name
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.rate_limiter = rate_limiter

    def _call_with_tool(self, messages: list, tool: dict) -> dict:
        """
        Call the Mistral API with a single function tool and return the parsed args.

        Args:
            messages: List of message dicts for the API call.
            tool: The tool definition dict.

        Returns:
            Parsed JSON arguments from the tool call.

        Raises:
            SDKError: After all retries are exhausted.
            ValueError: If the response contains no tool calls.
        """
        if self.rate_limiter:
            self.rate_limiter.wait()

        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.chat.complete(
                    model=self.model_name,
                    messages=messages,
                    tools=[tool],
                    tool_choice="any",
                )
                # Extract first tool call arguments
                tool_calls = response.choices[0].message.tool_calls
                if not tool_calls:
                    raise ValueError(
                        f"Mistral returned no tool calls for tool " f"'{tool['function']['name']}'"
                    )
                args_json = tool_calls[0].function.arguments
                return json.loads(args_json)

            except SDKError as e:
                is_retryable = (
                    hasattr(e, "raw_response")
                    and e.raw_response is not None
                    and e.raw_response.status_code in (429, 500, 502, 503)
                )
                if is_retryable and attempt < self.max_retries:
                    status_code = e.raw_response.status_code
                    if status_code == 429:
                        # Respect Retry-After header; fallback to 60s floor
                        retry_after = None
                        try:
                            retry_after = float(e.raw_response.headers.get("Retry-After", 0))
                        except (TypeError, ValueError):
                            pass
                        delay = max(
                            retry_after or 0,
                            60.0 * (attempt + 1),  # 60s, 120s, 180s, ...
                        )
                    else:
                        delay = self.retry_delay * (2**attempt)
                    logger.warning(
                        f"Mistral API error (attempt {attempt + 1}/"
                        f"{self.max_retries + 1}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                else:
                    raise

            except Exception as e:
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2**attempt)
                    logger.warning(
                        f"Unexpected error (attempt {attempt + 1}/"
                        f"{self.max_retries + 1}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                else:
                    raise


# ---------------------------------------------------------------------------
# MistralDocumentGrader
# ---------------------------------------------------------------------------


class MistralDocumentGrader(_MistralGraderBase, DocumentGrader):
    """
    Document relevance grader using Mistral function calling.

    Subclasses DocumentGrader. Inherits SINGLE_GRADING_PROMPT and
    BATCH_GRADING_PROMPT prompt constants and the public interface.
    Overrides grade_single and grade_batch to use function calling —
    eliminating the entire _parse_batch_grades() fallback chain.

    Args:
        client: mistralai.Mistral instance.
        model_name: Mistral model to use for grading.
        confidence_threshold: Grade as relevant if confidence >= threshold.
        multi_hop: If True, use a multi-hop-aware prompt that grades documents
            as part of a reasoning chain rather than requiring direct relevance.
            Reduces false rejections of bridge documents in multi-hop QA.
        max_retries: Retry attempts on transient API errors.
        retry_delay: Base backoff delay in seconds.
        rate_limiter: Shared RateLimiter instance.
    """

    MULTI_HOP_GRADING_PROMPT = (
        "You are an expert document relevance grader for multi-hop reasoning tasks.\n"
        "\n"
        "Task: Determine if the following document is part of the evidence chain needed\n"
        "to answer the user's question.\n"
        "\n"
        "Context: This question may require multiple documents working together. A document\n"
        "is RELEVANT if it provides information needed at any step of the reasoning chain,\n"
        "even if it does not directly answer the full question.\n"
        "\n"
        "Example — Question: 'What is the birthplace of the director of Inception?'\n"
        "  → Doc about Christopher Nolan directing Inception → RELEVANT (bridge fact)\n"
        "  → Doc about Christopher Nolan's birthplace → RELEVANT (final answer)\n"
        "  → Both are needed; neither alone is sufficient to answer the question.\n"
        "\n"
        "Guidelines:\n"
        "- Grade as RELEVANT if the document provides any fact in the reasoning chain\n"
        "- Grade as NOT RELEVANT only if the document has no connection to the question\n"
        "- When uncertain, prefer RELEVANT — missing a bridge document breaks the chain\n"
        "\n"
        "Question: {query}\n"
        "\n"
        "Document: {document}\n"
    )

    def __init__(
        self,
        client: Mistral,
        model_name: str = "mistral-small-latest",
        confidence_threshold: float = 0.6,
        multi_hop: bool = False,
        max_retries: int = 5,
        retry_delay: float = 1.2,
        rate_limiter: Optional[RateLimiter] = None,
    ):
        # Do NOT call DocumentGrader.__init__() — no LLMGenerator needed
        _MistralGraderBase.__init__(
            self,
            client=client,
            model_name=model_name,
            max_retries=max_retries,
            retry_delay=retry_delay,
            rate_limiter=rate_limiter,
        )
        self.confidence_threshold = confidence_threshold
        self._grading_prompt = (
            self.MULTI_HOP_GRADING_PROMPT if multi_hop else self.SINGLE_GRADING_PROMPT
        )

    def grade_single(self, query: str, document: str) -> bool:
        """
        Grade a single document for relevance using function calling.

        Reuses SINGLE_GRADING_PROMPT (inherited from DocumentGrader).

        Args:
            query: User's search query.
            document: Document text.

        Returns:
            True if relevant and confidence >= threshold.
        """
        result = self.grade_single_with_details(query, document)
        return result["relevant"] and result["confidence"] >= self.confidence_threshold

    def grade_batch(self, query: str, documents: List[str]) -> List[bool]:
        """
        Grade multiple documents, one API call per document.

        Args:
            query: User's search query.
            documents: List of document texts.

        Returns:
            List of booleans (True=relevant), same length as documents.
        """
        if not documents:
            return []
        return [self.grade_single(query, doc) for doc in documents]

    def grade_batch_with_scores(self, query: str, documents: List[str]) -> List[Dict]:
        """
        Grade multiple documents and return full structured output.

        Args:
            query: User's search query.
            documents: List of document texts.

        Returns:
            List of dicts with keys: relevant (bool), confidence (float),
            reason (str, may be absent if not returned).
        """
        return [self.grade_single_with_details(query, doc) for doc in documents]

    def grade_single_with_details(self, query: str, document: str) -> Dict:
        """
        Grade a single document and return full tool call output.

        Args:
            query: User's search query.
            document: Document text.

        Returns:
            Dict with keys: relevant (bool), confidence (float),
            reason (str, optional).
        """
        prompt = self._grading_prompt.format(query=query, document=document)
        messages = [{"role": "user", "content": prompt}]
        return self._call_with_tool(messages, GRADE_DOCUMENT_TOOL)


# ---------------------------------------------------------------------------
# MistralAnswerGrader
# ---------------------------------------------------------------------------


class MistralAnswerGrader(_MistralGraderBase, AnswerGrader):
    """
    Answer quality grader using Mistral function calling.

    Subclasses AnswerGrader. Inherits GRADING_PROMPT. Overrides grade()
    to use function calling — replacing _parse_grade() and its yes/no
    text parsing.

    Args:
        client: mistralai.Mistral instance.
        model_name: Mistral model to use for grading.
        max_retries: Retry attempts on transient API errors.
        retry_delay: Base backoff delay in seconds.
        rate_limiter: Shared RateLimiter instance.
    """

    def __init__(
        self,
        client: Mistral,
        model_name: str = "mistral-small-latest",
        max_retries: int = 5,
        retry_delay: float = 1.2,
        rate_limiter: Optional[RateLimiter] = None,
    ):
        # Do NOT call AnswerGrader.__init__() — no LLMGenerator needed
        _MistralGraderBase.__init__(
            self,
            client=client,
            model_name=model_name,
            max_retries=max_retries,
            retry_delay=retry_delay,
            rate_limiter=rate_limiter,
        )

    def grade(self, query: str, answer: str, documents: List[str]) -> bool:
        """
        Grade whether the answer properly addresses the question.

        Reuses GRADING_PROMPT (inherited from AnswerGrader).

        Args:
            query: User's search query.
            answer: Generated answer.
            documents: Context document texts.

        Returns:
            True if the answer is acceptable.
        """
        if not answer or not answer.strip():
            return False
        return self.grade_with_details(query, answer, documents)["acceptable"]

    def grade_with_details(self, query: str, answer: str, documents: List[str]) -> Dict:
        """
        Grade the answer and return full structured output.

        Args:
            query: User's search query.
            answer: Generated answer.
            documents: Context document texts.

        Returns:
            Dict with keys: acceptable (bool), confidence (float), issue (str).
        """
        docs_text = "\n".join(f"{i + 1}. {doc[:500]}" for i, doc in enumerate(documents))
        prompt = self.GRADING_PROMPT.format(query=query, answer=answer, documents=docs_text)
        messages = [{"role": "user", "content": prompt}]
        return self._call_with_tool(messages, GRADE_ANSWER_TOOL)


# ---------------------------------------------------------------------------
# MistralQueryRewriter
# ---------------------------------------------------------------------------


class MistralQueryRewriter(_MistralGraderBase, QueryRewriter):
    """
    Query rewriter using Mistral function calling.

    Subclasses QueryRewriter. Inherits REWRITE_PROMPT. Overrides rewrite()
    to use function calling — replacing the entire prefix-stripping /
    truncation logic in _parse_rewrite().

    Args:
        client: mistralai.Mistral instance.
        model_name: Mistral model to use.
        max_retries: Retry attempts on transient API errors.
        retry_delay: Base backoff delay in seconds.
        rate_limiter: Shared RateLimiter instance.
    """

    def __init__(
        self,
        client: Mistral,
        model_name: str = "mistral-small-latest",
        max_retries: int = 5,
        retry_delay: float = 1.2,
        rate_limiter: Optional[RateLimiter] = None,
    ):
        # Do NOT call QueryRewriter.__init__() — no LLMGenerator needed
        _MistralGraderBase.__init__(
            self,
            client=client,
            model_name=model_name,
            max_retries=max_retries,
            retry_delay=retry_delay,
            rate_limiter=rate_limiter,
        )

    def rewrite(self, query: str, num_total: int, num_relevant: int) -> str:
        """
        Generate an improved search query using function calling.

        Reuses REWRITE_PROMPT (inherited from QueryRewriter).
        Returns the rewritten query string and logs the strategy.

        Args:
            query: Original search query.
            num_total: Total documents retrieved.
            num_relevant: Documents that passed relevance grading.

        Returns:
            Rewritten query string. Falls back to original if rewriting fails.
        """
        prompt = self.REWRITE_PROMPT.format(
            query=query,
            num_total=num_total,
            num_relevant=num_relevant,
        )
        messages = [{"role": "user", "content": prompt}]

        try:
            result = self._call_with_tool(messages, REWRITE_QUERY_TOOL)
        except Exception as e:
            logger.warning(f"Query rewrite failed: {e}. Using original query.")
            return query

        rewritten = result.get("rewritten_query", "").strip()
        strategy = result.get("strategy", "unknown")

        if not rewritten or len(rewritten) < 3:
            logger.warning(f"Rewrite too short: '{rewritten}'. Using original query.")
            return query

        logger.info(f"Query rewrite [{strategy}]: '{query}' -> '{rewritten}'")
        return rewritten
