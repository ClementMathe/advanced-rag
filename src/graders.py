"""
Document relevance grading and query rewriting for Agentic RAG.

This module provides LLM-based utilities for the self-correcting RAG pipeline:
- DocumentGrader: Assesses whether retrieved documents are relevant to a query
- QueryRewriter: Rephrases queries to improve retrieval quality on retry

Both classes reuse the existing LLMGenerator to avoid loading additional models.
"""

import json
import re
from typing import List, Optional

from loguru import logger

from src.generator import LLMGenerator


class DocumentGrader:
    """
    Grades document relevance using LLM binary classification.

    Uses the same LLM as generation (Qwen2.5-3B) to determine whether
    each retrieved document is relevant to the user's query. Supports
    both single-document grading (for debugging) and efficient batch
    grading (for production use).

    Attributes:
        generator: LLMGenerator instance for inference.
        single_prompt: Prompt template for grading one document.
        batch_prompt: Prompt template for grading multiple documents.
    """

    SINGLE_GRADING_PROMPT = (
        "You are an expert document relevance grader.\n"
        "\n"
        "Your task: Determine if the following document is relevant to the user's question.\n"
        "\n"
        "Guidelines:\n"
        '- "Relevant" = The document contains information that could help answer the question\n'
        "- Look for semantic meaning, not just keyword matching\n"
        '- Even partial relevance counts as "yes"\n'
        "\n"
        "Question: {query}\n"
        "\n"
        "Document: {document}\n"
        "\n"
        'Is this document relevant? Respond with ONLY "yes" or "no".'
    )

    BATCH_GRADING_PROMPT = (
        "You are an expert document relevance grader.\n"
        "\n"
        "Your task: Determine if each document is relevant to the user's question.\n"
        "\n"
        "Guidelines:\n"
        '- "Relevant" = The document contains information that could help answer the question\n'
        "- Look for semantic meaning, not just keyword matching\n"
        "- Even partial relevance counts as true\n"
        "\n"
        "Question: {query}\n"
        "\n"
        "Documents:\n"
        "{documents}\n"
        "\n"
        "Respond with ONLY a JSON object in this exact format (no other text):\n"
        '{{"grades": [true, false, true, ...]}}\n'
        "\n"
        "The array must have exactly {num_docs} boolean values, one per document."
    )

    def __init__(self, generator: LLMGenerator):
        """
        Initialize the document grader.

        Args:
            generator: An already-loaded LLMGenerator instance. The grader
                      reuses this model to avoid additional VRAM usage.
        """
        self.generator = generator

    def grade_single(self, query: str, document: str) -> bool:
        """
        Grade a single document for relevance to the query.

        Uses zero-shot binary classification: the LLM responds with "yes" or "no".
        Useful for debugging and testing, but less efficient than grade_batch()
        for multiple documents.

        Args:
            query: The user's search query.
            document: The text content of the retrieved document.

        Returns:
            True if the document is relevant, False otherwise.
        """
        prompt = self.SINGLE_GRADING_PROMPT.format(query=query, document=document)

        # Save and override max_new_tokens for short response
        original_max_tokens = self.generator.max_new_tokens
        self.generator.max_new_tokens = 10
        try:
            response = self.generator._generate_text(prompt)
        finally:
            self.generator.max_new_tokens = original_max_tokens

        return self._parse_single_grade(response)

    def grade_batch(self, query: str, documents: List[str]) -> List[bool]:
        """
        Grade multiple documents in a single LLM call.

        This is the preferred method for production use. All documents are
        included in one prompt, and the LLM returns a JSON array of grades.
        ~6x faster than calling grade_single() per document.

        Args:
            query: The user's search query.
            documents: List of document text contents to grade.

        Returns:
            List of boolean grades (True=relevant, False=not), same length
            as the input documents list. Falls back to all-True if parsing fails.
        """
        if not documents:
            return []

        # Format numbered document list
        docs_text = "\n".join(f"{i+1}. {doc[:500]}" for i, doc in enumerate(documents))

        prompt = self.BATCH_GRADING_PROMPT.format(
            query=query,
            documents=docs_text,
            num_docs=len(documents),
        )

        # Save and override max_new_tokens for structured response
        original_max_tokens = self.generator.max_new_tokens
        self.generator.max_new_tokens = 100
        try:
            response = self.generator._generate_text(prompt)
        finally:
            self.generator.max_new_tokens = original_max_tokens

        return self._parse_batch_grades(response, len(documents))

    def _parse_single_grade(self, response: str) -> bool:
        """
        Parse a single grading response into a boolean.

        Looks for "yes" or "no" in the response. Defaults to True (permissive)
        if the response is ambiguous, to avoid filtering out potentially
        useful documents.

        Args:
            response: Raw LLM output text.

        Returns:
            True if relevant, False if not.
        """
        cleaned = response.strip().lower()
        logger.debug(f"Grading response: '{cleaned}'")

        if "no" in cleaned and "yes" not in cleaned:
            return False
        if "yes" in cleaned:
            return True

        # Ambiguous response: default to relevant (permissive)
        logger.warning(f"Ambiguous grading response: '{cleaned}', defaulting to relevant")
        return True

    def _parse_batch_grades(self, response: str, expected_count: int) -> List[bool]:
        """
        Parse a batch grading JSON response into a list of booleans.

        Attempts multiple parsing strategies:
        1. Direct JSON parsing of the full response
        2. Extract JSON object from mixed text using regex
        3. Fallback: return all True (permissive)

        Args:
            response: Raw LLM output containing JSON.
            expected_count: Expected number of grades (for validation).

        Returns:
            List of boolean grades matching expected_count length.
        """
        cleaned = response.strip()
        logger.debug(f"Batch grading response: '{cleaned}'")

        # Strategy 1: Direct JSON parse
        grades = self._try_parse_json(cleaned, expected_count)
        if grades is not None:
            return grades

        # Strategy 2: Extract JSON from mixed text
        json_match = re.search(r'\{[^{}]*"grades"\s*:\s*\[[^\]]*\][^{}]*\}', cleaned)
        if json_match:
            grades = self._try_parse_json(json_match.group(), expected_count)
            if grades is not None:
                return grades

        # Strategy 3: Look for array of true/false values directly
        bool_matches = re.findall(r"\b(true|false)\b", cleaned, re.IGNORECASE)
        if len(bool_matches) == expected_count:
            grades = [m.lower() == "true" for m in bool_matches]
            logger.debug(f"Parsed grades from boolean pattern: {grades}")
            return grades

        # Fallback: all relevant (permissive)
        logger.warning(
            f"Could not parse batch grades from: '{cleaned}'. "
            f"Defaulting to all relevant ({expected_count} docs)."
        )
        return [True] * expected_count

    def _try_parse_json(self, text: str, expected_count: int) -> Optional[List[bool]]:
        """
        Attempt to parse JSON grades from text.

        Args:
            text: JSON string to parse.
            expected_count: Expected number of grade values.

        Returns:
            List of booleans if parsing succeeds and count matches, None otherwise.
        """
        try:
            data = json.loads(text)
            if isinstance(data, dict) and "grades" in data:
                grades = [bool(g) for g in data["grades"]]
                if len(grades) == expected_count:
                    logger.debug(f"Parsed batch grades: {grades}")
                    return grades
                else:
                    logger.warning(
                        f"Grade count mismatch: got {len(grades)}, " f"expected {expected_count}"
                    )
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
        return None


class QueryRewriter:
    """
    Rewrites search queries using LLM to improve retrieval quality.

    When retrieved documents score poorly in relevance grading, the
    QueryRewriter generates an improved version of the query by:
    - Resolving pronouns and vague references
    - Adding specific names, dates, or context
    - Clarifying ambiguous terminology

    Attributes:
        generator: LLMGenerator instance for inference.
        rewrite_prompt: Prompt template for query rewriting.
    """

    REWRITE_PROMPT = (
        "You are a search query optimizer.\n"
        "\n"
        "The original query retrieved {num_total} documents, but only {num_relevant} were relevant.\n"
        "\n"
        "Improve the query to get better search results.\n"
        "\n"
        "Strategies:\n"
        "1. Resolve pronouns (she \u2192 Beyonc\u00e9)\n"
        '2. Add context (debut \u2192 debut solo album "Dangerously in Love")\n'
        "3. Clarify ambiguity (awards \u2192 Grammy awards)\n"
        "\n"
        "Original query: {query}\n"
        "\n"
        "Improved query (one line, no explanation):"
    )

    def __init__(self, generator: LLMGenerator):
        """
        Initialize the query rewriter.

        Args:
            generator: An already-loaded LLMGenerator instance. The rewriter
                      reuses this model to avoid additional VRAM usage.
        """
        self.generator = generator

    def rewrite(self, query: str, num_total: int, num_relevant: int) -> str:
        """
        Generate an improved search query.

        Takes the original query and retrieval statistics, then uses the
        LLM to produce a more specific, unambiguous version of the query
        that should yield better retrieval results.

        Args:
            query: The current search query (original or previously rewritten).
            num_total: Total number of documents retrieved.
            num_relevant: Number of documents that passed relevance grading.

        Returns:
            The rewritten query string. Returns the original query if
            rewriting fails or produces an empty result.
        """
        prompt = self.REWRITE_PROMPT.format(
            query=query,
            num_total=num_total,
            num_relevant=num_relevant,
        )

        # Save and override max_new_tokens for short response
        original_max_tokens = self.generator.max_new_tokens
        self.generator.max_new_tokens = 50
        try:
            response = self.generator._generate_text(prompt)
        finally:
            self.generator.max_new_tokens = original_max_tokens

        rewritten = self._parse_rewrite(response, query)
        logger.info(f"Query rewrite: '{query}' -> '{rewritten}'")
        return rewritten

    def _parse_rewrite(self, response: str, original_query: str) -> str:
        """
        Parse and clean the rewritten query from LLM output.

        Extracts the first meaningful line from the response, removing
        any explanations, quotes, or prefixes the LLM may have added.

        Args:
            response: Raw LLM output text.
            original_query: The original query, used as fallback.

        Returns:
            Cleaned rewritten query, or original query if parsing fails.
        """
        cleaned = response.strip()

        if not cleaned:
            logger.warning("Empty rewrite response, returning original query")
            return original_query

        # Take only the first line (ignore explanations)
        first_line = cleaned.split("\n")[0].strip()

        # Remove common prefixes the LLM might add
        prefixes_to_remove = [
            "Improved query:",
            "Rewritten query:",
            "Better query:",
            "Query:",
        ]
        for prefix in prefixes_to_remove:
            if first_line.lower().startswith(prefix.lower()):
                first_line = first_line[len(prefix) :].strip()

        # Remove surrounding quotes if present
        if (first_line.startswith('"') and first_line.endswith('"')) or (
            first_line.startswith("'") and first_line.endswith("'")
        ):
            first_line = first_line[1:-1].strip()

        # Fallback if result is empty or too short
        if len(first_line) < 3:
            logger.warning(f"Rewrite too short: '{first_line}', returning original")
            return original_query

        return first_line
