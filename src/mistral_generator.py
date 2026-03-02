"""
Mistral API-based answer generation for the Advanced RAG system.

Drop-in replacement for LLMGenerator that calls the Mistral API instead of
running a local model. Same public interface: generate() + build_prompt().

Key differences from LLMGenerator:
- Uses mistralai.Mistral client (no tokenizer, no VRAM)
- Built-in exponential backoff for free-tier rate limits (HTTP 429)
- Returns input_tokens + output_tokens in result dict for cost tracking
"""

import os
import time
from typing import Dict, List, Optional

from loguru import logger
from mistralai import Mistral
from mistralai.models import SDKError


class MistralAPIGenerator:
    """
    API-based answer generator that calls the Mistral chat completion endpoint.

    Implements the same public interface as LLMGenerator (generate, build_prompt)
    so it can be used as a drop-in replacement in evaluation scripts.

    Attributes:
        model_name: Mistral model identifier (e.g. "mistral-small-latest").
        max_new_tokens: Maximum tokens in the generated response.
        temperature: Sampling temperature (lower = more deterministic).
        max_retries: Number of retry attempts on rate limit / transient errors.
        retry_delay: Base delay in seconds for exponential backoff.
    """

    # Same prompt template as LLMGenerator.DEFAULT_PROMPT — ensures quality
    # differences between generators reflect the model, not the prompt.
    DEFAULT_PROMPT = """Answer the question using ONLY the provided context. Be direct and concise (1-3 sentences maximum).

DO NOT:
- Repeat the question
- Say "Based on the context" or similar phrases
- Continue after answering

Context:
{context}

Question: {question}

Direct answer:"""

    def __init__(
        self,
        model_name: str = "mistral-small-latest",
        api_key: Optional[str] = None,
        max_new_tokens: int = 80,
        temperature: float = 0.1,
        max_retries: int = 5,
        retry_delay: float = 1.2,
    ):
        """
        Initialize the Mistral API generator.

        Args:
            model_name: Mistral model identifier.
            api_key: Mistral API key. Falls back to MISTRAL_API_KEY env var.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            max_retries: Maximum retry attempts on transient errors.
            retry_delay: Base delay (seconds) for exponential backoff.

        Raises:
            ValueError: If no API key is available.
        """
        resolved_key = api_key or os.environ.get("MISTRAL_API_KEY")
        if not resolved_key:
            raise ValueError(
                "No Mistral API key provided. "
                "Pass api_key= or set the MISTRAL_API_KEY environment variable."
            )

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.client = Mistral(api_key=resolved_key)

        logger.info(f"MistralAPIGenerator initialized: model={model_name}")

    # ------------------------------------------------------------------
    # Public interface (same as LLMGenerator)
    # ------------------------------------------------------------------

    def generate(
        self,
        query: str,
        context_chunks: List[Dict],
        max_chunks: int = 5,
    ) -> Dict:
        """
        Generate an answer from query and context chunks via the Mistral API.

        Args:
            query: User question.
            context_chunks: Retrieved chunks (dicts with 'content' key).
            max_chunks: Maximum number of chunks to include in the prompt.

        Returns:
            Dictionary with keys:
                - answer (str): Generated answer.
                - prompt (str): Full prompt sent to the API.
                - input_tokens (int): Tokens in the prompt.
                - output_tokens (int): Tokens in the response.
                - model (str): Model identifier used.
                - num_chunks_used (int): Chunks included in the prompt.
        """
        prompt = self.build_prompt(query, context_chunks, max_chunks)
        response = self._call_api_with_retry(prompt)

        content = response.choices[0].message.content or ""
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0

        return {
            "answer": content.strip(),
            "prompt": prompt,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "model": self.model_name,
            "num_chunks_used": min(len(context_chunks), max_chunks),
        }

    def build_prompt(
        self,
        query: str,
        context_chunks: List[Dict],
        max_chunks: int = 5,
    ) -> str:
        """
        Build the full prompt string (identical format to LLMGenerator).

        Args:
            query: User question.
            context_chunks: Retrieved chunks (dicts with 'content' key).
            max_chunks: Maximum chunks to include.

        Returns:
            Formatted prompt string.
        """
        context = self._format_context(context_chunks[:max_chunks])
        return self.DEFAULT_PROMPT.format(context=context, question=query)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _format_context(self, chunks: List[Dict]) -> str:
        """Format context chunks into a numbered string (matches LLMGenerator)."""
        return "\n\n".join(f"[{i}] {chunk.get('content', '')}" for i, chunk in enumerate(chunks, 1))

    def _call_api_with_retry(self, prompt: str):
        """
        Call the Mistral chat completion API with exponential backoff.

        Retries on HTTP 429 (rate limit) and transient SDKErrors.

        Args:
            prompt: The prompt to send as a user message.

        Returns:
            Mistral chat completion response object.

        Raises:
            SDKError: If all retries are exhausted.
            Exception: On non-retryable errors.
        """
        messages = [{"role": "user", "content": prompt}]

        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.chat.complete(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                )
                return response

            except SDKError as e:
                is_rate_limit = (
                    hasattr(e, "raw_response")
                    and e.raw_response is not None
                    and e.raw_response.status_code == 429
                )
                is_retryable = is_rate_limit or (
                    hasattr(e, "raw_response")
                    and e.raw_response is not None
                    and e.raw_response.status_code >= 500
                )

                if is_retryable and attempt < self.max_retries:
                    delay = self.retry_delay * (2**attempt)
                    logger.warning(
                        f"Mistral API error (attempt {attempt + 1}/{self.max_retries + 1}): "
                        f"{e}. Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                else:
                    raise

            except Exception as e:
                # Connection errors etc. — retry with backoff
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2**attempt)
                    logger.warning(
                        f"Unexpected error (attempt {attempt + 1}/{self.max_retries + 1}): "
                        f"{e}. Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                else:
                    raise
