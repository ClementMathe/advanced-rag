"""
Comprehensive metrics for RAG evaluation.

Four components:
1. MetricsCalculator — Local metrics: EM, F1, ROUGE-L, faithfulness, BERTScore
2. CostTracker — Token counting and hypothetical API cost estimation
3. TTFTMeasurer — Time to First Token measurement via LogitsProcessor injection
4. RagasEvaluator — LLM-based metrics via RAGAS + Mistral API (optional)

Design principles:
- All quality metrics return float in [0, 1] range
- Handle multiple ground truth answers (take max score)
- Lazy-load heavy models to avoid startup overhead
- Graceful degradation when optional dependencies are missing
"""

from __future__ import annotations

import os
import re
import string
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional

import numpy as np
from loguru import logger

# ---------------------------------------------------------------------------
# MetricsCalculator
# ---------------------------------------------------------------------------


class MetricsCalculator:
    """
    Compute local (free, no API) evaluation metrics for RAG answers.

    Metrics:
    - Exact Match (EM): binary, normalized text comparison
    - F1 Score: token-level precision/recall overlap
    - ROUGE-L: longest common subsequence F-measure
    - Faithfulness: embedding-based context grounding (cosine similarity)
    - BERTScore: contextual embedding similarity

    Args:
        embed_model: EmbeddingModel instance for faithfulness. None to skip.
        use_bertscore: Whether to compute BERTScore. Default True.
        bertscore_model: HuggingFace model for BERTScore.
    """

    def __init__(
        self,
        embed_model: Optional[Any] = None,
        use_bertscore: bool = True,
        bertscore_model: str = "distilbert-base-uncased",
    ) -> None:
        self.embed_model = embed_model
        self._use_bertscore = use_bertscore
        self._bertscore_model = bertscore_model
        self._bert_scorer: Optional[Any] = None

        # Lazy-loaded utilities
        self._rouge_scorer: Optional[Any] = None

    # ------------------------------------------------------------------
    # Text normalization
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize: lowercase, strip punctuation, remove articles, collapse whitespace."""
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"\b(a|an|the)\b", " ", text)
        text = " ".join(text.split())
        return text

    # ------------------------------------------------------------------
    # Answer quality metrics
    # ------------------------------------------------------------------

    def exact_match(self, prediction: str, ground_truth: str) -> float:
        """Return 1.0 if normalized texts match, 0.0 otherwise."""
        return 1.0 if self.normalize_text(prediction) == self.normalize_text(ground_truth) else 0.0

    def f1_score(self, prediction: str, ground_truth: str) -> float:
        """Token-level F1 overlap between prediction and ground truth."""
        pred_tokens = self.normalize_text(prediction).split()
        gt_tokens = self.normalize_text(ground_truth).split()

        if len(pred_tokens) == 0 or len(gt_tokens) == 0:
            return 0.0

        pred_counts = Counter(pred_tokens)
        gt_counts = Counter(gt_tokens)
        overlap = sum((pred_counts & gt_counts).values())

        if overlap == 0:
            return 0.0

        precision = overlap / len(pred_tokens)
        recall = overlap / len(gt_tokens)
        return 2 * precision * recall / (precision + recall)

    def rouge_l(self, prediction: str, ground_truth: str) -> float:
        """ROUGE-L F-measure using rouge_score library with stemming."""
        if self._rouge_scorer is None:
            from rouge_score import rouge_scorer

            self._rouge_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

        scores = self._rouge_scorer.score(ground_truth, prediction)
        return scores["rougeL"].fmeasure

    def faithfulness(
        self,
        answer: str,
        context_chunks: List[str],
        threshold: float = 0.65,
    ) -> float:
        """
        Embedding-based faithfulness: fraction of answer sentences supported by context.

        Splits answer into sentences, encodes each with embed_model, computes
        cosine similarity against context chunk embeddings. A sentence is
        "supported" if max similarity >= threshold.

        Returns 0.0 if embed_model is None or answer is empty.
        """
        if self.embed_model is None:
            return 0.0

        if not answer.strip():
            return 0.0

        try:
            from nltk.tokenize import sent_tokenize

            sentences = sent_tokenize(answer)
        except (LookupError, OSError, ImportError):
            sentences = [s.strip() for s in answer.split(".") if s.strip()]

        if not sentences:
            return 0.0

        sentence_embs = self.embed_model.encode(sentences)
        chunk_embs = self.embed_model.encode(context_chunks)

        supported_count = 0
        for sent_emb in sentence_embs:
            similarities = np.dot(chunk_embs, sent_emb) / (
                np.linalg.norm(chunk_embs, axis=1) * np.linalg.norm(sent_emb)
            )
            if np.max(similarities) >= threshold:
                supported_count += 1

        return supported_count / len(sentences)

    def bert_score(self, prediction: str, ground_truth: str) -> float:
        """
        BERTScore F1 using contextual embeddings.

        Lazy-loads the scorer on first call. Returns 0.0 if disabled or
        if the bert_score package is unavailable.
        """
        if not self._use_bertscore:
            return 0.0

        if self._bert_scorer is None:
            try:
                from bert_score import BERTScorer

                self._bert_scorer = BERTScorer(
                    model_type=self._bertscore_model,
                    lang="en",
                    rescale_with_baseline=False,
                )
                logger.info(f"BERTScorer loaded: {self._bertscore_model}")
            except ImportError:
                logger.warning("bert_score package not installed, skipping BERTScore")
                self._use_bertscore = False
                return 0.0

        _, _, f1 = self._bert_scorer.score([prediction], [ground_truth])
        return float(f1.item())

    # ------------------------------------------------------------------
    # Unified compute
    # ------------------------------------------------------------------

    def compute_all(
        self,
        prediction: str,
        ground_truth: str,
        context_chunks: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Compute all available metrics.

        Returns dict with metric_name -> score. Keys depend on what
        is available (embed_model for faithfulness, BERTScore enabled).
        """
        result: Dict[str, float] = {
            "exact_match": self.exact_match(prediction, ground_truth),
            "f1": self.f1_score(prediction, ground_truth),
            "rouge_l": self.rouge_l(prediction, ground_truth),
        }

        if context_chunks is not None and self.embed_model is not None:
            result["faithfulness"] = self.faithfulness(prediction, context_chunks)

        if self._use_bertscore:
            result["bert_score"] = self.bert_score(prediction, ground_truth)

        return result


# ---------------------------------------------------------------------------
# CostTracker
# ---------------------------------------------------------------------------

# Pricing per 1M tokens (USD), as of Feb 2026
PRICING_TABLE: Dict[str, Dict[str, float]] = {
    "qwen2.5-3b-instruct": {"input": 0.0, "output": 0.0},
    "mistral-large-latest": {"input": 2.0, "output": 6.0},
    "gpt-4o": {"input": 2.50, "output": 10.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "claude-3.5-sonnet": {"input": 3.0, "output": 15.0},
}


@dataclass
class QueryCostRecord:
    """Token usage and cost for a single query."""

    input_tokens: int
    output_tokens: int
    cost_by_model: Dict[str, float]


class CostTracker:
    """
    Track token usage and compute hypothetical API cost per query.

    For local models, counts tokens via tokenizer and shows what the
    equivalent API cost would be on various providers.

    Args:
        tokenizer: HuggingFace tokenizer for token counting.
            If None, uses whitespace approximation (~1.3 tokens/word).
        model_name: Default model for cost lookups.
    """

    def __init__(
        self,
        tokenizer: Optional[Any] = None,
        model_name: str = "qwen2.5-3b-instruct",
    ) -> None:
        self._tokenizer = tokenizer
        self._model_name = model_name
        self._records: List[QueryCostRecord] = []

    def count_tokens(self, text: str) -> int:
        """Count tokens using tokenizer or whitespace approximation."""
        if self._tokenizer is not None:
            return len(self._tokenizer.encode(text))
        # Rough approximation: 1 word ≈ 1.3 tokens
        return max(1, int(len(text.split()) * 1.3))

    def compute_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        model_name: Optional[str] = None,
    ) -> float:
        """Compute cost in USD for given token counts and model."""
        name = model_name or self._model_name
        if name not in PRICING_TABLE:
            raise ValueError(f"Unknown model: {name}. Available: {list(PRICING_TABLE.keys())}")
        pricing = PRICING_TABLE[name]
        return (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000

    def record_query(self, prompt: str, answer: str) -> QueryCostRecord:
        """
        Record token usage for a single query and compute costs.

        Args:
            prompt: Full prompt sent to the model.
            answer: Generated answer text.

        Returns:
            QueryCostRecord with token counts and costs per model.
        """
        input_tokens = self.count_tokens(prompt)
        output_tokens = self.count_tokens(answer)

        cost_by_model = {}
        for model_name in PRICING_TABLE:
            cost_by_model[model_name] = self.compute_cost(input_tokens, output_tokens, model_name)

        record = QueryCostRecord(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_by_model=cost_by_model,
        )
        self._records.append(record)
        return record

    def get_summary(self) -> Dict[str, Any]:
        """
        Aggregate cost statistics across all recorded queries.

        Returns dict with total_queries, avg_input_tokens, avg_output_tokens,
        avg_cost_per_query (by model), total_cost (by model).
        """
        if not self._records:
            return {"total_queries": 0}

        input_tokens = [r.input_tokens for r in self._records]
        output_tokens = [r.output_tokens for r in self._records]
        n = len(self._records)

        summary: Dict[str, Any] = {
            "total_queries": n,
            "avg_input_tokens": float(np.mean(input_tokens)),
            "avg_output_tokens": float(np.mean(output_tokens)),
            "total_input_tokens": int(np.sum(input_tokens)),
            "total_output_tokens": int(np.sum(output_tokens)),
            "cost_by_model": {},
        }

        for model_name in PRICING_TABLE:
            costs = [r.cost_by_model[model_name] for r in self._records]
            summary["cost_by_model"][model_name] = {
                "total_cost_usd": float(np.sum(costs)),
                "avg_cost_per_query_usd": float(np.mean(costs)),
            }

        return summary

    def reset(self) -> None:
        """Clear all recorded queries."""
        self._records.clear()


# ---------------------------------------------------------------------------
# TTFTMeasurer
# ---------------------------------------------------------------------------


@dataclass
class InstrumentedState:
    """Holds TTFT measurements from an instrumented generator."""

    last_ttft_ms: Optional[float] = None
    measurements: List[float] = field(default_factory=list)

    def record(self, ttft_ms: float) -> None:
        """Record a TTFT measurement."""
        self.last_ttft_ms = ttft_ms
        self.measurements.append(ttft_ms)

    def get_summary(self) -> Dict[str, float]:
        """Return p50/p95/p99/mean/min/max of TTFT measurements."""
        if not self.measurements:
            return {}
        arr = np.array(self.measurements)
        return {
            "ttft_mean_ms": float(np.mean(arr)),
            "ttft_p50_ms": float(np.percentile(arr, 50)),
            "ttft_p95_ms": float(np.percentile(arr, 95)),
            "ttft_p99_ms": float(np.percentile(arr, 99)),
            "ttft_min_ms": float(np.min(arr)),
            "ttft_max_ms": float(np.max(arr)),
        }


class TTFTMeasurer:
    """
    Measure Time to First Token (TTFT) for LLM generation.

    Works without modifying generator.py by injecting a custom
    LogitsProcessor into model.generate() calls. The processor
    records the timestamp when it is first invoked (= first token
    being scored).

    Usage as context manager::

        measurer = TTFTMeasurer()
        with measurer.instrument(generator) as state:
            result = generator.generate(query, chunks)
            print(f"TTFT: {state.last_ttft_ms:.1f}ms")
    """

    @contextmanager
    def instrument(self, generator: Any) -> Generator[InstrumentedState, None, None]:
        """
        Instrument a generator for TTFT measurement.

        Temporarily wraps generator.model.generate to inject a timing
        LogitsProcessor. Restores the original method on exit.

        Args:
            generator: LLMGenerator instance with .model.generate attribute.

        Yields:
            InstrumentedState with .last_ttft_ms after each generation call.
        """
        state = InstrumentedState()
        original_generate = generator.model.generate

        def wrapped_generate(*args: Any, **kwargs: Any) -> Any:
            from transformers import LogitsProcessor, LogitsProcessorList

            class _FirstTokenTimer(LogitsProcessor):
                def __init__(self) -> None:
                    self.first_token_time: Optional[float] = None
                    self.start_time: float = time.perf_counter()

                def __call__(self, input_ids: Any, scores: Any) -> Any:
                    if self.first_token_time is None:
                        self.first_token_time = time.perf_counter()
                    return scores

            timer = _FirstTokenTimer()

            # Inject our processor into the logits_processor list
            existing = kwargs.get("logits_processor", None)
            if existing is None:
                existing = LogitsProcessorList()
            elif not isinstance(existing, LogitsProcessorList):
                existing = LogitsProcessorList(existing)
            existing.append(timer)
            kwargs["logits_processor"] = existing

            result = original_generate(*args, **kwargs)

            # Record TTFT
            if timer.first_token_time is not None:
                ttft_ms = (timer.first_token_time - timer.start_time) * 1000
                state.record(ttft_ms)

            return result

        generator.model.generate = wrapped_generate
        try:
            yield state
        finally:
            generator.model.generate = original_generate


# ---------------------------------------------------------------------------
# RagasEvaluator
# ---------------------------------------------------------------------------


class RagasEvaluator:
    """
    LLM-based RAG evaluation via RAGAS library with Mistral API.

    Gracefully degrades when API key or dependencies are missing.

    Metrics:
    - Faithfulness: LLM judges if answer is grounded in context
    - Answer Correctness: LLM judges if answer matches reference
    - Context Precision: relevance of retrieved contexts
    - Context Recall: coverage of retrieved contexts

    Args:
        api_key: Mistral API key. If None, checks MISTRAL_API_KEY env var.
        model_name: Mistral model for evaluation.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "mistral-large-latest",
    ) -> None:
        self._available = False
        self._llm = None
        self._model_name = model_name

        resolved_key = api_key or os.environ.get("MISTRAL_API_KEY")
        if not resolved_key:
            logger.info("No Mistral API key found — RAGAS LLM metrics disabled")
            return

        try:
            from mistralai import Mistral
            from ragas.llms import llm_factory

            client = Mistral(api_key=resolved_key)
            self._llm = llm_factory(model_name, provider="mistral", client=client)
            self._available = True
            logger.info(f"RAGAS evaluator initialized with {model_name}")
        except ImportError as e:
            logger.warning(f"Missing dependency for RAGAS+Mistral: {e}")
        except Exception as e:
            logger.warning(f"Failed to initialize RAGAS evaluator: {e}")

    @property
    def is_available(self) -> bool:
        """Whether RAGAS LLM metrics are available."""
        return self._available

    def evaluate_batch(
        self,
        samples: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """
        Evaluate a batch of samples using RAGAS.

        Each sample dict should have:
        - query: str
        - answer: str
        - contexts: List[str]
        - ground_truth: str (optional but recommended)

        Returns dict with per-metric mean scores. Empty dict if unavailable.
        """
        if not self._available:
            return {}

        try:
            from ragas import evaluate
            from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
            from ragas.metrics import (
                AnswerCorrectness,
                Faithfulness,
                LLMContextPrecisionWithReference,
                LLMContextRecall,
            )

            ragas_samples = []
            for s in samples:
                ragas_samples.append(
                    SingleTurnSample(
                        user_input=s["query"],
                        response=s["answer"],
                        retrieved_contexts=s["contexts"],
                        reference=s.get("ground_truth", ""),
                    )
                )

            dataset = EvaluationDataset(samples=ragas_samples)
            metrics = [
                Faithfulness(llm=self._llm),
                AnswerCorrectness(llm=self._llm),
                LLMContextPrecisionWithReference(llm=self._llm),
                LLMContextRecall(llm=self._llm),
            ]

            result = evaluate(dataset=dataset, metrics=metrics)
            return dict(result)
        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}")
            return {}
