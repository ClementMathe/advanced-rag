# Step 6.6: Generator Fix (Chat Template + Answer Length) — Experimental Results

## Overview

This document presents the evaluation results after fixing two critical issues in the LLM generator: missing chat template wrapping and excessive `max_new_tokens`. These changes produced the largest single-step quality improvement in the project.

**Key Finding**: Applying the Qwen2.5-3B-Instruct chat template and reducing `max_new_tokens` from 256 to 80 improved F1 by +24%, faithfulness by +9%, and reduced latency by 65% — all without changing the model, retrieval, or reranking.

---

## Executive Summary

| Metric | Pre-Fix | Post-Fix | Delta | Change |
|--------|---------|----------|-------|--------|
| **F1 (linear)** | 0.256 | **0.318** | +0.062 | **+24.2%** |
| **F1 (adaptive)** | 0.264 | **0.318** | +0.054 | **+20.5%** |
| **Faithfulness (linear)** | 0.729 | **0.792** | +0.063 | **+8.6%** |
| **Faithfulness (adaptive)** | 0.770 | **0.797** | +0.027 | **+3.5%** |
| **ROUGE-L (linear)** | 0.242 | **0.300** | +0.058 | **+24.0%** |
| **Latency (linear)** | 14.3s | **4.94s** | -9.4s | **-65.5%** |
| **Latency (adaptive)** | 13.4s | **6.24s** | -7.2s | **-53.4%** |
| **Exact Match** | 0.00 | **0.03** | +0.03 | 3 exact matches |

---

## What We Fixed

### Root Cause 1: Missing Chat Template

Qwen2.5-3B-Instruct expects input wrapped in `<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n` format. Without this, the model cannot distinguish the prompt from expected output, causing it to hallucinate meta-instructions ("You are an AI assistant") inside answers.

**Fix**: Added `tokenizer.apply_chat_template(tokenize=False, add_generation_prompt=True)` in `_generate_text()`. This automatically benefits all callers: `generate()`, graders, and retry prompts.

### Root Cause 2: Excessive `max_new_tokens`

The default of 256 tokens was far too high for SQuAD-style extractive QA where ground truths are typically 1-5 words. The model would generate verbose multi-sentence answers that diluted F1 scores.

**Fix**: Lowered default from 256 to 80. Answers are now focused single sentences.

### Root Cause 3: Dead Code + Missing Stop Phrases

Stop sequences were computed but never passed to `model.generate()`. Added 6 new stop phrases to `_parse_answer()` post-processing to catch remaining hallucination patterns: "You are an AI", "You are a helpful", "As an AI", "I'm an AI", "Note:", "Additional".

---

## Results (100 Queries, 3 Configs)

### Post-Fix Metrics

| Metric | Linear | Adaptive | Adaptive+Web |
|--------|--------|----------|--------------|
| **F1** | 0.318 | 0.318 | 0.322 |
| **Faithfulness** | 0.792 | 0.797 | 0.797 |
| **ROUGE-L** | 0.300 | 0.301 | 0.305 |
| **Exact Match** | 0.03 | 0.03 | 0.03 |
| **Latency (mean)** | 4.94s | 6.24s | 9.12s |
| **Latency (p50)** | 4.59s | 5.31s | 9.06s |
| **Latency (p95)** | 8.16s | 10.39s | 12.76s |
| Fallback rate | — | 60% | 60% |
| Web search rate | — | — | 39% |

### Prompt Leaking: Eliminated

Pre-fix, several predictions contained hallucinated prompt fragments ("You are an AI assistant designed to help"). Post-fix, zero instances of prompt leaking were observed across all 100 queries. The chat template cleanly separates instruction from generation.

### Exact Match: No Longer Zero

3 out of 100 queries now produce exact span matches (e.g., "Forbes", "2013 and 2014", "60 million."). The model occasionally generates concise enough answers to match ground truth verbatim. EM remains low by design — the prompt asks for natural language answers, not extracted spans.

---

## Impact on Adaptive Retrieval

The most important finding: **adaptive retrieval no longer provides meaningful improvement over linear**.

| | Pre-Fix (adaptive vs linear) | Post-Fix (adaptive vs linear) |
|---|---|---|
| F1 delta | +0.008 (+3.2%) | +0.0002 (~0%) |
| Faithfulness delta | +0.041 (+5.6%) | +0.005 (+0.6%) |
| Latency delta | -0.9s | **+1.3s** |

**Interpretation**: The generator was the bottleneck, not retrieval. With the chat template fix, the linear pipeline generates high-quality answers even from imperfect retrieval, leaving little room for adaptive retrieval to improve. The fallback retriever still triggers for 60% of queries but produces no measurable quality gain while adding 1.3s latency.

**Recommendation**: Consider raising the rerank threshold (from 0.0 to -2.0) to reduce unnecessary fallbacks, or use the linear pipeline as the default. The adaptive machinery is well-tested but no longer earns its latency cost.

---

## Qualitative Analysis

### Answer Quality (Post-Fix)

Answers are now consistently single focused sentences that paraphrase the context:

| Query | Ground Truth | Prediction | F1 |
|-------|-------------|------------|-----|
| Who is Beyonce married to? | Jay Z | Beyonce is married to Jay Z. | 0.50 |
| What was Beyonce's alter-ego? | Sasha Fierce | Beyonce's alter-ego is Sasha Fierce. | 0.57 |
| How long was Beyonce depressed? | a couple of years | Beyonce's depression lasted for a couple of years. | 0.60 |

### Remaining Error Patterns

1. **Factual errors**: "Beyonce left Destiny's Child in June 2005" (GT: 2003). The model confuses the disbanding date with the solo career start. Retrieval-level issue — the relevant chunk may not contain the right date.

2. **Wrong entity from context**: "In her music, Kanye West incorporates..." — the model answers about the wrong artist when the retrieved context contains information about multiple people.

3. **Numeric format mismatch**: GT "seven" vs prediction "7 years old" — semantically correct but scores EM=0, F1=0 due to format difference.

4. **Negation hallucinations**: "Beyonce did not record with anyone for The Best Man" (GT: Marc Nelson). The model generates a confident negative when it lacks the information.

---

## Files Modified

| File | Change |
|------|--------|
| `src/generator.py` | Chat template wrapping, `max_new_tokens` 256→80, dead code removal, 6 new stop phrases |
| `tests/test_generator.py` | 7 new tests (4 stop phrases, 3 chat template), 32 total tests passing |

**Not modified**: `src/agentic_pipeline.py`, `src/graders.py`, `src/retriever.py`, `src/reranker.py` — all automatically benefit from the `_generate_text()` fix.

---

## Lessons Learned

### 1. Use the Model's Chat Template

Instruction-tuned models require their specific chat format. Feeding raw text to Qwen2.5-3B-Instruct is equivalent to using a base model — it cannot distinguish instructions from expected output. This single fix eliminated prompt leaking and improved all metrics.

### 2. Match `max_new_tokens` to the Task

256 tokens is reasonable for open-ended generation but wasteful for extractive QA. Lowering to 80 produced shorter, more focused answers that score better on token-overlap metrics and generate 3x faster.

### 3. Fix the Generator Before Optimizing Retrieval

The adaptive retrieval gains from Step 6.5 (+3.2% F1, +5.6% faithfulness) were largely masking a generator problem. Once the generator was fixed, those gains evaporated. The lesson: always ensure the generator is working correctly before investing in retrieval optimization — a broken generator makes retrieval improvements invisible.

---

## Reproducibility

```bash
# Run evaluation (100 queries, 3 configs)
python scripts/evaluate_agentic.py

# Results saved to:
# outputs/agentic_eval/ablation_comparison.json
# outputs/agentic_eval/linear_results.json
# outputs/agentic_eval/adaptive_results.json
# outputs/agentic_eval/adaptive_web_results.json

# Run tests
pytest tests/test_generator.py -v          # 32 tests
pytest tests/test_graders.py tests/test_agentic_pipeline.py -v  # 169 tests
```
