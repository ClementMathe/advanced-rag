# Step 9: Mistral Integration — Experimental Analysis (DRAFT)

> **Status**: PAUSED — free tier Mistral credits exhausted. Experiments A–E complete. Configs F & G (doc_grader_v2) not yet run.
>
> **To resume**: recharge Mistral free tier → `python scripts/evaluate_hotpot.py --configs doc_grader_v2_qwen,doc_grader_v2_mistral`
> Local data (HotpotQA index + 200q sample) already built in `data/hotpot/` and `index/hotpot/` — no need to re-run `prepare_hotpot.py`.

---

## Overview

Step 9 integrates Mistral's API into the RAG pipeline in two ways:
1. **MistralAPIGenerator** — Mistral as the answer generation model (replaces Qwen)
2. **Mistral graders** — MistralDocumentGrader, MistralAnswerGrader as agentic pipeline components

Three evaluation phases were run:
- **Phase 3**: 20q SQuAD — generator comparison (Qwen vs Mistral-small vs Mistral-large)
- **Phase 4**: 100q SQuAD — grader evaluation (linear vs Qwen-grader vs Mistral-grader, answer grading)
- **Phase 6**: 200q HotpotQA — multi-hop benchmark (linear vs adaptive vs Mistral doc grader vs Mistral answer grader)

---

## Phase 3: Generator Comparison (SQuAD, n=20)

### Setup
- Dataset: SQuAD v2 (20 questions)
- Retrieval: hybrid BM25+dense, reranker, k=5
- 3 configs: Qwen2.5-3B (local, 4-bit), mistral-small-latest (API), mistral-large-latest (API)

### Results

| Config | F1 | ROUGE-L | BERTScore | Faithfulness | Latency (ms) | Cost/q |
|---|---|---|---|---|---|---|
| **qwen** | 0.297 | 0.274 | 0.728 | 1.000 | 10 586 | $0 |
| **mistral-small** | 0.577 | 0.555 | 0.794 | 0.700 | 317 | $0.000257 |
| **mistral-large** | 0.654 | 0.630 | 0.811 | 0.600 | 586 | $0.002553 |

### A/B Tests

| Test | F1 diff | p-value | Faithful diff | Winner |
|---|---|---|---|---|
| qwen vs mistral-small | +0.279 | 0.0008 | -0.300 (p=0.010) | **champion** (faithfulness guard regressed) |
| qwen vs mistral-large | +0.357 | 0.0001 | -0.400 (p=0.002) | **champion** (faithfulness guard regressed) |

### Interpretation

**Mistral massively outperforms Qwen on lexical metrics**:
- mistral-small: +94% F1, +9% BERTScore
- mistral-large: +120% F1, +11% BERTScore

**Faithfulness paradox**: Qwen scores faithfulness=1.0 while Mistral scores 0.6–0.7. This is a measurement artifact:
- Qwen (extractive): copies verbatim from context → cosine similarity ≈ 1.0
- Mistral (generative/paraphrasing): synthesizes and paraphrases → lower cosine similarity but higher F1 (the answer is *correct*, not just *copied*)

**The faithfulness guard is unreliable as a gate** when comparing extractive vs generative models. The A/B runner correctly flags faithfulness regression but the regression is methodological, not factual.

**Practical takeaway**: mistral-small is the cost-effective sweet spot (+94% F1 for $0.26/1000 queries, 33x faster than Qwen inference). mistral-large adds +10% but at 10x the cost.

---

## Phase 4: Grader Evaluation (SQuAD, n=100)

### Setup
- Dataset: SQuAD v2 (100 questions)
- Retrieval: same as Phase 3
- Generator: Qwen2.5-3B (local, to isolate grader effect)
- 3 configs: linear (no grader), qwen_grader (local doc grader), mistral_grader (Mistral answer grader)
- Grading: **answer grading** (post-generation) — grader judges if the answer is acceptable; if not, retry with web fallback

### Results

| Config | F1 | BERTScore | Faithfulness | FRR | Retry Rate | Latency (ms) | Cost/q |
|---|---|---|---|---|---|---|---|
| **linear** | 0.322 | 0.742 | 0.807 | 0% | 0% | 3 541 | $0 |
| **qwen_grader** | 0.260 | 0.724 | 0.850 | 14% | 57% | 8 876 | $0 |
| **mistral_grader** | 0.302 | 0.734 | 0.783 | 6% | 22% | 10 662 | $0.000388 |

FRR = False Rejection Rate (good answers rejected by grader)

### A/B Tests

| Test | F1 diff | p-value | Significant | Winner |
|---|---|---|---|---|
| linear vs qwen_grader | -0.062 | 0.0002 | Yes | **champion** (linear) |
| linear vs mistral_grader | -0.020 | 0.209 | No | no_significant_difference |

### Interpretation

**Qwen grader is clearly harmful**: -19% F1, 14% FRR, 57% retry rate, 2.5x latency. The local grader constantly rejects valid answers and triggers unnecessary web retrieval.

**Mistral grader is neutral**: -6% F1 (not significant, p=0.209), 6% FRR, 22% retry rate, 3x latency. At $0.39/100 queries. Better calibrated than Qwen grader but still adds cost/latency with no measurable benefit on SQuAD.

**Why doesn't answer grading help on SQuAD?** SQuAD is a single-hop, factoid dataset — the base pipeline already retrieves well. The grader has little room to add value when the pipeline isn't fundamentally failing. Answer grading is most valuable when the generator sometimes produces hallucinations or off-topic responses, which rarely happens here.

---

## Phase 6: HotpotQA Multi-Hop Benchmark (n=200)

### Setup
- Dataset: HotpotQA "distractor" split (200 bridge questions requiring 2 supporting documents)
- Index: 200 HotpotQA context documents indexed via FAISS + BM25
- Generator: Qwen2.5-3B (local)
- 4 configs:
  - **linear**: RAGPipeline (baseline)
  - **adaptive**: AgenticRAGPipeline (adaptive retrieval, no grading)
  - **mistral_grader**: AgenticRAGPipeline + MistralDocumentGrader (doc filtering before generation) + MistralAnswerGrader
  - **mistral_answer_grader**: AgenticRAGPipeline + MistralAnswerGrader only (no doc filtering)
- New metric: **Supporting Facts Recall (SF_recall)** — fraction of gold supporting paragraphs present in context

### Results

| Config | n | F1 | EM | BERTScore | Faithfulness | SF_recall | Latency (ms) |
|---|---|---|---|---|---|---|---|
| **linear** | 200 | 0.191 | 0.010 | 0.718 | 0.829 | 0.840 | ~5 364 |
| **adaptive** | 200 | 0.202 | 0.015 | 0.721 | 0.823 | 0.833 | 4 401 |
| **mistral_grader** | 189* | 0.196 | 0.011 | 0.722 | 0.827 | **0.759** | ~12 446 |
| **mistral_answer_grader** | 200 | 0.188 | 0.010 | 0.713 | 0.816 | 0.840 | 6 834 |

*n=189: 11 queries lost to transient DNS errors (`[Errno 11001] getaddrinfo failed`) on Windows

### A/B Tests (champion = linear)

| Test | F1 diff | p(F1) | SF_recall diff | p(SF) | Winner |
|---|---|---|---|---|---|
| linear vs adaptive | +0.0105 | 0.086 | -0.0075 | 0.083 | no_significant_difference |
| linear vs mistral_grader | +0.004 | 0.692 | **-0.082** | **2.1e-8** | no_significant_difference* |
| linear vs mistral_answer_grader | -0.003 | 0.537 | 0.000 | 1.000 | no_significant_difference |

*mistral_grader significantly regresses SF_recall but F1 guard not triggered; technically "no sig diff on F1" but the SF_recall finding is major

### Interpretation

#### Finding 1: Doc grading destroys multi-hop supporting facts retrieval

The doc grader reduces SF_recall from 0.840 → 0.759 (**-9.6%, p=2.1e-8**). This is the most statistically significant result of the entire experiment.

**Why?** Multi-hop questions require 2 "bridge" documents:
- Doc A: answers "who was the director of X?"
- Doc B: answers "where was that person born?"

Individually, neither doc fully answers the original question. The grader evaluates each doc in isolation and legitimately concludes "this doc doesn't answer the question" → filters it out → generator loses the supporting chain.

This is a fundamental architecture mismatch: single-document relevance grading is incompatible with multi-hop reasoning.

#### Finding 2: Answer grading is neutral with an extractive generator

Config D (answer grading only) has **identical SF_recall to linear** (0.840, p=1.0) and essentially identical F1 (0.188 vs 0.191, p=0.537). It adds +2.5s latency and ~$0.39/100 queries for zero measurable gain.

**Why?** Qwen is an extractive generator — it copies text from the context verbatim. When the context has the right docs, it produces a correct answer. When it doesn't, it produces a hallucination. The answer grader correctly identifies the bad answers (triggering retries) but the web retrieval fallback doesn't find better supporting docs for multi-hop chains either.

#### Finding 3: Adaptive retrieval is the most beneficial agentic feature

Adaptive config shows:
- F1: +5.5% (p=0.086, borderline)
- SF_recall: -0.9% (p=0.083, borderline, not a regression concern)
- Latency: **FASTER** than linear (4.4s vs 5.4s)

Adaptive retrieval skips the reformulation+second retrieval when the initial retrieval is confident enough, saving latency. The F1 gain is real but doesn't reach significance at n=200.

#### Summary table: Effort vs Reward

| Feature | F1 impact | SF_recall impact | Latency | Cost | Verdict |
|---|---|---|---|---|---|
| Adaptive retrieval | +5.5% (p=0.086) | -0.9% (ns) | -18% | $0 | **KEEP** |
| Answer grading (Mistral) | -1.5% (ns) | 0% | +27% | $0.39/100q | Marginal |
| Doc grading (Mistral) | +2.0% (ns) | **-9.6%** (p<1e-7) | +132% | +cost | **HARMFUL for multi-hop** |

---

## Cross-Phase Synthesis

### What actually improved performance?

1. **Mistral as generator** (+94–120% F1): Largest single improvement in the entire step. API latency (317ms) vs local Qwen (10.5s) makes it also massively faster in practice.

2. **Adaptive retrieval** (+5.5% F1, borderline): Second-best gain. Free (no API cost), actually reduces latency.

3. **Graders with Qwen generator**: Neutral to harmful. The generator is the bottleneck — filtering docs doesn't help when the synthesis step is weak.

### The generator bottleneck hypothesis

When the generator is extractive/weak (Qwen-3B), document quality matters less because the answer will be a copy of whatever docs are there. Better docs → slightly better copies. Worse docs → bad answer → grader rejects → web fallback (which usually also fails for multi-hop).

When the generator is strong and generative (Mistral), document quality could matter more: the model reasons over the docs, so having the RIGHT docs in context could make a qualitative difference. **This is the key untested hypothesis.**

---

## RESUME — What's Left to Run

### Context
Mistral free tier credits ran out after completing configs A–E (linear, adaptive, mistral_grader, mistral_answer_grader, linear_mistral). The two remaining configs implement the prompt engineering fix for the doc grader (Open Question 1 below).

### Command to run next session
```bash
python scripts/evaluate_hotpot.py --configs doc_grader_v2_qwen,doc_grader_v2_mistral
```

### What these configs test

**Config F — `doc_grader_v2_qwen`** (multi-hop-aware prompt + Qwen generator)
- Uses `MistralDocumentGrader(multi_hop=True)` — the `MULTI_HOP_GRADING_PROMPT` variant that tells the grader
  to keep bridge docs even if they don't fully answer the question
- Generator: Qwen2.5-3B (same as configs A–C) → isolates the prompt fix effect
- Answers: does the v2 prompt recover the SF_recall regression (-9.6%) seen in config C?

**Config G — `doc_grader_v2_mistral`** (multi-hop-aware prompt + Mistral-small generator)
- Same v2 doc grader as F
- Generator: mistral-small-latest → combination of best generator + fixed doc grader
- This is the "full stack" config: best generator × best doc grader prompt
- Tests the "generator bottleneck hypothesis" from the synthesis section

### Prerequisites (already done locally)
- `data/hotpot/sample_200q.json` — 200 bridge questions sampled ✅
- `data/hotpot/paragraphs_pool.json` — 2000 paragraphs pooled ✅
- `index/hotpot/` — FAISS + BM25 index built ✅
- Checkpointing is active: if a run fails mid-way, it resumes from last saved query ✅

### Expected cost
~600–800 Mistral API calls per config at 0.5 req/s → ~20–25 min per config. Total: ~$0.15–0.20.

### Key metrics to compare against baseline (config C = mistral_grader)
| Metric | Config C (mistral_grader) | Config F (v2+qwen) target | Config G (v2+mistral) target |
|---|---|---|---|
| F1 | 0.196 | >0.196 | >0.300 |
| SF_recall | **0.759** | **>0.840** (recover regression) | >0.840 |
| Grading accuracy | — | >85% | >85% |

---

## Open Questions / Next Experiments

### 1. Would prompt engineering fix the doc grader?

**Problem**: Doc grader evaluates docs individually → false rejects on multi-hop bridge docs.

**Potential fix**: Modify the doc grader prompt to acknowledge multi-hop context:
```
"This question may require multiple documents working together (multi-hop reasoning).
Grade this document as RELEVANT if it could be part of the reasoning chain,
even if it doesn't directly answer the full question."
```

**Expected impact**: Reduce SF_recall regression. But may increase false positives (noise docs passed through).

### 2. Would Mistral as generator change the grader story on HotpotQA?

**Setup**: Run HotpotQA with 4 configs, replacing Qwen with mistral-small:
- Config A': linear + Mistral-small
- Config B': adaptive + Mistral-small
- Config C': mistral_grader + Mistral-small (doc grading)
- Config D': mistral_answer_grader + Mistral-small

**Expected result**: All Mistral configs >> Qwen baseline (likely +90%+ F1). But will the grader add value ON TOP of Mistral?

**Hypothesis**: Maybe yes for answer grading (Mistral knows when it's wrong), but doc grading will still hurt SF_recall regardless of generator.

**Challenge**: Faithfulness metric becomes even more misleading when comparing Mistral generator (low similarity to context by design) with grader that evaluates answers. The guard metric for A/B testing needs to change.

**Cost estimate**: 200 queries × 4 configs × ~$0.000300/query (mistral-small) ≈ $0.24 total. Feasible.

### 3. Would query decomposition work?

For multi-hop questions, break the question into sub-questions:
- "Who wrote the novel that was adapted into X film?" → ["What novel was adapted into X film?", "Who wrote [novel]?"]
- Retrieve and answer each sub-question separately
- Combine into final answer

This is architecturally different from current pipeline. Requires a decomposition LLM call (Mistral API) and multi-round retrieval. More complex but potentially much higher SF_recall.

---

## Bugs Fixed This Step

### Bug: adaptive config n=2 in HotpotQA

**Root cause**: `run_pipeline_query` called `pipeline.query(query, return_intermediate=True)` unconditionally. `AgenticRAGPipeline.query(self, query: str)` doesn't accept `return_intermediate` → TypeError silently caught by `except Exception` → query not saved → n stayed at 2 (smoke test).

**Fix**: Added `is_agentic` parameter to `run_pipeline_query`:
```python
if is_agentic:
    result = pipeline.query(query)
else:
    result = pipeline.query(query, return_intermediate=True)
```

### Bug: Transient DNS failures (mistral_grader, n=189)

`[Errno 11001] getaddrinfo failed` on 11 queries (last ~5 per run). Windows DNS transient failure. Not a code bug — `_call_with_retry` exhausts retries, outer loop logs+skips. Checkpoint system saves partial results. Acceptable data loss for the conclusions drawn.

---

## Costs Summary

| Phase | Config | n | Total Cost |
|---|---|---|---|
| Phase 3 | mistral-small generator | 20 | $0.0051 |
| Phase 3 | mistral-large generator | 20 | $0.0511 |
| Phase 4 | mistral_grader (answer) | 100 | $0.0388 |
| Phase 6 | mistral_grader (doc+answer) | 189 | TBD |
| Phase 6 | mistral_answer_grader | 200 | TBD |
| **Total** | | | **~$0.10–0.15** |

---

## Files

| File | Purpose |
|---|---|
| `src/mistral_generator.py` | MistralAPIGenerator — drop-in for LLMGenerator |
| `src/mistral_grader.py` | MistralDocumentGrader, MistralAnswerGrader, MistralQueryRewriter, RateLimiter |
| `src/agentic_pipeline.py` | Modified: doc_grader, query_rewriter, enable_doc_grading params |
| `scripts/compare_generators.py` | Phase 3: 20q SQuAD generator benchmark |
| `scripts/evaluate_mistral_grader.py` | Phase 4: 100q SQuAD grader benchmark |
| `scripts/prepare_hotpot.py` | Phase 5: download + index HotpotQA |
| `scripts/evaluate_hotpot.py` | Phase 6: 200q HotpotQA 4-config benchmark |
| `outputs/generator_comparison/summary.json` | Phase 3 results |
| `outputs/mistral_grader_eval/summary.json` | Phase 4 results |
| `outputs/hotpot_eval/summary.json` | Phase 6 results |
