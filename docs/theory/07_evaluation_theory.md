# Step 7: Evaluation Framework — Theory

## Table of Contents
1. [The Evaluation Challenge in RAG](#1-the-evaluation-challenge-in-rag)
2. [Lexical Metrics: Exact Match, F1, ROUGE-L](#2-lexical-metrics)
3. [Semantic Metrics: BERTScore](#3-semantic-metrics-bertscore)
4. [Faithfulness: Grounding Answers in Context](#4-faithfulness)
5. [The RAG Triad & LLM-as-a-Judge (RAGAS)](#5-the-rag-triad--llm-as-a-judge-ragas)
6. [Error Analysis Taxonomy](#6-error-analysis-taxonomy)
7. [Statistical A/B Testing for Pipeline Comparison](#7-statistical-ab-testing)
8. [Human Evaluation & Inter-Annotator Agreement](#8-human-evaluation--inter-annotator-agreement)
9. [Cost Tracking & Production Considerations](#9-cost-tracking--production-considerations)
10. [Key Takeaways](#10-key-takeaways)

---

## 1. The Evaluation Challenge in RAG

### Why Is RAG Evaluation Hard?

Traditional NLP evaluation compares model output to a gold reference. In RAG systems, this becomes a **multi-dimensional problem** because the output quality depends on three interdependent stages:

```
Query → [Retriever] → Contexts → [Generator] → Answer
          │                         │
    Did we find the          Did we use it
    right information?       correctly?
```

A wrong answer can come from:
1. **Retrieval failure** — the right document was never retrieved
2. **Context noise** — the right document was retrieved but drowned in irrelevant chunks
3. **Generation error** — the right context was present but the LLM misinterpreted it
4. **Hallucination** — the LLM ignored the context and generated from parametric memory

**No single metric captures all of these.** This is why we need a multi-layered evaluation stack.

### The Evaluation Stack

Our framework implements four layers, each catching different failure modes:

| Layer | What It Measures | Metrics | Cost |
|-------|-----------------|---------|------|
| **Lexical** | Token overlap with ground truth | EM, F1, ROUGE-L | Free, instant |
| **Semantic** | Meaning similarity via embeddings | BERTScore | Free, ~50ms/query |
| **Grounding** | Faithfulness to retrieved context | Local faithfulness, RAGAS | Free or API |
| **LLM Judge** | Holistic quality via reasoning | RAGAS (faithfulness, recall, precision) | ~$0.002/query |
| **Human** | Gold standard subjective quality | Rubric ratings, Cohen's Kappa | ~$0.50-2/query |

Each layer is progressively more expensive but also more reliable.

---

## 2. Lexical Metrics

### 2.1 Exact Match (EM)

The simplest possible metric. Returns 1 if the prediction matches the ground truth exactly (after normalization), 0 otherwise.

```
EM(prediction, ground_truth) = 1 if normalize(prediction) == normalize(ground_truth) else 0
```

**Normalization** typically includes: lowercasing, stripping whitespace, removing articles ("a", "an", "the"), and removing punctuation.

**Strengths**: Unambiguous, no false positives.

**Weaknesses**: Catastrophically strict for generative models. If the ground truth is "Houston, Texas" and the model outputs "Beyonce grew up in Houston, Texas", EM = 0 despite being correct. This makes EM nearly useless for generative QA (we observed EM = 0.03 on 100 queries).

### 2.2 Token-Level F1 Score

F1 treats prediction and ground truth as **bags of tokens** and computes the harmonic mean of precision and recall:

```
Precision = |tokens(prediction) ∩ tokens(ground_truth)| / |tokens(prediction)|
Recall    = |tokens(prediction) ∩ tokens(ground_truth)| / |tokens(ground_truth)|

F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

#### Step-by-Step Example

```
Ground truth: "Houston, Texas"
Prediction:   "Beyonce grew up in Houston, Texas."

tokens(GT)   = {"houston", "texas"}
tokens(Pred) = {"beyonce", "grew", "up", "in", "houston", "texas"}

Overlap = {"houston", "texas"} → |overlap| = 2

Precision = 2 / 6 = 0.333
Recall    = 2 / 2 = 1.000

F1 = 2 × (0.333 × 1.000) / (0.333 + 1.000) = 0.500
```

**Interpretation**: F1 = 0.50 means the answer contains the right information but is diluted with extra tokens. For extractive QA (SQuAD), F1 > 0.8 is typical. For generative QA with a 3B parameter model, F1 ~ 0.30-0.40 is expected because the model adds context around the answer.

**Strengths**: Handles partial matches; rewards both finding the answer (recall) and being concise (precision).

**Weaknesses**: Still purely lexical — "6" and "six" have zero overlap. "automobile" and "car" have zero overlap. Does not capture semantic equivalence.

### 2.3 ROUGE-L (Longest Common Subsequence)

ROUGE-L uses the **Longest Common Subsequence (LCS)** instead of bag-of-words overlap. The LCS captures word order, which F1 ignores.

```
LCS(X, Y) = length of the longest common subsequence

ROUGE-L Precision = LCS(prediction, ground_truth) / |prediction|
ROUGE-L Recall    = LCS(prediction, ground_truth) / |ground_truth|

ROUGE-L F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

#### What Is a Subsequence?

A subsequence maintains **relative order** but does not require contiguity:

```
Sequence:     "the cat sat on the mat"
Subsequence:  "cat on mat"        ✓ (order preserved)
Subsequence:  "mat cat on"        ✗ (order violated)
Substring:    "cat sat on"        ✓ (contiguous)
Subsequence:  "cat on mat"        ✓ (not contiguous, but order preserved)
```

#### Step-by-Step Example

```
Ground truth: "late 1990s"
Prediction:   "Beyonce became famous in the late 1990s"

LCS = "late 1990s" → length = 2 (tokens: "late", "1990s")

Precision = 2 / 7 = 0.286
Recall    = 2 / 2 = 1.000

ROUGE-L = 2 × (0.286 × 1.000) / (0.286 + 1.000) = 0.444
```

**When ROUGE-L > F1**: Usually not by much for short answers. They tend to track closely.

**When ROUGE-L < F1**: When the matching tokens appear in a different order.

---

## 3. Semantic Metrics: BERTScore

### The Problem with Lexical Metrics

```
Ground truth: "automobile"
Prediction:   "car"

F1 = 0.0  (no token overlap)
EM = 0.0

But semantically, this is a perfect answer!
```

### BERTScore: Using Contextual Embeddings

BERTScore replaces exact token matching with **cosine similarity between contextualized token embeddings**. Each token in the prediction is soft-matched to its most similar token in the reference, and vice versa.

#### Algorithm

```
1. Encode both strings with a pretrained model (e.g., distilbert-base-uncased):
   prediction_embeddings = BERT("car")        → [e₁]
   reference_embeddings  = BERT("automobile")  → [e₂]

2. Compute pairwise cosine similarities:
   sim(eᵢ, eⱼ) = (eᵢ · eⱼ) / (||eᵢ|| × ||eⱼ||)

3. Greedy matching:
   BERTScore Precision = (1/|pred|) × Σᵢ maxⱼ sim(pred_eᵢ, ref_eⱼ)
   BERTScore Recall    = (1/|ref|)  × Σⱼ maxᵢ sim(pred_eᵢ, ref_eⱼ)

4. F1:
   BERTScore F1 = 2 × (P × R) / (P + R)
```

**Key insight**: The cosine similarity between "car" and "automobile" in BERT's embedding space is ~0.85, so BERTScore ≈ 0.85 instead of 0.0.

#### Why distilbert-base-uncased?

We chose this model for BERTScore computation because:
- **Speed**: 2x faster than bert-base, ~6x faster than roberta-large
- **Size**: 66M parameters (vs 110M for bert-base, 355M for roberta-large)
- **Quality**: Within 1-2 points of roberta-large on BERTScore correlation benchmarks
- **CPU-friendly**: Runs at ~50ms/query on CPU (no GPU required)

#### Interpretation Scale

```
BERTScore ≥ 0.90  →  Very high semantic similarity (near-paraphrase)
BERTScore ≥ 0.80  →  Good semantic match
BERTScore ≥ 0.70  →  Moderate match (related but may differ in specifics)
BERTScore < 0.60  →  Low similarity (likely wrong answer)
```

**Our results**: Mean BERTScore = 0.744, which indicates the model generally captures the right meaning even when F1 is low due to verbosity.

---

## 4. Faithfulness

### Definition

Faithfulness measures whether the answer is **grounded in the retrieved context**, as opposed to being generated from the LLM's parametric knowledge (memorized training data).

This is distinct from correctness:
- A **faithful but wrong** answer: The context is wrong, and the model accurately reproduces the wrong information.
- An **unfaithful but correct** answer: The model ignores the context and generates a correct answer from memory — this is **hallucination** in a RAG sense.

### Local Faithfulness (Token Overlap)

Our local implementation uses a simple heuristic: what fraction of tokens in the prediction appear in the retrieved contexts?

```
faithfulness_local = |tokens(prediction) ∩ tokens(contexts)| / |tokens(prediction)|
```

**Strengths**: Free, instant, no API needed.

**Weaknesses**: Dramatically overestimates faithfulness. If the model says "Forbes magazine" and the context mentions "Forbes" elsewhere in an unrelated sentence, local faithfulness = 1.0 even though the model may have fabricated the connection.

### RAGAS Faithfulness (LLM-as-Judge)

RAGAS uses an LLM to decompose the answer into individual **claims** and check each one against the context:

```
1. LLM extracts claims from the answer:
   Answer: "Beyonce won 6 Grammy awards in 2010"
   → Claim 1: "Beyonce won Grammy awards"
   → Claim 2: "The number was 6"
   → Claim 3: "This happened in 2010"

2. For each claim, LLM checks: "Is this claim supported by the given context?"
   → Claim 1: Yes (context mentions Grammy wins)
   → Claim 2: Yes (context says "six Grammy Awards")
   → Claim 3: Yes (context mentions 2010)

3. Faithfulness = supported_claims / total_claims = 3/3 = 1.0
```

**Our observation**: Local faithfulness = 1.0 vs RAGAS faithfulness = 0.85 on the same 10-query smoke test. The 15% gap represents cases where token overlap gives false confidence — the tokens appear in the context but the logical connection is fabricated.

---

## 5. The RAG Triad & LLM-as-a-Judge (RAGAS)

### The RAG Evaluation Triad

RAGAS (Retrieval-Augmented Generation Assessment) evaluates the three-way relationship between **Query**, **Context**, and **Answer**:

```
        Query
       /     \
      /       \
  Context ──── Answer

  Each edge = a metric:
  Context ↔ Answer  = Faithfulness     (is the answer grounded?)
  Query   ↔ Context = Context Recall   (did we retrieve the right info?)
  Query   ↔ Answer  = Answer Relevance (does the answer address the query?)
```

### Metrics in Detail

| Metric | Inputs | Formula (Conceptual) | What It Catches |
|--------|--------|---------------------|-----------------|
| **Faithfulness** | Answer + Context | supported_claims / total_claims | Hallucination |
| **Context Recall** | Context + Ground Truth | GT_sentences_attributable_to_context / total_GT_sentences | Retrieval gaps |
| **Context Precision** | Context + Ground Truth | relevant_chunks_rank_weighted / total_chunks | Context noise |
| **Answer Correctness** | Answer + Ground Truth | Weighted F1 + semantic similarity | Overall accuracy |

### Context Precision — Rank-Weighted

Context Precision is not a simple ratio. It uses a **rank-weighted** formula that rewards relevant chunks appearing earlier:

```
Context Precision@K = (1/K) × Σ_{k=1}^{K} (Precision@k × relevance(k))

Where:
  relevance(k) = 1 if chunk at rank k is relevant, 0 otherwise
  Precision@k  = relevant_chunks_in_top_k / k
```

This penalizes configurations where relevant chunks appear at rank 4-5 rather than rank 1-2, because the LLM pays more attention to earlier context.

### Our LLM Judge Setup

```
Judge LLM:    Mistral Large (via API, free tier)
Embeddings:   bge-base-en-v1.5 (HuggingFace, local, free)
Framework:    RAGAS + LangchainLLMWrapper
Cost:         ~$0.002 per query (Mistral API)
```

**Why not OpenAI?** No OpenAI API key available. Mistral Large is a strong alternative — it follows complex rubric instructions well and is available on a free tier (with rate limits and occasional retries).

**Why local embeddings for RAGAS?** Using `HuggingFaceEmbeddings(bge-base-en-v1.5)` locally avoids API calls for the embedding step. Only the LLM reasoning calls go to Mistral's API.

---

## 6. Error Analysis Taxonomy

### Why Categorize Errors?

A single number (e.g., "F1 = 0.32") doesn't tell you **what to fix**. Error taxonomy answers: "Is the problem in retrieval or generation? Is the model hallucinating or is the answer just verbose?"

### Category Definitions

Our taxonomy uses 8 mutually non-exclusive categories (except "correct" which is exclusive):

| Category | Detection Rule | Root Cause |
|----------|---------------|------------|
| **correct** | EM = 1.0 or F1 ≥ 0.8 | No error |
| **empty_response** | prediction is empty/blank | Model refused to answer |
| **retrieval_failure** | GT not found in any retrieved context | Index gap or query mismatch |
| **hallucination** | faithfulness < 0.5 and F1 > 0 | LLM ignored context |
| **low_context_relevance** | faithfulness < 0.5, not retrieval failure, not hallucination | Context present but misused |
| **incomplete_answer** | < 3 tokens and F1 < 0.3 | Truncated or too brief |
| **verbose_answer** | ROUGE-L > 0.4 and EM = 0 and F1 < 0.5 | Right content, too wordy |
| **wrong_answer** | F1 < 0.3 and EM = 0, catch-all | Factually incorrect |

### Composite Badness Score

To rank predictions from worst to best, we use a weighted composite score:

```
badness = 0.5 × F1 + 0.3 × faithfulness + 0.2 × EM
```

**Lower score = worse prediction.** This prioritizes F1 (answer quality), then faithfulness (grounding), then exact match (precision). The 20 worst-scoring predictions are saved for manual inspection.

### Decision Flow

```
Is EM = 1.0 or F1 ≥ 0.8?
  └─ Yes → "correct" (stop)
  └─ No ↓
Is prediction empty?
  └─ Yes → "empty_response" (stop)
  └─ No ↓
Is ground truth absent from all contexts?
  └─ Yes → add "retrieval_failure"
Is faithfulness < 0.5 AND F1 > 0?
  └─ Yes → add "hallucination"
Is faithfulness < 0.5 AND not retrieval_failure AND not hallucination?
  └─ Yes → add "low_context_relevance"
Is prediction < 3 tokens AND F1 < 0.3?
  └─ Yes → add "incomplete_answer"
Is ROUGE-L > 0.4 AND EM = 0 AND F1 < 0.5?
  └─ Yes → add "verbose_answer"
Is F1 < 0.3 AND EM = 0 AND not hallucination AND not incomplete?
  └─ Yes → add "wrong_answer"
No categories assigned?
  └─ Fallback → "wrong_answer"
```

---

## 7. Statistical A/B Testing

### The Problem: "Is This Improvement Real?"

When comparing two RAG pipelines (e.g., Linear vs. Agentic), looking at mean F1 is insufficient:

```
Pipeline A:  F1 = 0.322
Pipeline B:  F1 = 0.354

Is this +0.032 improvement real, or just noise from the random sample of queries?
```

LLM outputs have high variance — the same pipeline can produce different F1 values on different query subsets. We need **statistical tests** to distinguish signal from noise.

### 7.1 Paired t-Test

We use a **paired** t-test because both pipelines are evaluated on the **same queries**. This eliminates query difficulty as a confounding variable.

```
For each query i:
  d_i = score_challenger_i - score_champion_i

Null hypothesis H₀: mean(d) = 0 (no real difference)
Alternative H₁:    mean(d) ≠ 0 (real difference exists)

Test statistic:
  t = d̄ / (s_d / √n)

Where:
  d̄   = mean of paired differences
  s_d = standard deviation of paired differences
  n   = number of query pairs

p-value = P(|T| ≥ |t|) under H₀, with df = n-1

Decision: reject H₀ if p < α (typically α = 0.05)
```

#### Why Paired, Not Independent?

```
Independent test:  Var(X̄_A - X̄_B) = Var(X̄_A) + Var(X̄_B)
Paired test:       Var(d̄) = Var(X̄_A) + Var(X̄_B) - 2·Cov(X̄_A, X̄_B)

Since pipeline scores on the same query are positively correlated:
  Cov(X̄_A, X̄_B) > 0

Therefore:
  Var(d̄) < Var(X̄_A - X̄_B)

→ Paired test has more statistical power (detects smaller differences).
```

### 7.2 Cohen's d — Effect Size

Statistical significance (p-value) tells you **if** there's a difference. Effect size tells you **how large** that difference is.

```
Cohen's d = d̄ / s_d

Where:
  d̄  = mean paired difference
  s_d = standard deviation of paired differences (with Bessel's correction, ddof=1)
```

**Interpretation scale** (Cohen, 1988):

```
|d| < 0.2  →  Negligible effect (not worth the complexity)
|d| ≈ 0.2  →  Small effect
|d| ≈ 0.5  →  Medium effect
|d| ≈ 0.8  →  Large effect
|d| > 1.0  →  Very large effect
```

**Practical implication**: Even if p < 0.001 (highly significant), if d = 0.1 the improvement is tiny and may not justify the extra cost/latency of the new pipeline.

### 7.3 Bootstrap Confidence Intervals

The bootstrap provides a **non-parametric** confidence interval for the mean difference, without assuming normality.

```
Algorithm:
  1. Compute paired differences: d_i = challenger_i - champion_i
  2. Repeat B = 1000 times:
     a. Sample n differences WITH replacement → d*
     b. Compute mean(d*)
  3. Sort the 1000 bootstrap means
  4. CI = [percentile(2.5%), percentile(97.5%)]
```

**Interpretation**:
- If CI = [+0.02, +0.08]: We're 95% confident the true improvement is between 2% and 8%. **Significant** (CI doesn't cross 0).
- If CI = [-0.01, +0.05]: The true difference might be negative. **Not significant** (CI crosses 0).

### 7.4 Power Analysis — Minimum Sample Size

Before running an expensive evaluation, estimate how many queries you need:

```
N = ((z_{α/2} + z_β) / d)²

Where:
  z_{α/2} = 1.96 (for α = 0.05, two-tailed)
  z_β     = 0.84 (for power = 0.80)
  d       = expected Cohen's d (effect size)
```

**Example calculations**:

| Expected Effect | d | Required N |
|----------------|---|-----------|
| Small (0.2) | 0.2 | 197 queries |
| Medium (0.5) | 0.5 | 32 queries |
| Large (0.8) | 0.8 | 13 queries |

**Our setup**: 100 queries can detect medium effects (d ≥ 0.28) with 80% power at α = 0.05.

### 7.5 Winner Determination Logic

Our ABTestRunner uses a **primary metric + guard metrics** pattern:

```
1. Check guard metrics (e.g., faithfulness):
   If ANY guard metric significantly REGRESSES → champion wins (safety first)

2. Check primary metric (e.g., F1):
   If significantly BETTER for challenger → challenger wins
   If significantly WORSE for challenger → champion wins
   If not significant → "no significant difference"
     → Suggest running more queries (report minimum_sample_size)
```

This prevents the common mistake of optimizing F1 at the cost of faithfulness (trading accuracy for hallucination).

---

## 8. Human Evaluation & Inter-Annotator Agreement

### Why Human Evaluation?

Automated metrics are proxies. They correlate with quality but don't define it. The only ground truth for "Is this a good answer?" is a human judgment.

**However**, humans are inconsistent. Two annotators may disagree on whether an answer is "4/5 relevant" or "3/5 relevant". We need to **quantify this disagreement** to know if our human labels are trustworthy.

### 8.1 The Quality Rubric

We define three evaluation dimensions, each on a 1-5 Likert scale:

**Relevance** — Does the answer address the query?
```
1: Completely irrelevant — answer does not address the query at all
2: Mostly irrelevant — tangentially related but misses the point
3: Partially relevant — addresses query but missing key information
4: Mostly relevant — addresses query with minor gaps
5: Fully relevant — directly and completely addresses the query
```

**Faithfulness** — Is the answer supported by the retrieved contexts?
```
1: Hallucinated — no support in contexts
2: Mostly unsupported — minor overlap with contexts
3: Partially supported — some claims supported, some not
4: Mostly supported — main claims grounded in contexts
5: Fully supported — every claim traceable to contexts
```

**Conciseness** — Is the answer appropriately concise?
```
1: Extremely verbose — mostly filler or repetition
2: Verbose — significant unnecessary content
3: Acceptable — some unnecessary content but mostly focused
4: Concise — minimal unnecessary content
5: Perfectly concise — every word contributes to the answer
```

### 8.2 Sampling Strategies

We cannot annotate all 100+ predictions. We sample a representative subset using three strategies:

**Stratified sampling** (default): Uses ErrorAnalyzer to identify error categories, then samples proportionally from each category. Ensures all failure modes are represented.

**Worst-case sampling**: Selects the N predictions with the lowest composite badness score. Focuses annotation effort on the most problematic cases.

**Random sampling**: Uniform random selection. Gives unbiased but potentially unrepresentative sample.

### 8.3 Cohen's Kappa — Inter-Annotator Agreement

Cohen's Kappa measures agreement between two annotators **while correcting for chance agreement**.

```
κ = (p_o - p_e) / (1 - p_e)

Where:
  p_o = observed agreement (proportion of ratings that match)
  p_e = expected agreement by chance (if annotators rated randomly)
```

#### Why Not Just Percentage Agreement?

```
Scenario: Two annotators rating 100 samples, 90% of answers are "5/5"

If both annotators just always rate "5":
  Observed agreement = 100%  → Looks perfect!
  But they're not really evaluating anything.

Kappa corrects for this:
  p_o = 1.00
  p_e = 0.90 × 0.90 + 0.10 × 0.10 = 0.82
  κ = (1.00 - 0.82) / (1 - 0.82) = 1.00  → Still perfect (they really do agree)

But if one annotator rates 50% as "5" and the other rates 90% as "5":
  p_o might still be high, but p_e would be different
  κ would be lower → correctly reflecting the disagreement pattern
```

#### Quadratic Weighting

For ordinal scales (1-5), we use **quadratic weights** because a disagreement of 1→5 is much worse than 1→2:

```
Weight matrix w(i,j) = 1 - ((i - j)² / (R - 1)²)

For a 5-point scale (R = 5):
  w(1,1) = 1.000  w(1,2) = 0.938  w(1,3) = 0.750  w(1,4) = 0.438  w(1,5) = 0.000
  w(2,2) = 1.000  w(2,3) = 0.938  w(2,4) = 0.750  w(2,5) = 0.438
  w(3,3) = 1.000  w(3,4) = 0.938  w(3,5) = 0.750
  w(4,4) = 1.000  w(4,5) = 0.938
  w(5,5) = 1.000
```

**Effect**: Rating [1,2] (disagree by 1) gets weight 0.938 (nearly full agreement). Rating [1,5] (disagree by 4) gets weight 0.000 (complete disagreement). This is appropriate for Likert scales where adjacent categories are similar.

#### Interpretation Scale (Landis & Koch, 1977)

```
κ < 0.00  →  Poor (worse than chance)
κ 0.00–0.20  →  Slight agreement
κ 0.21–0.40  →  Fair agreement
κ 0.41–0.60  →  Moderate agreement
κ 0.61–0.80  →  Substantial agreement
κ 0.81–1.00  →  Almost perfect agreement
```

**Target**: κ ≥ 0.60 for each dimension. Below this, the rubric or annotator training needs revision.

### 8.4 Correlation with Automated Metrics

After collecting human ratings, we compute **Spearman rank correlation** between human scores and automated metrics:

```
ρ = 1 - (6 × Σ d_i²) / (n × (n² - 1))

Where:
  d_i = rank(human_rating_i) - rank(automated_metric_i)
  n   = number of samples
```

**Expected correlations**:
- Human relevance ↔ F1: ρ ≈ 0.50-0.70
- Human faithfulness ↔ local faithfulness: ρ ≈ 0.30-0.50 (local metric is weak)
- Human faithfulness ↔ RAGAS faithfulness: ρ ≈ 0.60-0.80 (LLM judge is better)
- Human conciseness ↔ ROUGE-L: ρ ≈ 0.40-0.60

These correlations validate (or invalidate) whether our automated metrics are trustworthy proxies for human judgment.

---

## 9. Cost Tracking & Production Considerations

### Why Track Cost?

Our pipeline runs locally with Qwen2.5-3B (free), but in production you might use a cloud LLM. Cost tracking estimates what the pipeline would cost with different models:

```
cost = (input_tokens × input_price_per_token) + (output_tokens × output_price_per_token)
```

### Hypothetical Pricing Table

| Model | Input ($/1M tokens) | Output ($/1M tokens) |
|-------|-------------------|---------------------|
| Qwen2.5-3B (local) | $0.00 | $0.00 |
| Mistral Large | $2.00 | $6.00 |
| GPT-4o | $2.50 | $10.00 |
| GPT-4o-mini | $0.15 | $0.60 |
| Claude 3.5 Sonnet | $3.00 | $15.00 |

### Token Counting

We use the generator's tokenizer to count exact tokens per query:

```
input_tokens  = len(tokenizer.encode(prompt))
output_tokens = len(tokenizer.encode(answer))
```

This gives accurate cost projections even though we run inference locally.

### Time-to-First-Token (TTFT)

TTFT measures the latency until the first token is generated, which directly impacts perceived responsiveness in streaming applications.

Our `TTFTMeasurer` uses a `LogitsProcessor` injected into `model.generate()`:

```python
class _TTFTLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids, scores):
        if self.ttft is None:
            self.ttft = time.perf_counter() - self.start_time
        return scores
```

This captures the exact moment the model produces its first logits, without modifying the generator code.

---

## 10. Key Takeaways

1. **Don't trust F1/EM for generative QA**: They systematically underestimate quality when the model generates full sentences. BERTScore (0.744) is a better proxy than F1 (0.322) for our system.

2. **Local faithfulness is deceptive**: Token overlap gives 1.0 when RAGAS gives 0.85. The 15% gap represents real hallucinations that local metrics miss.

3. **Decouple retrieval from generation errors**: Error taxonomy lets you fix the right component. Retrieval failure → improve index/chunking. Hallucination → improve prompt/model. Verbose answer → improve post-processing.

4. **Always run significance tests**: A +3% F1 improvement means nothing without a p-value. Our A/B testing infrastructure provides paired t-test, bootstrap CI, and Cohen's d.

5. **Human evaluation is the gold standard but expensive**: Cohen's Kappa with quadratic weights ensures inter-annotator agreement is measured fairly for ordinal scales.

6. **Cost awareness matters**: Even though we run locally ($0.00/query), knowing that the same workload would cost $0.38 on Claude 3.5 Sonnet helps inform production decisions.

7. **The evaluation stack is incremental**: Start with free lexical metrics, add BERTScore for semantic matching, use RAGAS for hallucination detection, and reserve human evaluation for the most critical samples.

---

## References

- Rajpurkar et al. (2016). "SQuAD: 100,000+ Questions for Machine Comprehension of Text"
- Zhang et al. (2020). "BERTScore: Evaluating Text Generation with BERT"
- Es et al. (2024). "RAGAS: Automated Evaluation of Retrieval Augmented Generation"
- Cohen, J. (1988). "Statistical Power Analysis for the Behavioral Sciences"
- Landis, J.R. & Koch, G.G. (1977). "The Measurement of Observer Agreement for Categorical Data"
- Lin, C.-Y. (2004). "ROUGE: A Package for Automatic Evaluation of Summaries"

---

Continue to **[step7_evaluation_analysis.md](../experiments/step7_evaluation_analysis.md)** for experimental results →
