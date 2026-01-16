# Hybrid Retrieval Theory - Deep Dive

## Table of Contents
1. [Why Hybrid Retrieval?](#why-hybrid-retrieval)
2. [BM25: Sparse Retrieval Explained](#bm25-sparse-retrieval)
3. [Reciprocal Rank Fusion (RRF)](#reciprocal-rank-fusion)
4. [Hybrid Retrieval Architecture](#hybrid-architecture)
5. [Theoretical Analysis](#theoretical-analysis)
6. [Practical Considerations](#practical-considerations)

---

## Why Hybrid Retrieval?

### The Complementarity Principle

**Dense retrieval** (embeddings) and **sparse retrieval** (BM25) have **orthogonal strengths**:

| Query Type | Example | Dense | Sparse | Winner |
|------------|---------|-------|--------|--------|
| **Semantic** | "How do plants make food?" → "photosynthesis" | ✅ | ❌ | Dense |
| **Exact keyword** | "Python 3.11 release date" | ❌ | ✅ | Sparse |
| **Paraphrase** | "capital of France" → "Paris is the capital" | ✅ | ❌ | Dense |
| **Rare term** | "NVIDIA RTX 3090 specs" | ❌ | ✅ | Sparse |
| **Entity name** | "Frédéric Chopin in Paris" | ❌ | ✅ | Sparse |

**Observation**: No single method dominates all query types.

**Solution**: **Combine both** to leverage complementary strengths.

---

### Empirical Evidence

**From Step 3 results**:

```
Dense retrieval (BGE):
- Recall@5: 95%
- Failure cases: Multi-hop reasoning, exact entity matching

Expected with Hybrid:
- Recall@5: 97-98% (+2-3%)
- Resolves: Exact keyword failures
```

**Research findings** (from MS MARCO, TREC benchmarks):
- Dense-only: 85-90% Recall@10
- BM25-only: 75-80% Recall@10
- Hybrid: 90-95% Recall@10

**Improvement**: +5-10% over best single method.

---

## BM25: Sparse Retrieval Explained

### The Core Idea

**BM25 (Best Matching 25)** ranks documents based on **term frequency** weighted by **term rarity**.

**Intuition**:
- Frequent terms in document → relevant
- Rare terms in corpus → important
- Long documents → penalize (normalize)

### Mathematical Formula

```
score(D, Q) = Σ_{i=1}^{n} IDF(q_i) × [f(q_i, D) × (k₁ + 1)] / [f(q_i, D) + k₁ × (1 - b + b × |D| / avgdl)]

Where:
- D = document
- Q = query = {q₁, q₂, ..., qₙ} (query terms)
- f(q_i, D) = frequency of term q_i in document D
- |D| = length of document D (in words)
- avgdl = average document length in corpus
- k₁ = term frequency saturation parameter (typically 1.2-2.0)
- b = length normalization parameter (typically 0.75)
- IDF(q_i) = inverse document frequency of term q_i
```

**IDF (Inverse Document Frequency)**:
```
IDF(q_i) = log[(N - n(q_i) + 0.5) / (n(q_i) + 0.5)]

Where:
- N = total number of documents
- n(q_i) = number of documents containing term q_i
```

---

### Step-by-Step Example

**Corpus**:
```
Doc 1: "The cat sat on the mat." (6 words)
Doc 2: "The dog played in the park." (6 words)
Doc 3: "Machine learning is fascinating." (4 words)

avgdl = (6 + 6 + 4) / 3 = 5.33
```

**Query**: "cat mat"

**Parameters**: k₁ = 1.5, b = 0.75

#### Step 1: Compute IDF

```
IDF("cat"):
- n("cat") = 1 (appears in Doc 1 only)
- IDF = log[(3 - 1 + 0.5) / (1 + 0.5)] = log(2.5 / 1.5) = 0.51

IDF("mat"):
- n("mat") = 1 (appears in Doc 1 only)
- IDF = log[(3 - 1 + 0.5) / (1 + 0.5)] = log(2.5 / 1.5) = 0.51
```

#### Step 2: Compute Term Frequencies

**For Doc 1**:
```
f("cat", Doc1) = 1
f("mat", Doc1) = 1
|Doc1| = 6
```

#### Step 3: Compute BM25 Score

**For term "cat" in Doc 1**:
```
score("cat") = IDF("cat") × [f × (k₁ + 1)] / [f + k₁ × (1 - b + b × |D| / avgdl)]
             = 0.51 × [1 × 2.5] / [1 + 1.5 × (1 - 0.75 + 0.75 × 6 / 5.33)]
             = 0.51 × 2.5 / [1 + 1.5 × (0.25 + 0.75 × 1.13)]
             = 0.51 × 2.5 / [1 + 1.5 × 1.09]
             = 0.51 × 2.5 / 2.64
             = 0.48
```

**Similarly for "mat"**: score("mat") ≈ 0.48

**Total score(Doc1, Query)** = 0.48 + 0.48 = **0.96**

**For Doc 2 and Doc 3**:
- Neither contains "cat" or "mat"
- score = **0.00**

**Ranking**: Doc 1 (0.96) > Doc 2 (0.00) > Doc 3 (0.00)

---

### Intuition Behind Components

#### 1. Term Frequency: f(q_i, D)

**Purpose**: Reward documents where query terms appear frequently.

**Saturation**: Not linear!
```
f=1 → contributes a lot
f=2 → contributes less (marginal gain)
f=10 → barely contributes more
```

**Why?** Seeing "Python" 10 times doesn't make document 10x more relevant.

**Formula behavior**:
```
k₁ = 1.5:
f=1 → weight = 1.2
f=2 → weight = 1.7
f=5 → weight = 2.1
f=10 → weight = 2.3
```

**Diminishing returns** → prevents keyword stuffing.

---

#### 2. IDF: Inverse Document Frequency

**Purpose**: Weight rare terms higher than common terms.

**Examples**:
```
Term         | n (docs containing) | IDF
-------------|---------------------|------
"the"        | 1000 / 1000         | 0.00 (worthless)
"machine"    | 100 / 1000          | 2.20 (informative)
"photosynthesis" | 5 / 1000        | 5.29 (very specific)
```

**Intuition**:
- Common words like "the", "is" → low IDF → ignored
- Rare, specific terms → high IDF → important

**Mathematical property**:
```
If n(q_i) → N (term in all docs):
  IDF → 0 (term is useless)

If n(q_i) → 0 (term very rare):
  IDF → log(N) (term is highly discriminative)
```

---

#### 3. Length Normalization: b × |D| / avgdl

**Purpose**: Prevent long documents from dominating simply because they contain more words.

**Problem without normalization**:
```
Doc A: "Python is great." (3 words)
  → f("Python") = 1

Doc B: "Python is great. Python is awesome. Python rocks." (8 words)
  → f("Python") = 3

Doc B scores higher just because it's longer and repeats terms.
```

**Solution**: Normalize by document length relative to average.

**Parameter b**:
- `b = 0`: No length normalization (long docs favored)
- `b = 1`: Full length normalization (short docs favored)
- `b = 0.75`: Balanced (standard)

**Effect**:
```
Long document (|D| > avgdl):
  → denominator increases → score decreases

Short document (|D| < avgdl):
  → denominator decreases → score increases
```

---

### BM25 Strengths

✅ **Exact keyword matching**: Finds documents containing query terms
✅ **Rare term boosting**: Specific terms weighted more
✅ **Fast**: No neural network, just counting
✅ **Interpretable**: Can explain why document ranked high
✅ **No training needed**: Works out-of-the-box

---

### BM25 Weaknesses

❌ **No semantic understanding**: "car" ≠ "automobile"
❌ **Vocabulary mismatch**: Query and document must share words
❌ **Synonym problem**: "buy" vs "purchase" treated as different
❌ **No word order**: "dog bites man" = "man bites dog"
❌ **No negation**: "not good" treated same as "good"

---

## Reciprocal Rank Fusion (RRF)

### The Fusion Problem

**Goal**: Combine rankings from multiple retrievers.

**Challenge**: Different retrievers have incompatible scores.

**Example**:
```
Query: "Python 3.11 features"

Dense retrieval:
  Doc A: score = 0.87
  Doc C: score = 0.72
  Doc B: score = 0.65

BM25:
  Doc B: score = 15.3
  Doc A: score = 8.7
  Doc D: score = 6.2
```

**Problem**: How to combine scores?
- Can't add (0.87 + 15.3 = meaningless)
- Can't normalize (scales are fundamentally different)

**Bad approaches**:
1. **Simple addition**: `final_score = dense + sparse` → **Wrong scales**
2. **Min-max normalization**: `(score - min) / (max - min)` → **Sensitive to outliers**
3. **Z-score**: `(score - mean) / std` → **Assumes normal distribution**

---

### RRF Solution

**Insight**: Forget scores, use **ranks** instead!

**Formula**:
```
RRF_score(d) = Σ_{r ∈ retrievers} 1 / (k + rank_r(d))

Where:
- d = document
- rank_r(d) = position of document d in retriever r's ranking (1-indexed)
- k = constant (typically 60)
```

**If document not in retriever's top-K**: rank = ∞ → contributes 0.

---

### RRF Example

**Query**: "Python 3.11 features"

**Dense retrieval** (top-5):
```
Rank 1: Doc A
Rank 2: Doc C
Rank 3: Doc B
Rank 4: Doc E
Rank 5: Doc F
```

**BM25** (top-5):
```
Rank 1: Doc B
Rank 2: Doc A
Rank 3: Doc D
Rank 4: Doc G
Rank 5: Doc H
```

**RRF scores** (k=60):

**Doc A**:
```
Dense: rank = 1 → 1/(60+1) = 0.0164
BM25:  rank = 2 → 1/(60+2) = 0.0161
Total: 0.0325
```

**Doc B**:
```
Dense: rank = 3 → 1/(60+3) = 0.0159
BM25:  rank = 1 → 1/(60+1) = 0.0164
Total: 0.0323
```

**Doc C**:
```
Dense: rank = 2 → 1/(60+2) = 0.0161
BM25:  not in top-5 → 0
Total: 0.0161
```

**Doc D**:
```
Dense: not in top-5 → 0
BM25:  rank = 3 → 1/(60+3) = 0.0159
Total: 0.0159
```

**Final ranking**:
```
1. Doc A (0.0325) ← Appears high in BOTH
2. Doc B (0.0323) ← Appears high in BOTH
3. Doc C (0.0161) ← Only in dense
4. Doc D (0.0159) ← Only in BM25
5. Doc E, F, G, H (< 0.0159)
```

**Key insight**: Documents that rank high in **multiple retrievers** get boosted.

---

### Why RRF Works

#### 1. **Score-agnostic**

No need to normalize or calibrate scores. Ranks are universal.

#### 2. **Emphasizes consensus**

Documents appearing in **both** top-K lists get highest scores.

#### 3. **Robust to outliers**

A single retriever's high score doesn't dominate.

#### 4. **Constant k controls smoothing**

**Effect of k**:
- `k = 0`: Only rank matters → 1st gets 1.00, 2nd gets 0.50, 3rd gets 0.33
- `k = 60`: Smoothed → 1st gets 0.016, 2nd gets 0.016, 3rd gets 0.016
- `k = ∞`: All ranks equal → random

**Standard k = 60**: Good balance.

#### 5. **Simple and fast**

No learning, no tuning (beyond k).

---

### RRF Mathematical Properties

#### Theorem 1: Monotonicity

If document d ranks higher in retriever r₁ than r₂:
```
rank_{r₁}(d) < rank_{r₂}(d)
⇒ 1/(k + rank_{r₁}(d)) > 1/(k + rank_{r₂}(d))
```

**Implication**: Better ranks always contribute more.

#### Theorem 2: Consensus Amplification

For two retrievers:
```
RRF(d) = 1/(k + r₁) + 1/(k + r₂)

If r₁ = r₂ = r (same rank in both):
  RRF = 2/(k + r)

If r₁ = 1, r₂ = ∞ (in one retriever only):
  RRF = 1/(k + 1)

Ratio: [2/(k + r)] / [1/(k + 1)]
     = 2(k + 1) / (k + r)
     > 1 if r < k + 2
```

**Implication**: Consensus documents get >2x boost.

---

## Hybrid Retrieval Architecture

### Pipeline Design

```
Query
  ↓
  ├───────────────────────────────────┐
  ↓                                   ↓
[Dense Retriever]              [Sparse Retriever]
(BGE + FAISS)                  (BM25)
  ↓                                   ↓
Top-K₁ docs                     Top-K₂ docs
(e.g., K₁=20)                   (e.g., K₂=20)
  ↓                                   ↓
  └───────────────────────────────────┤
                  ↓
          [RRF Fusion]
                  ↓
          Top-K final docs
          (e.g., K=5)
```

### Parameter Choices

**K₁, K₂ (retriever top-K)**:
- **Trade-off**: Recall vs Speed
- **Recommendation**: K₁ = K₂ = 20-50
- **Reasoning**: RRF needs sufficient overlap

**K_final (output top-K)**:
- **Standard**: K = 5-10
- For re-ranking (Step 5): K = 20

**k_rrf (RRF constant)**:
- **Standard**: k = 60
- **Robust** across datasets
- Rarely needs tuning

---

## Theoretical Analysis

### Recall Upper Bound

**Theorem**: Hybrid Recall@K ≥ max(Dense Recall@K, Sparse Recall@K)

**Proof sketch**:
- If ground truth in dense top-K → RRF will include it (gets positive score)
- If ground truth in sparse top-K → RRF will include it
- If ground truth in BOTH → even better (higher RRF score)

**Implication**: Hybrid **cannot be worse** than best single retriever.

---

### Expected Improvement

**Model assumptions**:
- Dense and sparse failures are **independent**
- P(dense finds doc) = p₁
- P(sparse finds doc) = p₂

**Hybrid recall**:
```
P(hybrid finds doc) = P(dense OR sparse)
                    = p₁ + p₂ - p₁ · p₂

If p₁ = p₂ = 0.90 (each 90% recall):
  P(hybrid) = 0.90 + 0.90 - 0.81 = 0.99 (99% recall!)
```

**In practice**: Failures are **partially correlated** (some queries hard for both).

**Empirical observation**:
```
Dense: 95%
Sparse: 70%
Hybrid: 97-98% (not 99.5%, due to correlation)
```

---

### Diversity-Accuracy Trade-off

**RRF amplifies consensus** but may reduce diversity.

**Example**:
- Dense and BM25 both rank Doc A highly
- Doc A gets boosted by RRF
- Potentially relevant Doc B (high in dense, low in BM25) gets pushed down

**Mitigation**: Use higher K₁, K₂ (e.g., 50 instead of 20) to preserve diversity before fusion.

---

## Practical Considerations

### When to Use Hybrid vs Single Retriever

| Scenario | Recommended | Reasoning |
|----------|-------------|-----------|
| **General QA** | Hybrid | Covers all query types |
| **Code search** | Hybrid (favor BM25) | Exact identifiers matter |
| **Semantic search** | Dense-only | Paraphrasing common |
| **Fact retrieval** | Hybrid | Mix of keywords and semantics |
| **Low latency (<5ms)** | Dense-only | BM25 + fusion adds overhead |

---

### Latency Considerations

**Breakdown**:
```
Dense retrieval:  2-3ms
BM25 retrieval:   0.5-1ms
RRF fusion:       0.1ms
Total:            2.6-4.1ms
```

**Optimization**: **Parallelize** dense and BM25 retrieval:
```python
import concurrent.futures

with concurrent.futures.ThreadPoolExecutor() as executor:
    future_dense = executor.submit(dense_retriever.search, query, k=20)
    future_sparse = executor.submit(sparse_retriever.search, query, k=20)

    dense_results = future_dense.result()
    sparse_results = future_sparse.result()

# Total time: max(2-3ms, 0.5-1ms) = 2-3ms (no overhead!)
```

---

### Storage Requirements

**Dense index** (FAISS):
- 744 vectors × 1024 dim × 4 bytes = 3.0 MB

**BM25 index**:
- Inverted index: term → list of (doc_id, frequency)
- Typical size: ~0.5-1 MB (text-only, no vectors)

**Total**: ~4 MB (minimal)

---

### Tuning BM25 Parameters

**Standard values**:
- `k₁ = 1.2` (TREC recommendation)
- `b = 0.75`

**When to adjust**:

**k₁** (term frequency saturation):
- **Increase k₁ (1.5-2.0)** if documents have **high term repetition** (e.g., legal docs)
- **Decrease k₁ (0.8-1.0)** if documents are **short** (e.g., tweets)

**b** (length normalization):
- **Increase b (0.85-0.95)** if **long documents dominate** unfairly
- **Decrease b (0.5-0.65)** if **short documents should be favored** (e.g., titles)

**For SQuAD**: Standard k₁=1.2, b=0.75 works well (documents ~700 chars, balanced).

---

## Key Takeaways

1. **BM25 provides exact keyword matching** that dense retrieval lacks

2. **RRF elegantly fuses rankings** without score normalization

3. **Hybrid retrieval combines strengths** of dense and sparse methods

4. **Expected improvement**: 2-3% absolute Recall@5 (95% → 97-98%)

5. **Latency overhead is minimal** with parallelization

6. **No additional training** required (works out-of-the-box)

7. **Complementarity is key**: Dense and sparse fail on different queries

---

## Next Steps

1. Implement BM25 retriever
2. Implement RRF fusion
3. Compare Dense vs Sparse vs Hybrid on SQuAD queries
4. Analyze failure cases to validate improvements
5. Visualize where each method excels

Continue to **[04_reranking_theory.md](04_reranking_theory.md)** →
