# Re-Ranking Theory - Deep Dive

## Table of Contents
1. [Why Re-Ranking?](#why-re-ranking)
2. [Bi-Encoder vs Cross-Encoder](#bi-encoder-vs-cross-encoder)
3. [Cross-Encoder Architecture](#cross-encoder-architecture)
4. [BGE-Reranker Model](#bge-reranker-model)
5. [Integration Strategies](#integration-strategies)
6. [Trade-offs and Optimization](#trade-offs)

---

## Why Re-Ranking?

### The Two-Stage Paradigm

**Problem**: Computing exact relevance for all documents is too expensive.

**Solution**: Two-stage retrieval
1. **Stage 1 (Retrieval)**: Fast, approximate ranking on full corpus
2. **Stage 2 (Re-ranking)**: Precise, expensive ranking on top-K candidates

### The Accuracy-Speed Trade-off

| Method | Speed | Accuracy | Use Case |
|--------|-------|----------|----------|
| **BM25** | Very Fast | Low | Initial filtering |
| **Bi-encoder (BGE)** | Fast | Medium-High | First-stage retrieval |
| **Cross-encoder** | Slow | Very High | Re-ranking top-K |

**Key Insight**: Use fast methods to reduce candidates (1M → 20), then precise method on small set.

---

## Bi-Encoder vs Cross-Encoder

### Bi-Encoder (Dual Encoder)

**Architecture**: Encode query and document **independently**

```
Query: "What is the capital of France?"
  ↓
[Encoder]
  ↓
q_embedding = [0.23, -0.15, ..., 0.89]  (1024-dim)

Document: "Paris is the capital of France"
  ↓
[Encoder]  (same or different encoder)
  ↓
d_embedding = [0.25, -0.14, ..., 0.87]  (1024-dim)

Similarity = cosine(q_embedding, d_embedding) = 0.92
```

**Key Property**: Query and document **never interact** during encoding.


---

### Cross-Encoder (Interaction Model)

**Architecture**: Encode query and document **jointly**

```
Input: "[CLS] What is the capital of France? [SEP] Paris is the capital of France [SEP]"
  ↓
[Transformer Encoder]
  ↓
[Self-Attention across ALL tokens]
  ↓
[CLS] token representation
  ↓
[Classification Head]
  ↓
Relevance Score = 0.987  (single scalar)
```

**Key Property**: Query and document tokens **attend to each other**.

---

### Comparison: Bi-Encoder vs. Cross-Encoder

| Encoder | Key Property | Architecture Type | Advantages | Disadvantages |
| :--- | :--- | :--- | :--- | :--- |
| **Bi-Encoder** | Query and document **never interact** during encoding. | **Cosine Similarity** on fixed embeddings. | ✅ Pre-compute document embeddings<br>✅ Fast search (FAISS-like)<br>✅ Scalable to millions of documents | ❌ Limited expressiveness<br>❌ Cannot model complex relevance patterns<br>❌ Lower precision |
| **Cross-Encoder** | Query and document tokens **attend to each other**. | **Full Self-Attention** across all tokens. | ✅ Full interaction between query and document<br>✅ Higher precision (5-10% better)<br>✅ Captures fine-grained relevance | ❌ Cannot pre-compute<br>❌ Slow (~100x slower)<br>❌ Not scalable to full corpus |

---

### Why Cross-Encoder is More Powerful

#### Example: Negation Handling

**Query**: "movies NOT about war"

**Bi-Encoder**:
```python
q_emb = encode("movies NOT about war")
d1_emb = encode("Saving Private Ryan: A war movie")
d2_emb = encode("The Notebook: A romance movie")

similarity(q_emb, d1_emb) = 0.78  # High! (shares "movies", "war")
similarity(q_emb, d2_emb) = 0.65  # Lower (only "movies")

Ranking: [d1, d2]  ← WRONG! User wants non-war movies
```

**Cross-Encoder**:
```python
score("[CLS] movies NOT about war [SEP] Saving Private Ryan: war movie [SEP]")
  → Attention sees "NOT" + "war" in both contexts
  → Score = 0.12 (low)

score("[CLS] movies NOT about war [SEP] The Notebook: romance movie [SEP]")
  → Attention sees "NOT" + absence of "war"
  → Score = 0.89 (high)

Ranking: [d2, d1]  ← CORRECT!
```

**Why it works**: Cross-encoder sees "NOT" modifying "war" during attention.

---

### Computational Comparison

**Task**: Rank 1000 documents for a query

**Bi-Encoder**:
```
Encode query: 1 forward pass
Search pre-computed embeddings: ~1ms (FAISS)
Total: ~2ms
```

**Cross-Encoder**:
```
Encode each (query, doc) pair: 1000 forward passes
Each forward pass: ~10ms
Total: ~10,000ms (10 seconds!)
```

**Speedup**: Bi-encoder is **5000x faster**.

**Solution**: Use bi-encoder to reduce to top-20, then cross-encoder on 20.
```
Bi-encoder: 1M docs → top-20 (2ms)
Cross-encoder: 20 docs (200ms)
Total: 202ms (vs 10,000ms)
```

---

## Cross-Encoder Architecture

### Input Construction

**Format**: `[CLS] query [SEP] document [SEP]`

**Example**:
```
Query: "Python programming tutorials"
Document: "Learn Python with our comprehensive guide"

Input tokens:
[CLS] Python programming tutorials [SEP] Learn Python with our comprehensive guide [SEP]
```

**Token IDs**:
```
[101, 3821, 4730, 15881, 102, 4553, 3821, 1059, 1039, 10122, 5009, 102]
  ↑                        ↑                                          ↑
[CLS]                    [SEP]                                      [SEP]
```

---

### Self-Attention Mechanism

**Standard Transformer self-attention** operates on the **concatenated sequence**.

**Attention Matrix** (simplified 6 tokens):
```
              [CLS] Python tutorials [SEP] Learn guide [SEP]
[CLS]          1.0   0.3     0.2     0.1    0.1   0.1   0.1
Python         0.3   1.0     0.6     0.1    0.7   0.2   0.1
tutorials      0.2   0.6     1.0     0.1    0.4   0.5   0.1
[SEP]          0.1   0.1     0.1     1.0    0.1   0.1   1.0
Learn          0.1   0.7     0.4     0.1    1.0   0.6   0.1
guide          0.1   0.2     0.5     0.1    0.6   1.0   0.1
[SEP]          0.1   0.1     0.1     1.0    0.1   0.1   1.0
```

**Key observations**:
1. **"Python" (query) attends to "Python" (doc)**: 0.7 (high)
2. **"tutorials" (query) attends to "guide" (doc)**: 0.5 (semantic match)
3. **Cross-interaction**: Query tokens see document tokens directly

**Bi-encoder cannot do this**: Each is encoded separately, no cross-attention.

---

### Output: Relevance Score

**After all Transformer layers**:

```
[CLS] representation → [linear layer] → sigmoid → score ∈ [0, 1]
```

**Training objective** (binary cross-entropy):
```
Loss = -[y × log(score) + (1-y) × log(1-score)]

Where:
- y = 1 if document is relevant
- y = 0 if document is not relevant
```

**At inference**:
```
score = model("[CLS] query [SEP] doc [SEP]")
```

Higher score = more relevant.

---

## BGE-Reranker Model

### Model Specifications

**BGE-reranker-base**:
- **Base**: BERT-base (110M parameters)
- **Layers**: 12 Transformer layers
- **Hidden size**: 768
- **Max sequence length**: 512 tokens
- **Output**: Single relevance score [0, 1]

**BGE-reranker-large** (optional, more accurate but slower):
- 24 layers, 1024 hidden, 355M parameters

---

### Training Procedure

**Dataset**: Same as BGE embeddings
- MS MARCO passages
- Natural Questions
- HotpotQA
- Synthetic hard negatives

**Positive pairs**:
```
(query, relevant_doc) → label = 1
```

**Negative pairs**:
```
(query, irrelevant_doc) → label = 0
```

**Hard negative mining**:
1. Use bi-encoder to retrieve top-100
2. Label ground truth as positive (label=1)
3. Label near-misses (rank 5-100) as negatives (label=0)
4. Train cross-encoder to distinguish

**Why hard negatives?**
Easy negatives (random docs) are trivial. Hard negatives (topically similar but not relevant) force the model to learn **fine-grained distinctions**.

---

### Inference

**Input**:
```python
query = "What is the capital of France?"
candidates = [
    "Paris is the capital of France",
    "France is a country in Europe",
    "Lyon is a city in France"
]
```

**Process**:
```python
scores = []
for doc in candidates:
    input_text = f"[CLS] {query} [SEP] {doc} [SEP]"
    score = reranker(input_text)
    scores.append(score)

# scores = [0.95, 0.72, 0.68]

# Sort by score
ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
# [("Paris is the capital...", 0.95), ...]
```

---

## Integration Strategies

### Strategy 1: Re-rank Hybrid Top-K

**Pipeline**:
```
Query
  ↓
Hybrid Retrieval (dense + sparse + RRF)
  ↓
Top-20 candidates
  ↓
Cross-Encoder Re-ranking
  ↓
Top-5 final results
```

**Parameters**:
- Retrieve `K=20` from hybrid
- Re-rank to get final `K=5`

**Why K=20?**
- Recall@20 is very high (>98%)
- Re-ranking 20 docs is fast (~200ms)
- Covers most relevant documents

---

### Strategy 2: Cascade Re-ranking

**For very large K** (e.g., top-100):

```
Query
  ↓
Dense Retrieval → Top-100
  ↓
BM25 Re-rank → Top-50
  ↓
Cross-Encoder Re-rank → Top-10
```

**Multiple stages** progressively refine.

---

### Strategy 3: Adaptive Re-ranking

**Idea**: Only re-rank when needed.

```python
if top1_score - top2_score < threshold:
    # Scores are close, re-rank
    reranked = cross_encoder.rerank(top_20)
else:
    # Top result is clearly best, skip re-ranking
    reranked = top_20
```

**Saves compute** when bi-encoder is already confident.

---

## Trade-offs and Optimization

### Latency Analysis

**Without re-ranking** (Hybrid):
```
Dense retrieval: 5.4ms
BM25 retrieval: 1.7ms (parallel)
RRF fusion: 0.1ms
Total: ~5.5ms
```

**With re-ranking**:
```
Hybrid retrieval: 5.5ms
Re-rank 20 docs: ~15ms (GPU, batched)
Total: ~20ms
```

**Trade-off**: **4x slower** for **+2% Recall@5**.

---

### Batch Processing

**Naive** (sequential):
```python
for doc in top_20:
    score = reranker(f"{query} [SEP] {doc}")
# 20 forward passes = 20 × 10ms = 200ms
```

**Optimized** (batched):
```python
inputs = [f"{query} [SEP] {doc}" for doc in top_20]
scores = reranker.batch_score(inputs)  # Single forward pass
# 1 batched forward pass = 15ms
```

**Speedup**: **13x faster** (200ms → 15ms).

---

### GPU Memory

**Cross-encoder memory** (BGE-reranker-base):
- Model: ~500 MB (FP16)
- Batch of 20: ~200 MB activations
- **Total**: ~700 MB

**With BGE embeddings** (from Step 3):
- BGE-large: 700 MB
- Reranker: 700 MB
- **Total**: 1.4 GB (fits in 6GB GPU ✓)

---

### When to Use Re-ranking

| Scenario | Use Re-ranking? | Reasoning |
|----------|-----------------|-----------|
| **Latency <10ms required** | ❌ No | Too slow |
| **Top-1 accuracy critical** | ✅ Yes | +5-10% improvement |
| **Cost-sensitive** | ❌ No | 4x more compute |
| **Production QA system** | ✅ Yes | Quality matters |
| **Real-time search** | ⚠️ Maybe | Depends on budget |

---

## Performance Expectations

### Empirical Gains (from literature)

| Metric | Bi-Encoder | + Re-ranking | Gain |
|--------|------------|--------------|------|
| **Recall@5** | 85-90% | 90-95% | +5% |
| **MRR** | 0.75-0.80 | 0.82-0.87 | +5-7% |
| **nDCG@10** | 0.60-0.65 | 0.68-0.73 | +8% |

**For our system** (already strong baseline):
- Dense: 95% Recall@5
- Hybrid: 96% Recall@5
- **Hybrid + Rerank**: 97-98% Recall@5 (expected)

**Why smaller gain?** Diminishing returns (already at 96%).

---

## Key Takeaways

1. **Cross-encoders are more accurate** (+5-10%) but 100x slower than bi-encoders

2. **Two-stage retrieval is essential**:
   - Stage 1: Bi-encoder (fast, retrieve top-100)
   - Stage 2: Cross-encoder (precise, re-rank top-20)

3. **BGE-reranker uses full attention** between query and document tokens

4. **Trade-off**: +2% accuracy for 4x latency

5. **Batch processing critical**: 13x speedup over sequential

6. **Memory efficient**: Fits in 6GB GPU alongside embeddings

7. **Use when**: Quality matters more than speed (<100ms latency OK)

---

## Next Steps

1. Implement BGE-reranker in `src/reranker.py`
2. Integrate with hybrid retrieval pipeline
3. Evaluate on SQuAD queries
4. Measure latency vs quality trade-off
5. Compare: Dense → Hybrid → Hybrid+Rerank

Continue to **[05_generation_theory.md](05_generation_theory.md)** →
