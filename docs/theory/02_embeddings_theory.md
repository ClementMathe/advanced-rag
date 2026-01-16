# Embeddings & Vector Search - Deep Dive

## Table of Contents
1. [What Are Embeddings?](#what-are-embeddings)
2. [Mathematical Foundation](#mathematical-foundation)
3. [Training Embeddings: Contrastive Learning](#training-embeddings)
4. [BGE: BAAI General Embedding](#bge-model)
5. [Similarity Metrics](#similarity-metrics)
6. [FAISS: Vector Search at Scale](#faiss)
7. [Dense vs Sparse Retrieval](#dense-vs-sparse)
8. [Practical Considerations](#practical-considerations)

---

## What Are Embeddings?

### The Core Idea

An **embedding** is a mathematical transformation that maps text (discrete symbols) into a continuous vector space where semantic similarity corresponds to geometric proximity.

**Formal definition**:
```
f: Text → ℝᵈ

Where:
- Text = sequence of tokens (words, subwords)
- ℝᵈ = d-dimensional real vector space
- d = embedding dimension (typically 384, 768, 1024)
```

### Why Embeddings Work

**The Distributional Hypothesis** (Harris, 1954):
> "Words that occur in similar contexts tend to have similar meanings."

**Extended to sentences/documents**:
> "Texts that discuss similar topics will have similar word distributions."

**Example**:
```
Sentence 1: "The cat sat on the mat."
Sentence 2: "A feline rested on the rug."

Traditional overlap: 0 words in common
Embedding similarity: 0.87 (very similar!)
```

The embedding model learned that:
- "cat" ≈ "feline"
- "sat" ≈ "rested"
- "mat" ≈ "rug"

---

## Mathematical Foundation


### Step 1: Tokenization
```
Text: "Paris is beautiful"
Tokens: ["Paris", "is", "beautiful"]
Token IDs: [3421, 67, 1289]
```

### Step 2: Token Embeddings
Each token ID maps to a learned vector:
```
Token embedding matrix E ∈ ℝ^(V × d)

Where:
- V = vocabulary size (e.g., 30,000)
- d = embedding dimension (e.g., 768)

E[3421] = [0.23, -0.15, ..., 0.89]  # "Paris"
E[67]   = [0.02, 0.33, ..., -0.12]  # "is"
E[1289] = [-0.45, 0.67, ..., 0.34]  # "beautiful"
```

### Step 3: Contextual Encoding (Transformer)

Token embeddings are processed through multiple Transformer layers:

```
Input: Token embeddings [e₁, e₂, ..., eₙ]
Output: Contextual embeddings [h₁, h₂, ..., hₙ]

Where each hᵢ considers context from all other tokens via self-attention.
```
#### **Transformer = Neural Encoder Block**

A **Transformer** refers **only** to the neural architecture composed of:

1. Multi‑Head Self‑Attention
2. Add & LayerNorm
3. Feed‑Forward Network (MLP)
4. Add & LayerNorm


This block is **stacked N times** (e.g. 12, 24 layers).

❌ Not part of the Transformer : Tokenization / Vocabulary / Padding logic / Initial token embeddings

> The Transformer starts **after** tokens have already been converted into vectors.
```
+---------------------------+
|        Input Text         |
| "Paris is beautiful"      |
+---------------------------+
             |
             v
+---------------------------+
| Tokenization & Embedding  |
| x1 = "Paris" → [1.0,0.0] |
| x2 = "is"    → [0.0,1.0] |
| x3 = "beautiful" → [1,1] |
+---------------------------+
             |
             v
+--------------------------------+
| Transformer Layer 1            |
|                                |
| Multi-Head Self-Attention:     |
| head1 → H1_head1               |
| head2 → H2_head2               |
| ...                            |
| Concatenate → Linear Projection|
| → H1_tokens = [h1,h2,h3]       |
| FFN + Residual + LayerNorm     |
+--------------------------------+
             |
             v
+--------------------------------+
| Transformer Layer 2            |
| Multi-Head Self-Attention      |
| on H1_tokens → H2_tokens       |
| FFN + Residual + LayerNorm     |
+--------------------------------+
             |
             v
            ...
             |
             v
+---------------------------+
|      Pooling Layer        |
| Mean Pooling:             |
| sentence_embedding =      |
| (h1 + h2 + h3)/3          |
| => 1 vector ∈ ℝ^{d_model} |
+---------------------------+
             |
             v
+---------------------------+
|  Sentence Embedding Ready |
|   (Dense Vector, 1024D)   |
+---------------------------+
```

#### **Self-Attention mechanism**:

> **Each token decides which other tokens are important to understand its meaning.**

```
H_Attention(Q, K, V) = softmax(QKᵀ / √dₖ) .  V

Where:
- Q = queries (what we're looking for)
- K = keys (what's available)
- V = values (the actual content)
- dₖ = dimension of keys (for scaling)

Computed as:
Q = X · W_Q
K = X · W_K
V = X · W_V

With W_Q, W_K, W_V ∈ ℝ^(d × dₖ) three learned projection matrices:
```

Output = **contextualized token embeddings** :
- each token becomes a **weighted sum** of other tokens
- weights = semantic relevance


**Example**:
```
Sentence: "The bank is near the river."

Token "bank" attends to:
- "river" (high attention) → financial bank or riverbank?
- "near" (medium attention)
- "the" (low attention)

Output: h_bank incorporates "river" context → riverbank meaning
```
> **Multi‑Head Self‑Attention** = Multiple **self-attention** mechanisms computed in **parallel**, each with its own Q, K, and V projections, then concatenated and linearly combined.

### Step 4: Pooling

Convert sequence of token embeddings into a single sentence embedding:

**Common pooling strategies**:

1. **CLS token** (BERT-style):
   ```
   sentence_embedding = h[CLS]
   ```
   - First token represents entire sequence.
   - Designed to represent the **entire sequence**

   Used for classification / sentiment analysis / entailment


2. **Mean pooling** (most common for sentence embeddings):
   ```
   sentence_embedding = (1/n) Σᵢ hᵢ
   ```
   Average all token embeddings.

3. **Max pooling**:
   ```
   sentence_embedding[j] = maxᵢ hᵢ[j]
   ```
   Take maximum value in each dimension.

**BGE uses mean pooling** with attention mask (ignores padding tokens).

---

## Training Embeddings: Contrastive Learning

### The Goal

Learn embeddings such that:
- **Similar texts** → vectors close together (high cosine similarity)
- **Dissimilar texts** → vectors far apart (low/negative cosine similarity)

### Contrastive Loss Function

**InfoNCE Loss** (Noise Contrastive Estimation):

```
ℒ = -log( exp(sim(q, p⁺) / τ) / Σₖ exp(sim(q, pₖ) / τ) )

Where:
- q = query embedding
- p⁺ = positive example (semantically similar)
- {pₖ} = negative examples (dissimilar)
- sim() = similarity function (cosine similarity)
- τ = temperature hyperparameter (controls sharpness)
```

**Intuition**: Maximize similarity to positive, minimize to negatives.

### Training Data Construction

**Positive pairs** (texts that should be similar):
1. **Question-Answer pairs**:
   ```
   Query: "What is the capital of France?"
   Positive: "Paris is the capital city of France."
   ```

2. **Title-Body pairs**:
   ```
   Title: "Introduction to Machine Learning"
   Body: "Machine learning is a subset of AI that enables..."
   ```

3. **Paraphrase pairs**:
   ```
   Text 1: "The cat is on the mat."
   Text 2: "A feline sits on the rug."
   ```

4. **Same document, different excerpts**:
   ```
   Chunk 1: "Einstein published general relativity in 1915..."
   Chunk 2: "...this theory revolutionized our understanding of gravity."
   ```

**Negative examples**:
- Random documents from the corpus
- Hard negatives: Similar topics but different meanings
- In-batch negatives: Other examples in the same training batch

### Hard Negative Mining

**Problem**: Random negatives are too easy (model learns trivial patterns).

**Solution**: Use **hard negatives** - texts that are topically similar but semantically different.

**Example**:
```
Query: "How does photosynthesis work?"

Easy negative: "The stock market crashed in 1929." (irrelevant)
Hard negative: "Cellular respiration converts glucose to ATP." (related biology, but different process)
```

**Impact**: Hard negatives force the model to learn **fine-grained distinctions**.

---

## BGE: BAAI General Embedding

### Model Architecture

**BGE-large-en-v1.5**:
- **Base model**: RoBERTa-large (355M parameters)
- **Architecture**: 24 Transformer layers, 1024 hidden size
- **Output dimension**: 1024
- **Max sequence length**: 512 tokens
- **Pooling**: Mean pooling with attention mask

### Training Procedure

1. **Pre-training**: Masked Language Modeling on large corpus
2. **Contrastive fine-tuning**: On diverse retrieval datasets
   - MS MARCO (web search queries)
   - Natural Questions (Wikipedia Q&A)
   - HotpotQA (multi-hop reasoning)
   - Custom synthetic data

3. **Instruction fine-tuning**: Prefix-based task specification
   ```
   Query: "Represent this sentence for searching relevant passages: [QUERY]"
   Document: "Represent this passage: [DOCUMENT]"
   ```

### Why BGE is Strong

1. **Large training corpus**: 1B+ text pairs
2. **Diverse tasks**: Search, QA, classification, clustering
3. **Hard negative mining**: High-quality contrastive pairs
4. **Instruction-aware**: Can adapt to different retrieval scenarios

### Performance Benchmarks

**MTEB (Massive Text Embedding Benchmark)**:
```
Model               Avg Score  Retrieval  Classification
─────────────────────────────────────────────────────────
BGE-large-en-v1.5      63.98      53.94        75.06
e5-large-v2            62.25      52.01        73.84
instructor-xl          61.79      51.23        74.12
```

BGE achieves **state-of-the-art** on multiple benchmarks.

---

## Similarity Metrics

### Cosine Similarity (Primary)

**Formula**:
```
cos_sim(A, B) = (A · B) / (||A|| ||B||)

Where:
- A · B = dot product = Σᵢ Aᵢ × Bᵢ
- ||A|| = L2 norm = √(Σᵢ Aᵢ²)
```

**Properties**:
- Range: [-1, 1]
- 1 = identical direction
- 0 = orthogonal (no similarity)
- -1 = opposite direction

**Why cosine instead of Euclidean distance?**

Cosine measures **angle**, Euclidean measures **magnitude**.

**Example**:
```
A = [1, 0]    B = [2, 0]    C = [0, 1]

Euclidean:
  dist(A, B) = 1 (close)
  dist(A, C) = √2 (far)

Cosine:
  sim(A, B) = 1.0 (identical direction)
  sim(A, C) = 0.0 (orthogonal)
```

For text, **direction matters more than magnitude**:
- "Paris is great." → [0.5, 0.3, -0.2, ...]
- "Paris is really, really great!" → [1.0, 0.6, -0.4, ...] (scaled version)

Cosine similarity = 1.0 (same meaning, different emphasis).

### Normalized Embeddings

**Key optimization**: If embeddings are normalized (||v|| = 1), then:

```
cos_sim(A, B) = A · B

(No division needed!)
```

**BGE normalizes embeddings by default**, so similarity = dot product.

**Benefit**: Faster computation (1 operation instead of 3).

### Other Similarity Metrics

#### Dot Product (Unnormalized)
```
sim(A, B) = A · B
```
- Faster than cosine
- Combines angle and magnitude
- Used in some retrieval systems

#### Euclidean Distance
```
dist(A, B) = √(Σᵢ (Aᵢ - Bᵢ)²)
```
- Common in clustering
- Not ideal for text embeddings (magnitude bias)

#### Manhattan Distance (L1)
```
dist(A, B) = Σᵢ |Aᵢ - Bᵢ|
```
- Simpler than Euclidean
- Rarely used for embeddings

**For RAG, always use cosine similarity (or dot product if normalized).**

---

## FAISS: Vector Search at Scale

### The Problem

**Naive search** (brute force):
```python
def search(query, vectors, k=5):
    similarities = [cosine_sim(query, v) for v in vectors]
    return top_k(similarities, k)
```

**Complexity**: O(N × d)
- N = number of vectors (millions)
- d = dimension (1024 for BGE)

**For 1M vectors**: ~1 billion operations per query (too slow!).

### FAISS Solution

**Facebook AI Similarity Search** (FAISS) uses **Approximate Nearest Neighbor** (ANN) algorithms.

**Trade-off**: Slightly lower recall for massive speedup.

```
Exact search:  O(N × d) → 100% recall, slow
FAISS:         O(√N × d) → 95-99% recall, 10-100x faster
```

> FAISS is not a model : it does vector search.

> It takes as input precomputed embeddings (dense vectors) and allows you to quickly find the top-k most similar vectors to a query.

### Index Types in FAISS

#### 1. Flat Index (IndexFlatL2, IndexFlatIP)

**How it works**: Brute force (no approximation).

```
- Stores all vectors
- Computes distance to every vector
- Returns exact top-K
```

**Use case**:
- Small datasets (<10k vectors)
- Baseline for comparison
- When 100% recall is critical

**Our choice for learning**: Flat index (exact search, simple).

#### 2. IVF (Inverted File Index)

**How it works**:
1. Cluster vectors into N cells (using k-means)
2. For each query, search only nearest M cells
3. Reduces candidates from N to ~M × (N/clusters)

**Parameters**:
- `nlist`: Number of clusters (e.g., 100)
- `nprobe`: Number of clusters to search (e.g., 10)

**Speedup**: ~10-20x faster than flat.

**Example**:
```
1M vectors, 1000 clusters, nprobe=10

Flat search: 1M comparisons
IVF search: 10 × 1000 = 10k comparisons (100x speedup!)
```

#### 3. HNSW (Hierarchical Navigable Small World)

**How it works**: Graph-based search
- Build hierarchical graph of connections
- Navigate from coarse to fine layers
- Follow edges to nearest neighbors

**Properties**:
- Very fast queries
- High recall (>95%)
- More memory than IVF

**Use case**: Production systems with large datasets (>1M vectors).

#### 4. PQ (Product Quantization)

**How it works**: Compress vectors
- Split vector into M sub-vectors
- Quantize each sub-vector (256 values)
- Store compressed representation (1 byte per sub-vector)

**Compression**: 1024D float32 (4KB) → 128 bytes (32x smaller!).

**Trade-off**: Lower precision, but massive memory savings.

**Use case**: Billion-scale datasets.

### Choosing an Index

| Dataset Size | Recommended Index | Speed | Recall |
|--------------|-------------------|-------|--------|
| <10k | Flat | Slow | 100% |
| 10k-100k | IVF (nlist=100) | Fast | 98% |
| 100k-1M | IVF (nlist=1000) | Fast | 97% |
| 1M-10M | HNSW | Very Fast | 95% |
| >10M | IVF+PQ | Fast | 95% |

**For this project**: Flat index (741 docs × ~1 chunk/doc = ~741 vectors).

---

## Dense vs Sparse Retrieval

### Dense Retrieval (Embeddings)

**Mechanism**: Semantic vector search
   - BGE produces embeddings (semantics)
   - FAISS stores and searches them efficiently


**Example success**:
```
Query: "How do I fix a leaky faucet?"
Match: "Repairing dripping taps in your home"
(Different words, same meaning)
```

**Example failure**:
```
Query: "Python 3.11 release date"
Match: "Python 3.10 features" (semantically similar but wrong version!)
```

### Sparse Retrieval (BM25, TF-IDF)

**Mechanism**: Keyword matching with weighting

**BM25 formula** (reminder):
```
score(D, Q) = Σ IDF(qᵢ) × f(qᵢ, D) × (k₁ + 1) / (f(qᵢ, D) + k₁ × (1 - b + b × |D| / avgdl))

Where
- f(qᵢ, D) = term frequency in the document
- IDF(qᵢ) = inverse document frequency
- |D| = document length
- avgdl = average document length
- k1, b = BM25 hyperparameters
```
**Ranking**: sort all documents by `score(D, Q)` in descending order → retrieve top-K documents.

**Example success**:
```
Query: "NVIDIA RTX 3090 specs"
Match: "NVIDIA RTX 3090 specifications: 24GB GDDR6X..."
(Exact keyword match)
```

**Example failure**:
```
Query: "How do I fix a leaky faucet?"
Match: (no match if document says "repairing dripping taps")
```

|Retrieval Strategy | ✅ **Strengths**  |  ❌ **Weaknesses** |
|-----------------|-------------|-------------|
| **Dense Retrieval (Embeddings)** | - Captures semantic similarity<br>- Handles paraphrasing ("car" = "automobile")<br>- Cross-lingual capabilities<br>- Works for vague queries  | - Struggles with rare terms / proper nouns<br>- "Out-of-distribution" queries fail<br>- No exact keyword matching<br>- Computationally expensive|
| **Sparse Retrieval (BM25, TF-IDF)**  | - Exact keyword matching<br>- Fast (no neural network)<br>- Handles rare terms well<br>- Interpretable scores | - No semantic understanding<br>- Vocabulary mismatch problem<br>- Fails on paraphrasing<br>- Requires same language|

### Hybrid Retrieval (Best of Both)

**Strategy**: Combine dense + sparse, then fuse rankings.

**Next step**: We'll implement Reciprocal Rank Fusion (RRF).

---

## Practical Considerations

### Attention Masks & Padding Reminder

- **Padding** ensures that all sequences in a batch have the same length (required for vectorization and GPU processing).
- Without a mask, the Transformer would treat `PAD` tokens as real tokens → introduces noise in embeddings.
- **Attention mask** = binary vector (1 = real token, 0 = padding) that **ignores padding tokens** during self-attention softmax computation and during pooling.
- BGE uses **mean pooling with attention mask**, so **padded tokens do not affect the final sentence embedding**.


### Embedding Dimensions

**Trade-offs**:

| Dimension | Pros | Cons |
|-----------|------|------|
| 384 | Fast, low memory | Lower expressiveness |
| 768 | Good balance | Standard choice |
| 1024 (BGE) | High expressiveness | Slower, more memory |

**Rule**: Higher dimensions → better quality, but diminishing returns.

### Batch Processing

**Single query embedding**:
```python
embedding = model.encode("query")  # ~50ms on GPU
```

**Batch (100 queries)**:
```python
embeddings = model.encode(queries)  # ~200ms on GPU (2.5x speedup per query!)
```

**Why batching helps**: GPU parallelization, amortized overhead.

### GPU vs CPU

**BGE-large-en-v1.5 on RTX 3060 (6GB)**:
- GPU: ~100 sentences/sec
- CPU: ~10 sentences/sec

**Speedup**: ~10x with GPU

### Memory Management

**BGE-large model size**:
- FP32: ~1.4 GB
- FP16: ~700 MB (2x smaller, minimal quality loss)

**FAISS index size**:
- Flat index: N × d × 4 bytes
- Example: 1000 vectors × 1024 dim × 4 bytes = 4 MB

**Total memory** (for our project):
- Model: ~700 MB (FP16)
- Index: ~4 MB
- **Total: <1 GB** (fits easily in 6GB GPU!)

### Inference Optimization

1. **Use FP16**: `model.half()` (2x speedup, minimal quality loss)
2. **Batch queries**: Process multiple at once
3. **Normalize embeddings**: Pre-compute for faster similarity
4. **Cache frequent queries**: Store results for repeated queries

---

## Key Takeaways

1. **Embeddings map text to vectors** where similarity = proximity

2. **Trained via contrastive learning** to maximize similarity for related texts

3. **BGE is state-of-the-art** for retrieval tasks (trained on 1B+ pairs)

4. **Cosine similarity** is the standard metric (measures angle, not magnitude)

5. **FAISS enables fast search** via approximate nearest neighbor algorithms

6. **Dense retrieval excels at semantic understanding** but struggles with exact matches

7. **Batch processing + GPU** are essential for performance

8. **Hybrid retrieval** (dense + sparse) combines strengths of both

---

## Next Steps

1. Implement `src/embeddings.py` with BGE model
2. Build FAISS index from SQuAD chunks
3. Test retrieval quality with sample queries
4. Measure latency and recall

Continue to **[03_reranking_theory.md](03_reranking_theory.md)** →
