# Advanced RAG System - Theoretical Overview

## Table of Contents
1. [Introduction](#introduction)
2. [RAG Architecture Fundamentals](#rag-architecture-fundamentals)
3. [Key Components](#key-components)
4. [Why This Architecture?](#why-this-architecture)
5. [Learning Path](#learning-path)

---

## Introduction

### What is RAG?

**Retrieval-Augmented Generation (RAG)** is a framework that enhances Large Language Models (LLMs) by grounding their responses in external knowledge sources. Instead of relying solely on parametric memory (knowledge encoded in model weights during training), RAG systems dynamically retrieve relevant information at inference time.

### The Fundamental Problem

LLMs face several limitations when used in isolation:
- **Knowledge Cutoff**: Training data ends at a specific date
- **Hallucination**: Generate plausible but incorrect information
- **No Source Attribution**: Cannot cite where information comes from
- **Domain Specificity**: General models lack deep expertise in specialized domains

RAG addresses these by providing **grounded, verifiable, up-to-date** responses.

---

## RAG Architecture Fundamentals

### Basic RAG Flow

```
Query → Retrieve Relevant Docs → Augment Prompt → Generate Answer
```

### Enhanced RAG Flow (This Project)

```
Query → Preprocess → Hybrid Retrieval → Re-rank → Select Context → Generate → Verify
          ↓              ↓                 ↓           ↓             ↓         ↓
      Normalize     Dense+Sparse      Cross-Encoder  Top-K      Mistral-7B  Metrics
```

### Why "Advanced"?

Standard RAG implementations often use:
- Single retrieval method (semantic search only)
- No re-ranking
- Fixed chunk size
- Limited evaluation

**This project implements**:
- **Hybrid retrieval**: Combines semantic (dense) and keyword (sparse) search
- **Cross-encoder re-ranking**: Improves precision significantly
- **Multiple chunking strategies**: Optimized for different document types
- **Comprehensive evaluation**: Both automated metrics and human-aligned scores

---

## Key Components

### 1. Document Processing & Chunking

**Challenge**: LLMs have context length limits. Documents must be split into manageable chunks.

**Strategies**:
- **Fixed-size chunking**: Simple, consistent (e.g., 512 tokens)
- **Semantic chunking**: Split at natural boundaries (paragraphs, sections)
- **Sliding window**: Overlapping chunks to preserve context

**Trade-offs**:
- **Smaller chunks**: Better precision, may lose context
- **Larger chunks**: Better context, may reduce relevance

**Implementation**: See `src/chunking.py`

---

### 2. Embeddings & Vector Search

#### What are Embeddings?

Embeddings transform text into high-dimensional vectors where semantic similarity corresponds to vector distance.

```
"Paris is the capital of France"
    → [0.23, -0.45, 0.67, ..., 0.12]  # 1024-dim vector

"France's capital is Paris"
    → [0.24, -0.44, 0.66, ..., 0.13]  # Very close in vector space!
```

#### Dense Retrieval

**Model**: BGE-large-en-v1.5 (1024 dimensions)

**Process**:
1. Encode all document chunks → Store in FAISS index
2. Encode user query → Search index for nearest neighbors
3. Return top-K most similar chunks

**Advantages**:
- Captures semantic meaning
- Handles paraphrasing
- Cross-lingual capabilities

**Limitations**:
- Struggles with exact keyword matches
- "Out-of-distribution" queries may fail

**Mathematical Foundation**:
```
similarity(query, doc) = cosine_similarity(embed(query), embed(doc))
                       = (q · d) / (||q|| ||d||)
```

**Implementation**: See `src/embeddings.py` and [detailed theory](01_embeddings.md)

---

### 3. Sparse Retrieval (BM25)

#### What is BM25?

BM25 (Best Matching 25) is a probabilistic ranking function based on term frequency and document length.

**Formula**:
```
score(D, Q) = Σ IDF(qi) · f(qi, D) · (k1 + 1) / (f(qi, D) + k1 · (1 - b + b · |D| / avgdl))
```

Where:
- `IDF(qi)`: Inverse Document Frequency (rarity of term)
- `f(qi, D)`: Term frequency in document
- `k1, b`: Tuning parameters (typically k1=1.5, b=0.75)
- `|D|`: Document length
- `avgdl`: Average document length in corpus

**Advantages**:
- Excellent for exact keyword matches
- Computationally efficient
- Interpretable scores

**Limitations**:
- No semantic understanding
- Vocabulary mismatch problem (synonyms not matched)

**Implementation**: See `src/retriever.py`

---

### 4. Hybrid Retrieval & Fusion

#### Why Combine Dense + Sparse?

Each method has complementary strengths:
- **Dense**: Semantic similarity, handles paraphrasing
- **Sparse**: Exact matches, rare terms, domain-specific keywords

#### Reciprocal Rank Fusion (RRF)

Combines rankings from multiple retrievers without needing score normalization.

**Formula**:
```
RRF_score(d) = Σ 1 / (k + rank_retriever_i(d))
```

Where:
- `k`: Constant (typically 60)
- `rank_retriever_i(d)`: Rank of document d in retriever i

**Example**:
```
Dense retrieval ranks: [doc1, doc3, doc5]
Sparse retrieval ranks: [doc3, doc1, doc7]

RRF scores:
- doc1: 1/(60+1) + 1/(60+2) ≈ 0.0328
- doc3: 1/(60+2) + 1/(60+1) ≈ 0.0328
- doc5: 1/(60+3) ≈ 0.0159
- doc7: 1/(60+3) ≈ 0.0159

Final ranking: [doc1, doc3, doc5, doc7]
```

**Implementation**: See [detailed theory](02_retrieval.md)

---

### 5. Cross-Encoder Re-Ranking

#### The Problem with Bi-Encoders

Bi-encoders (like BGE for dense retrieval) encode query and document independently:
```
score = similarity(encode(query), encode(doc))
```

This is **fast** but limits expressiveness—the model never sees query and document together.

#### Cross-Encoders

Cross-encoders process query and document **jointly**:
```
score = cross_encoder([query, doc])
```

**Architecture**:
```
[CLS] query [SEP] document [SEP] → Transformer → Single score
```

**Why Better?**:
- Full attention between query and document tokens
- Can model complex relevance patterns
- Significantly higher precision

**Trade-off**:
- ~100x slower than bi-encoders
- Used only for re-ranking top-K candidates (e.g., top-20 → top-5)

**Model**: BGE-reranker-base

**Typical Gains**:
- Recall@5: +15-25% improvement
- nDCG: +20-30% improvement

**Implementation**: See `src/reranker.py` and [detailed theory](03_reranking.md)

---

### 6. LLM Generation

#### Model: Mistral-7B-Instruct-v0.2

**Architecture**:
- 7 billion parameters
- Sliding Window Attention (window size: 4096)
- Grouped-Query Attention (GQA) for efficiency

**Prompt Structure**:
```
<s>[INST] Context:
{retrieved_documents}

Question: {user_query}

Provide a detailed answer based on the context above. [/INST]
```

**Generation Parameters**:
- `temperature`: 0.7 (balance creativity and consistency)
- `max_tokens`: 512
- `top_p`: 0.9

**Implementation**: See `src/llm.py`

---

### 7. Orchestration with LangGraph

#### Why LangGraph?

Traditional RAG implementations often use ad-hoc scripting. LangGraph provides:
- **Deterministic execution**: Graph-based workflow
- **Observability**: Track each step
- **Modularity**: Easy to modify individual components
- **Reproducibility**: Same input → same output path

#### Graph Structure

```python
StateGraph:
  - input_node: Validate and preprocess query
  - embed_node: Generate query embedding
  - retrieve_dense_node: Semantic search
  - retrieve_sparse_node: BM25 search
  - fusion_node: Combine with RRF
  - rerank_node: Cross-encoder scoring
  - select_node: Choose top-K contexts
  - generate_node: LLM generation
  - output_node: Format response
```

**Advantages**:
- Each node is testable independently
- Can add conditional logic (e.g., skip re-ranking if retrieval confidence is high)
- Visualizable workflow for debugging

**Implementation**: See `src/graph.py`

---

## Why This Architecture?

### Compared to Basic RAG

| Component | Basic RAG | Advanced RAG (This Project) | Improvement |
|-----------|-----------|----------------------------|-------------|
| Retrieval | Dense only | Dense + Sparse + Fusion | +15-20% recall |
| Ranking | None | Cross-encoder | +20-30% nDCG |
| Chunking | Fixed | Adaptive | Better context |
| Orchestration | Scripts | LangGraph | Reproducible |
| Evaluation | Manual | Automated metrics | Quantifiable |

### Production Considerations

This architecture balances:
1. **Accuracy**: Hybrid retrieval + re-ranking
2. **Latency**: Efficient bi-encoder for initial retrieval, cross-encoder only for top-K
3. **Cost**: Local models (no API costs)
4. **Scalability**: FAISS can handle millions of vectors
5. **Maintainability**: Modular, tested, documented

---

## Learning Path

### Phase 1: Fundamentals (Current)
- ✅ Project setup
- ✅ Understanding architecture
- 🔄 Implementing embeddings & vector search

### Phase 2: Core RAG
- Document processing & chunking
- Dense retrieval with FAISS
- Basic generation pipeline

### Phase 3: Advanced Retrieval
- BM25 implementation
- Hybrid retrieval with RRF
- Cross-encoder re-ranking

### Phase 4: Orchestration
- LangGraph integration
- State management
- Error handling

### Phase 5: Evaluation
- Ragas metrics (Precision, Recall, Faithfulness)
- Custom metrics (Recall@K, MRR, nDCG)
- Ablation studies

### Phase 6: Production
- API server (FastAPI)
- Monitoring & logging
- Performance optimization

---

## Key Takeaways

1. **RAG is not just vector search**: Hybrid retrieval + re-ranking make the difference
2. **Evaluation is critical**: Without metrics, you're flying blind
3. **Trade-offs are inevitable**: Speed vs. accuracy, cost vs. performance
4. **Modularity enables iteration**: Each component should be independently improvable
5. **Theory guides practice**: Understanding *why* methods work helps you debug and optimize

---

## Next Steps

1. Read detailed theory documents:
   - [Embeddings & Semantic Search](01_embeddings.md)
   - [Retrieval Strategies](02_retrieval.md)
   - [Re-ranking Techniques](03_reranking.md)
   - [Evaluation Metrics](04_metrics.md)

2. Implement components incrementally
3. Run experiments and analyze results
4. Document your findings

---

**Questions to Consider**:
- When would you use RAG over fine-tuning an LLM?
- How does chunk size affect retrieval quality?
- What are the failure modes of semantic search?
- How do you balance retrieval depth vs. generation quality?

Continue to **[01_chunking_theory.md](01_chunking_theory.md)** →
