# Generation Theory - LLM Integration for RAG

## Table of Contents
1. [Why Generation?](#why-generation)
2. [From Retrieval to Generation](#from-retrieval-to-generation)
3. [LLM Selection: Phi-3-mini](#llm-selection)
4. [Quantization: 4-bit with bitsandbytes](#quantization)
5. [Prompt Engineering](#prompt-engineering)
6. [Evaluation Metrics](#evaluation-metrics)
7. [Baseline vs Agentic Preview](#baseline-vs-agentic)

---

## Why Generation?

### The RAG Pipeline Evolution

**Retrieval alone** returns relevant chunks, but:
- User gets raw text fragments
- No synthesis or reasoning
- No direct answer to question

**Generation** transforms chunks into answers:
```
Chunks: ["Paris is the capital of France", "France is in Europe"]
Question: "What is the capital of France?"

Without Generation → User reads 2 chunks manually
With Generation    → "Paris is the capital of France."
```

---

### Retrieval vs Generation Tasks

| Stage | Input | Output | Model Type |
|-------|-------|--------|------------|
| **Retrieval** | Query | Top-K chunks | Embeddings (BGE) |
| **Re-ranking** | Query + Chunks | Ranked chunks | Cross-encoder |
| **Generation** | Query + Chunks | Natural language answer | LLM (Phi-3) |

**Generation is the final step** that makes RAG user-friendly.

---

## From Retrieval to Generation

### The Complete Pipeline

```
User Query: "What programming language was used for the Apollo 11 guidance system?"

↓

[RETRIEVAL] (Hybrid: Dense + BM25 + RRF)
Retrieved 20 candidates

↓

[RE-RANKING] (Cross-encoder)
Top 5 chunks:
1. "The Apollo Guidance Computer used Assembly language..."
2. "AGC assembly code was hand-optimized for memory..."
3. "Margaret Hamilton led the software team..."
4. "The guidance system had 72KB of memory..."
5. "Real-time constraints required low-level programming..."

↓

[GENERATION] (LLM with context)
Prompt:
"""
Based on the provided context, answer the question concisely.

Context:
[1] The Apollo Guidance Computer used Assembly language...
[2] AGC assembly code was hand-optimized for memory...
...

Question: What programming language was used for the Apollo 11 guidance system?

Answer:
"""

↓

Generated Answer: "Assembly language was used for the Apollo 11 guidance
computer. The code was hand-optimized due to the system's 72KB memory constraint."
```

---

## LLM Selection

### Why Phi-3-mini-4k-instruct?

**Constraints**:
- VRAM: 6GB total
- Already loaded: BGE-large (1.4GB) + Reranker (0.7GB) = 2.1GB
- **Available for LLM**: ~3.9GB

**Options evaluated**:

| Model | Parameters | VRAM (4-bit) | Speed | Quality |
|-------|-----------|--------------|-------|---------|
| **Mistral-7B** | 7.2B | 4GB | Medium | Excellent |
| **Llama-3.2-3B** | 3.2B | 2GB | Fast | Good |
| **Phi-3-mini-4k** | 3.8B | 2GB | Fast | Excellent |
| Gemma-2B | 2B | 1GB | Very Fast | Fair |

**Winner: Phi-3-mini-4k-instruct (Microsoft)**

**Why?**
1. **SOTA performance**: Outperforms Llama-3.2-3B on MMLU, HumanEval
2. **Optimized for instructions**: Excellent at following RAG prompts
3. **4K context**: Can handle 5 chunks (~2K tokens) + query comfortably
4. **Memory efficient**: 2GB leaves 1.9GB headroom
5. **Commercial-friendly**: MIT license

**Benchmark comparison** (from Microsoft paper):
```
MMLU (knowledge):
- Phi-3-mini: 68.8%
- Llama-3.2-3B: 63.0%

HumanEval (coding):
- Phi-3-mini: 59.1%
- Llama-3.2-3B: 50.6%

MT-Bench (instruction following):
- Phi-3-mini: 8.38/10
- Llama-3.2-3B: 7.95/10
```

---

## Quantization

### What is 4-bit Quantization?

**Full precision (FP32)**: Each weight = 32 bits
```
Weight = 0.42317894 → 4 bytes
```

**4-bit (INT4)**: Each weight = 4 bits
```
Weight ≈ 0.4 → 0.5 bytes (8x compression)
```

**Memory savings**:
```
Phi-3-mini (3.8B parameters):
- FP32: 3.8B × 4 bytes = 15.2 GB
- 4-bit: 3.8B × 0.5 bytes = 1.9 GB
```

---

### bitsandbytes Library

**How it works**:
1. **NF4 (NormalFloat4)**: Custom 4-bit format optimized for neural network weights
2. **Dynamic dequantization**: Weights stored as 4-bit, converted to FP16 during computation
3. **Minimal accuracy loss**: ~1-2% degradation vs FP16

**Implementation**:
```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,  # Compute in FP16
    bnb_4bit_use_double_quant=True,        # Quantize quantization constants
    bnb_4bit_quant_type="nf4"              # NormalFloat4
)

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    quantization_config=quantization_config,
    device_map="auto"
)
```

**Trade-offs**:
- ✅ 8x memory reduction
- ✅ 1.5-2x faster inference (smaller memory transfers)
- ❌ 1-2% quality degradation (acceptable for RAG)
- ❌ Requires CUDA (no CPU quantization)

---

## Prompt Engineering

### Prompt Structure for RAG

**Goal**: Get accurate, concise answers that cite the provided context.

**Anti-patterns** (what NOT to do):
```python
# BAD: No context grounding
prompt = f"Answer: {query}"

# BAD: Context after question (models forget context)
prompt = f"Question: {query}\n\nContext: {chunks}"

# BAD: Encourages hallucination
prompt = f"Use your knowledge and the context to answer: {query}"
```

**Best practice**: Context-first, explicit instructions

---

### Prompt Template (Baseline)

```python
PROMPT_TEMPLATE = """Based on the provided context, answer the question concisely.
If the answer cannot be found in the context, respond with "I cannot answer based on the provided information."

Context:
{context}

Question: {question}

Answer:"""
```

**Why this works**:
1. **Context before question**: Model sees evidence first
2. **Explicit grounding**: "Based on the provided context"
3. **Refusal instruction**: Prevents hallucination on unanswerable questions
4. **Concise directive**: Avoids verbose responses

---

### Few-Shot Prompting (Optional Enhancement)

Add 1-2 examples to guide the model:

```python
PROMPT_TEMPLATE_FEW_SHOT = """Based on the provided context, answer the question concisely.
If the answer cannot be found in the context, respond with "I cannot answer based on the provided information."

Example 1:
Context: The Eiffel Tower was completed in 1889 for the Paris World's Fair.
Question: When was the Eiffel Tower built?
Answer: The Eiffel Tower was completed in 1889.

Example 2:
Context: The Great Wall of China is approximately 21,196 kilometers long.
Question: How tall is the Great Wall?
Answer: I cannot answer based on the provided information.

Now answer the following:

Context:
{context}

Question: {question}

Answer:"""
```

**Trade-off**: +100 tokens per query, but +5-10% accuracy on edge cases.

---

### Chain-of-Thought (CoT) for Complex Queries

For multi-hop questions, encourage reasoning:

```python
COT_TEMPLATE = """Based on the provided context, answer the question.
First explain your reasoning, then provide the final answer.

Context:
{context}

Question: {question}

Reasoning:
Answer:"""
```

**Example**:
```
Question: "Who led the software team for the Apollo 11 guidance system
           and what language did they use?"

Reasoning: The context states Margaret Hamilton led the software team.
           It also mentions the guidance computer used Assembly language.

Answer: Margaret Hamilton led the software team, and they used Assembly language.
```

**Trade-off**: 2x longer responses, but better for complex queries.

**For Step 6 baseline**: We'll use the **simple template** (not CoT).

---

## Evaluation Metrics

### Why Multiple Metrics?

No single metric captures all aspects:

| Aspect | Metric | What it measures |
|--------|--------|------------------|
| **Correctness** | EM, F1 | Does answer match ground truth? |
| **Fluency** | ROUGE-L | Is answer well-formed? |
| **Faithfulness** | Faithfulness Score | Does answer cite context (no hallucination)? |

---

### 1. Exact Match (EM)

**Definition**: Binary - answer exactly matches ground truth (after normalization).

**Normalization**:
1. Lowercase
2. Remove punctuation
3. Remove articles (a, an, the)
4. Strip whitespace

**Example**:
```python
Question: "What is the capital of France?"
Ground truth: "Paris"
Prediction: "The capital is Paris."

Normalized GT: "paris"
Normalized Pred: "capital paris"

EM = 0 (not exact match)
```

**Characteristics**:
- Strict, unambiguous
- Too harsh (penalizes synonyms, paraphrasing)

**Typical RAG scores**: 40-60% EM (lower than you'd expect)

---

### 2. F1 Token Overlap

**Definition**: Harmonic mean of precision and recall on word tokens.

**Formula**:
```
Precision = |GT tokens ∩ Pred tokens| / |Pred tokens|
Recall    = |GT tokens ∩ Pred tokens| / |GT tokens|

F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Example**:
```python
GT: "Paris is the capital"
Pred: "The capital is Paris"

GT tokens: {paris, is, the, capital}
Pred tokens: {the, capital, is, paris}

Overlap: {paris, is, the, capital} = 4 tokens

Precision = 4/4 = 1.0
Recall = 4/4 = 1.0
F1 = 1.0
```

**Characteristics**:
- Handles paraphrasing
- More lenient than EM
- Ignores word order

**Typical RAG scores**: 60-80% F1

---

### 3. ROUGE-L (Longest Common Subsequence)

**Definition**: Measures longest common subsequence between GT and prediction.

**Why "Longest"?**: Captures sentence structure, not just bag-of-words.

**Example**:
```python
GT: "Paris is the capital of France"
Pred: "The capital of France is Paris"

LCS: "capital of France" (length = 3 words)

ROUGE-L = 2 × LCS / (len(GT) + len(Pred))
        = 2 × 3 / (6 + 6) = 0.5
```

**Characteristics**:
- ✅ Captures fluency/structure
- ✅ Used in summarization
- ❌ Can be gamed by repeating words

**Typical RAG scores**: 50-70% ROUGE-L

---

### 4. Faithfulness Score (Hallucination Detection)

**Definition**: Fraction of answer content that is supported by retrieved chunks.

**Method** (NLI-based):
1. Split answer into sentences
2. For each sentence, check if it's **entailed** by any chunk
3. Faithfulness = % entailed sentences

**Example**:
```python
Chunks:
- "Paris is the capital of France"
- "The Eiffel Tower is in Paris"

Answer: "Paris is the capital of France, located near Berlin"

Sentences:
1. "Paris is the capital of France" → Entailed by Chunk 1
2. "located near Berlin" → NOT entailed (hallucination!)

Faithfulness = 1/2 = 0.5 (50% faithful)
```

**Implementation options**:

**Option A: NLI model** (slower, more accurate)
```python
from transformers import pipeline

nli = pipeline("text-classification", model="microsoft/deberta-v3-base-mnli")

for sentence in answer_sentences:
    for chunk in retrieved_chunks:
        result = nli(f"{chunk} [SEP] {sentence}")
        if result['label'] == 'ENTAILMENT':
            # Sentence is supported
```

**Option B: Embedding similarity** (faster, approximate)
```python
sentence_emb = embed_model.encode([sentence])
chunk_embs = embed_model.encode(retrieved_chunks)

max_similarity = max(cosine(sentence_emb, chunk_embs))

if max_similarity > 0.8:
    # Sentence is supported
```

**For Step 6**: We'll use **Option B** (embedding-based) for speed.

**Typical scores**: 80-95% faithfulness (lower = more hallucination)

---

### Metrics Summary Table

| Metric | Range | What it measures | Good score | Bad score |
|--------|-------|------------------|------------|-----------|
| **EM** | 0-1 | Exact correctness | >0.50 | <0.30 |
| **F1** | 0-1 | Token overlap | >0.70 | <0.50 |
| **ROUGE-L** | 0-1 | Sequence similarity | >0.60 | <0.40 |
| **Faithfulness** | 0-1 | Anti-hallucination | >0.85 | <0.70 |

**Combined interpretation**:
```
High EM + High Faithfulness → Excellent (correct + grounded)
Low EM + High Faithfulness  → Paraphrasing (correct idea, different words)
High EM + Low Faithfulness  → Lucky guess (correct but not from context)
Low EM + Low Faithfulness   → Hallucination (wrong + made up)
```

---

## Baseline vs Agentic Preview

### Linear Pipeline (Step 6 - This Session)

```
Query → Retrieval → Re-ranking → Generation → Answer
```

**Characteristics**:
- Simple, fast (~500ms end-to-end)
- Easy to debug
- No error correction
- Fails if retrieval misses relevant docs

**Failure mode example**:
```
Query: "What was the first programming language used in space?"

Retrieved chunks (all irrelevant):
- "Modern rockets use C++ for guidance systems"
- "Python is popular in data science"
- "JavaScript runs in web browsers"

Generation (hallucination):
"The first programming language used in space was FORTRAN."
```

**Why it fails**: Garbage in, garbage out. No mechanism to detect/fix bad retrieval.

---

### Agentic RAG (Step 6.5 - Next Step)

```
Query → Retrieval → [Grade Relevance]
                        ↓
                    Relevant?
                    ↙      ↘
                  Yes       No
                   ↓         ↓
              Generation  Rewrite Query
                           ↓
                      Retry Retrieval
```

**Additional capabilities**:
1. **Document grading**: LLM checks if retrieved docs are useful
2. **Query rewriting**: Reformulates query if docs are bad
3. **Iterative refinement**: Retries up to N times
4. **Hallucination detection**: Validates answer against context

**Same query, agentic approach**:
```
Query: "What was the first programming language used in space?"

Retrieval → 5 chunks (all about modern languages)

Grade: "These documents discuss modern languages, not historical space programs.
        Not relevant to the query."

Rewrite Query: "Apollo program assembly language guidance computer"

Retry Retrieval → Better chunks:
- "The Apollo Guidance Computer used Assembly..."

Generation: "Assembly language was used for the Apollo guidance computer."
```

**Trade-offs**:
- ✅ Higher accuracy on hard queries (+10-15% F1)
- ✅ Self-correcting
- ❌ 2-3x slower (multiple LLM calls)
- ❌ More complex to implement

---

### When to Use Each Approach?

| Use Case | Best Approach | Why |
|----------|---------------|-----|
| **High-volume, low-stakes** (e.g., FAQ bot) | Linear | Speed matters, errors acceptable |
| **High-stakes QA** (e.g., medical, legal) | Agentic | Accuracy critical |
| **Known good retrieval** (e.g., curated docs) | Linear | Retrieval rarely fails |
| **Noisy corpus** (e.g., web scraping) | Agentic | Retrieval often fails |

**For this project**: We build **both** to demonstrate the trade-off.

---

## Implementation Roadmap

### Step 6 (This Session): Baseline

**Goals**:
1. Integrate Phi-3-mini-4k (4-bit quantized)
2. Implement prompt template
3. Compute EM, F1, ROUGE-L, Faithfulness
4. Establish baseline on 500-query dataset

**Expected results**:
- EM: 45-55%
- F1: 65-75%
- ROUGE-L: 55-65%
- Faithfulness: 85-90%
- Latency: 400-600ms per query

---

### Step 6.5 (Next Session): Agentic with LangGraph

**Enhancements**:
1. Document relevance grading
2. Query rewriting
3. Iterative retrieval (max 2 retries)
4. Hallucination detection + regeneration

**Expected improvements**:
- Overall: +2-3% F1 (marginal, most queries already work)
- **Hard subset** (Recall@5 < 80%): +10-15% F1 (significant!)
- Latency: 800-1500ms (2-3x slower, acceptable for quality gain)

---

## Key Takeaways

1. **Generation completes RAG**: Transforms chunks → natural language answers

2. **Phi-3-mini is optimal** for 6GB VRAM: 3.8B params, SOTA quality, 2GB memory

3. **4-bit quantization** enables large models on consumer GPUs: 8x memory savings, minimal quality loss

4. **Prompt engineering matters**: Context-first, explicit grounding, refusal handling

5. **Multiple metrics required**:
   - EM/F1 for correctness
   - ROUGE-L for fluency
   - Faithfulness for anti-hallucination

6. **Baseline vs Agentic**: Linear is fast, Agentic handles edge cases

7. **Next step**: Build baseline, measure metrics, identify failure modes for LangGraph

---

## References

- **Phi-3 Paper**: [Microsoft Research](https://arxiv.org/abs/2404.14219)
- **bitsandbytes**: [GitHub](https://github.com/TimDettmers/bitsandbytes)
- **RAG Evaluation**: [Ragas Framework](https://docs.ragas.io/)
- **SQuAD Metrics**: [Rajpurkar et al., 2016](https://arxiv.org/abs/1606.05250)

Continue to **scripts/generation_baseline.py** for implementation →
