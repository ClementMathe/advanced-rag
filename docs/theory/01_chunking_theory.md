# Chunking Theory - Deep Dive

## Table of Contents
1. [Why Chunking Matters](#why-chunking-matters)
2. [The Fundamental Trade-off](#the-fundamental-trade-off)
3. [Chunking Strategies Explained](#chunking-strategies-explained)
4. [Mathematical Analysis](#mathematical-analysis)
5. [Empirical Guidelines](#empirical-guidelines)
6. [Implementation Considerations](#implementation-considerations)

---

## Why Chunking Matters

### The Context Window Problem

Large Language Models have **fixed context windows**:
- GPT-3.5: 4,096 tokens (~3,000 words)
- GPT-4: 8,192-32,768 tokens
- Mistral-7B: 8,192 tokens
- Claude 2: 100,000 tokens

**Problem**: Most documents exceed these limits.

**Example**: A 50-page technical manual = ~25,000 words = ~33,000 tokens

You **cannot** feed the entire document to the LLM. You must:
1. Split the document into chunks
2. Retrieve only relevant chunks
3. Provide those chunks as context

### Why Not Just Retrieve Paragraphs?

Consider this scenario:

**Document**: "Python Tutorial - Chapter 5: Functions"
```
Functions are reusable blocks of code. They are defined using the def keyword.

def greet(name):
    return f"Hello, {name}"

The above function takes a parameter called name.
```

**Query**: "How do I define a function in Python?"

If you split by paragraphs:
- Chunk 1: "Functions are reusable blocks of code. They are defined using the def keyword."
- Chunk 2: "def greet(name): return f'Hello, {name}'"
- Chunk 3: "The above function takes a parameter called name."

**Problem**: None of these chunks alone fully answers the question!

**Solution**: Chunk size must be carefully chosen to preserve **semantic coherence**.

---

## The Fundamental Trade-off

Chunking involves balancing **three competing objectives**:

### 1. Semantic Coherence
**Definition**: Each chunk should represent a complete, self-contained idea.

**Why it matters**:
- Incomplete ideas confuse the LLM
- Context is lost if ideas are split mid-sentence
- Answer quality degrades with fragmented information

**Example**:
```
BAD CHUNK (mid-sentence split):
"The Pythagorean theorem states that in a right triangle, the square of"

GOOD CHUNK:
"The Pythagorean theorem states that in a right triangle, the square of
the hypotenuse equals the sum of squares of the other two sides: a² + b² = c²"
```

### 2. Retrieval Precision
**Definition**: How well can you retrieve the exact information needed?

**Smaller chunks** → Higher precision
- Easier to match specific queries
- Less irrelevant information
- Better for factual questions

**Larger chunks** → Lower precision
- More noise in retrieved context
- Harder to rank relevance
- LLM may get distracted

### 3. Context Richness
**Definition**: How much surrounding information is available?

**Larger chunks** → More context
- Better understanding of relationships
- Can answer complex, multi-part questions
- Preserves narrative flow

**Smaller chunks** → Less context
- May lose important background
- Struggles with "why" and "how" questions
- Isolated facts without explanation

### The Optimal Balance

There is **no universal best chunk size**. It depends on:

| Use Case | Optimal Chunk Size | Reasoning |
|----------|-------------------|-----------|
| **FAQ / Q&A** | 100-200 tokens | Each Q&A is self-contained |
| **Technical docs** | 300-500 tokens | Balances code + explanation |
| **Narrative text** | 500-1000 tokens | Preserves story/argument flow |
| **Academic papers** | 200-400 tokens | Each paragraph = one idea |
| **Legal documents** | 100-300 tokens | Precision critical, clauses independent |

---

## Chunking Strategies Explained

### Strategy 1: Fixed-Size Chunking

#### How It Works
Split text into chunks of exactly N tokens (or characters).

```
Text: "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
Chunk size: 5
Overlap: 0

Result:
Chunk 1: "ABCDE"
Chunk 2: "FGHIJ"
Chunk 3: "KLMNO"
Chunk 4: "PQRST"
Chunk 5: "UVWXY"
Chunk 6: "Z"
```

#### Mathematical Model

Given:
- `D` = document length (tokens)
- `C` = chunk size (tokens)
- `O` = overlap (tokens)

Number of chunks:
```
N = ⌈(D - O) / (C - O)⌉
```

#### Advantages
✅ **Simple**: Easy to implement and understand
✅ **Predictable**: Always know chunk count and size
✅ **Fast**: O(n) complexity, no parsing needed
✅ **Consistent**: All chunks roughly same size → balanced retrieval

#### Disadvantages
❌ **Ignores semantics**: Can split mid-sentence, mid-paragraph, mid-thought
❌ **Context loss**: May separate related information
❌ **Arbitrary boundaries**: No respect for document structure

#### When to Use
- Quick prototyping
- Unstructured text (social media, chat logs)
- When speed matters more than quality
- As a baseline for comparison

#### Implementation Detail: Overlap

**Why overlap?**

Without overlap:
```
Chunk 1: "The capital of France is"
Chunk 2: "Paris, known for its"
```
Query: "What is the capital of France?" → May miss Chunk 2!

With overlap (50 tokens):
```
Chunk 1: "The capital of France is Paris, known for its"
Chunk 2: "Paris, known for its Eiffel Tower"
```
Query: "What is the capital of France?" → Matches Chunk 1 ✓

**Optimal overlap**: 10-20% of chunk size
- Too small: Still lose context
- Too large: Redundancy, slower retrieval, index bloat

---

### Strategy 2: Semantic Chunking

#### How It Works
Split at **natural boundaries**: sentence endings, paragraph breaks, section headers.

```python
# Hierarchy of split points (priority order):
1. Section headers (##, ###)
2. Paragraph breaks (\n\n)
3. Sentence endings (. ! ?)
4. Clause boundaries (, ; :)
5. Token limit (fallback)
```

#### Algorithm Pseudocode
```
function semantic_chunk(text, target_size):
    chunks = []
    current_chunk = ""

    for paragraph in text.split("\n\n"):
        if len(current_chunk) + len(paragraph) < target_size:
            current_chunk += paragraph
        else:
            if len(current_chunk) > 0:
                chunks.append(current_chunk)
            current_chunk = paragraph

    if len(current_chunk) > 0:
        chunks.append(current_chunk)

    return chunks
```

#### Advantages
✅ **Semantic coherence**: Respects natural thought boundaries
✅ **Better retrieval**: Each chunk = one complete idea
✅ **Human-readable**: Chunks make sense when read
✅ **Context preserved**: Related sentences stay together

#### Disadvantages
❌ **Variable size**: Chunks range from 50 to 2000 tokens
❌ **Complexity**: Requires sentence detection, paragraph parsing
❌ **Edge cases**: What if one paragraph is 5000 tokens?
❌ **Language-dependent**: Sentence detection varies by language

#### When to Use
- Structured documents (articles, papers, books)
- When answer quality is critical
- Technical documentation
- Educational content

#### Advanced Variant: Heading-Aware Chunking

For documents with clear structure (Markdown, HTML):

```markdown
# Chapter 1: Introduction
This is the intro...

## Section 1.1: Background
Background information...

## Section 1.2: Motivation
Why this matters...
```

Strategy:
1. Keep headings with their content
2. Don't split sections unless absolutely necessary
3. Preserve hierarchical context

**Result**: Each chunk includes heading → better retrieval context

---

### Strategy 3: Sliding Window Chunking

#### How It Works
Create overlapping windows that slide across the text.

```
Text: "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
Window size: 10
Stride: 5 (moves 5 positions each time)

Windows:
[ABCDEFGHIJ]
     [FGHIJKLMNO]
          [KLMNOPQRSTU]
               [PQRSTUVWXY]
                    [UVWXYZ]
```

#### Mathematical Model

Given:
- `D` = document length
- `W` = window size
- `S` = stride (step size)

Number of windows:
```
N = ⌈(D - W) / S⌉ + 1
```

Overlap between consecutive windows:
```
Overlap = W - S
```

#### Example with Numbers

Document: 1000 tokens
Window: 200 tokens
Stride: 150 tokens

```
Window 1: tokens [0-200]     (overlap with next: 50 tokens)
Window 2: tokens [150-350]   (overlap with next: 50 tokens)
Window 3: tokens [300-500]
Window 4: tokens [450-650]
Window 5: tokens [600-800]
Window 6: tokens [750-950]
Window 7: tokens [900-1000]  (shorter final window)

Total windows: 7
Average overlap: 50 tokens (25%)
```

#### Advantages
✅ **No information loss**: Every part of document in multiple chunks
✅ **Robust retrieval**: If query matches boundary, still captured
✅ **Flexibility**: Tune window size and stride independently
✅ **Good for dense information**: Ensures no critical sentence missed

#### Disadvantages
❌ **High redundancy**: Same content in multiple chunks
❌ **Storage overhead**: More chunks → larger index
❌ **Slower retrieval**: More candidates to rank
❌ **Deduplication needed**: May retrieve same info multiple times

#### When to Use
- Critical applications (medical, legal) where missing information is costly
- Dense technical documents where every sentence matters
- When retrieval recall is more important than precision
- Documents with unpredictable structure

#### Optimization: Adaptive Stride

Instead of fixed stride, adjust based on content:

```python
def adaptive_stride(window_text, base_stride):
    # If window ends mid-sentence, extend to next sentence
    if not ends_with_sentence(window_text):
        stride = base_stride + distance_to_next_sentence(window_text)
    else:
        stride = base_stride
    return stride
```

This ensures windows end at natural boundaries while maintaining overlap benefits.

---

## Mathematical Analysis

### Information Loss Quantification

**Definition**: How much information is inaccessible due to chunking?

#### Metric 1: Boundary Fragmentation Rate (BFR)

```
BFR = (Number of sentences split across chunks) / (Total sentences)
```

**Example**:
- Document: 100 sentences
- Fixed chunking splits 15 sentences across boundaries
- BFR = 15/100 = 0.15 (15% fragmentation)

**Goal**: BFR < 0.05 (less than 5% sentences split)

#### Metric 2: Context Coverage (CC)

```
CC = (Tokens in chunks with full context) / (Total tokens)
```

**Full context** = sentence + previous sentence + next sentence all in same chunk

**Example**:
- Document: 1000 tokens
- 800 tokens have full context
- CC = 800/1000 = 0.80 (80% coverage)

**Goal**: CC > 0.90

### Retrieval Efficiency Analysis

#### Precision Impact

Smaller chunks → Higher precision → Better top-K accuracy

**Mathematical model**:
```
P(relevant | retrieved) ∝ 1 / chunk_size

Intuition: Smaller chunks have less irrelevant content
```

**Empirical data** (from research papers):
- 100-token chunks: ~65% precision@5
- 500-token chunks: ~50% precision@5
- 1000-token chunks: ~35% precision@5

#### Recall Impact

Larger chunks → Higher recall → More comprehensive answers

**Why?**: Related information stays together, less fragmentation

**Trade-off curve**:
```
Precision and Recall vs Chunk Size

Precision |     ╲
          |      ╲
          |       ╲___
          |           ╲____
          |________________╲____
          |________________________
          0   200  400  600  800  1000

Recall    |                    ____
          |               ____╱
          |          ____╱
          |     ____╱
          |____╱
          |________________________
          0   200  400  600  800  1000

Optimal = intersection point ≈ 300-500 tokens
```

### Token Overlap Efficiency

**Question**: How much overlap is optimal?

**Analysis**:

Let:
- `C` = chunk size
- `O` = overlap
- `N` = number of chunks
- `D` = document size

Total storage:
```
Storage = N × C
        = ⌈D / (C - O)⌉ × C
```

Storage overhead:
```
Overhead = (Storage - D) / D
         = (N × C - D) / D
```

**Example**:
- Document: 10,000 tokens
- Chunk: 500 tokens
- Overlap: 100 tokens (20%)

```
N = ⌈10000 / 400⌉ = 25 chunks
Storage = 25 × 500 = 12,500 tokens
Overhead = (12500 - 10000) / 10000 = 0.25 (25% more storage)
```

**Optimal overlap** (from empirical studies):
- **10-15%**: Good balance for most use cases
- **20-25%**: Dense technical content
- **5-10%**: Narrative text with clear structure
- **0%**: Only if using semantic chunking with good boundaries

---

## Empirical Guidelines

### By Document Type

#### 1. Technical Documentation (API docs, manuals)
**Recommended**: Semantic chunking, 300-500 tokens
**Why**:
- Code snippets need surrounding explanation
- Each function/API = one semantic unit
- Headings provide natural boundaries

**Example chunk**:
```
## Authentication

To authenticate API requests, include your API key in the header:

headers = {"Authorization": f"Bearer {api_key}"}
response = requests.get(url, headers=headers)

The API key can be found in your dashboard under Settings > API Keys.
```

#### 2. Academic Papers
**Recommended**: Semantic chunking, 200-400 tokens
**Why**:
- Each paragraph typically = one claim/idea
- References need context
- Figures/tables need captions

#### 3. Books / Long-form Content
**Recommended**: Sliding window, 500-1000 tokens, 20% overlap
**Why**:
- Narrative flow critical
- Arguments build over multiple paragraphs
- Character development / plot requires context

#### 4. Chat Logs / Social Media
**Recommended**: Fixed-size, 100-200 tokens
**Why**:
- Already fragmented (short messages)
- No semantic structure to preserve
- Speed matters more than quality

#### 5. Legal Documents
**Recommended**: Semantic chunking (clause-level), 150-300 tokens
**Why**:
- Each clause is independent
- Precision absolutely critical
- Exact wording matters

### By Query Type

#### Factual Queries ("What is X?")
**Optimal**: Smaller chunks (100-300 tokens)
**Reason**: Answer likely in single sentence/paragraph

#### Explanatory Queries ("How does X work?")
**Optimal**: Medium chunks (300-600 tokens)
**Reason**: Need context + explanation

#### Comparative Queries ("Difference between X and Y?")
**Optimal**: Larger chunks (500-1000 tokens)
**Reason**: Need both concepts in same context

---

## Implementation Considerations

### Performance Optimization

#### 1. Caching Tokenization
Tokenization is expensive. Cache token counts:

```python
# BAD: Tokenize every time
def chunk_text(text, chunk_size):
    tokens = tokenizer.encode(text)  # Slow!
    ...

# GOOD: Cache and reuse
token_cache = {}

def get_tokens(text):
    if text not in token_cache:
        token_cache[text] = tokenizer.encode(text)
    return token_cache[text]
```

#### 2. Batch Processing
Don't tokenize one sentence at a time:

```python
# BAD: Sequential
for sentence in sentences:
    tokens = tokenizer.encode(sentence)

# GOOD: Batch
all_tokens = tokenizer.batch_encode(sentences)
```

**Speedup**: 5-10x faster for large documents

### Quality Metrics

#### Evaluating Chunking Quality

1. **Manual Inspection**: Read random chunks, check coherence
2. **Automated BFR**: Measure sentence fragmentation
3. **Retrieval Test**: Run test queries, measure Recall@K
4. **End-to-end**: Evaluate final RAG answer quality

#### A/B Testing Chunking Strategies

```python
strategies = [
    ("fixed_256", FixedChunker(256, overlap=50)),
    ("fixed_512", FixedChunker(512, overlap=100)),
    ("semantic_500", SemanticChunker(target_size=500)),
    ("sliding_400_300", SlidingWindowChunker(400, stride=300))
]

for name, chunker in strategies:
    chunks = chunker.chunk(document)
    index = build_index(chunks)
    results = evaluate_retrieval(index, test_queries)
    print(f"{name}: Recall@5={results['recall']:.3f}, nDCG={results['ndcg']:.3f}")
```

---

## Key Takeaways

1. **No universal best**: Chunk size depends on document type, query type, and use case

2. **Fixed chunking**: Fast, simple, good baseline, but ignores semantics

3. **Semantic chunking**: Best for structured content, preserves meaning, but variable sizes

4. **Sliding window**: Maximizes recall, prevents information loss, but higher overhead

5. **Overlap is crucial**: 10-20% overlap dramatically improves retrieval quality

6. **Measure, don't guess**: Use BFR, Context Coverage, and end-to-end metrics

7. **Start simple, iterate**: Begin with fixed chunking, then optimize based on results

---

## Next Steps

1. Implement all three strategies in `src/chunking.py`
2. Run experiments comparing them on your dataset
3. Measure BFR and Context Coverage
4. Visualize chunk size distributions
5. A/B test with actual queries

Continue to **[02_retrieval.md](02_retrieval.md)** to learn about dense and sparse search →
