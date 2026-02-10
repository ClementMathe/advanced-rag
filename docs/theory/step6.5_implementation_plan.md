# Step 6.5 Implementation Plan

## Overview

This document provides a **step-by-step checklist** for implementing the agentic RAG system described in `06.5_agentic_theory.md`.

**Estimated time**: 4-6 hours (with Claude Code assistance)

**Difficulty**: ⭐⭐⭐ Intermediate (requires LangGraph understanding)

---

## Prerequisites

### 1. Environment Setup

```bash
# Install LangGraph + dependencies
pip install langgraph langchain langchain-community --break-system-packages
```



## Phase 1: Core Components (Priority 1)

**Goal**: Implement grading and basic graph without self-correction

**Estimated time**: 2 hours

---

### Task 1.1: Create `src/graders.py`

**What to implement**:

```python
"""
Document and query grading utilities for agentic RAG.

Functions:
- grade_document_relevance(): Binary LLM grading (yes/no)
- batch_grade_documents(): Efficient batch grading
- rephrase_query(): LLM-based query rewriting
"""

from typing import List, Dict
from src.generator import LLMGenerator

class DocumentGrader:
    """Grades document relevance using LLM."""

    def __init__(self, generator: LLMGenerator):
        self.generator = generator
        self.grading_prompt = """..."""  # See theory doc

    def grade_single(self, query: str, document: str) -> bool:
        """Grade one document (returns True/False)."""
        ...

    def grade_batch(self, query: str, documents: List[str]) -> List[bool]:
        """Grade multiple documents in one LLM call (efficient)."""
        ...

class QueryRewriter:
    """Rewrites queries using LLM."""

    def __init__(self, generator: LLMGenerator):
        self.generator = generator
        self.rewrite_prompt = """..."""  # See theory doc

    def rewrite(
        self,
        query: str,
        num_retrieved: int,
        num_relevant: int
    ) -> str:
        """Generate improved query."""
        ...
```

**Checklist**:
- [ ] Create `src/graders.py`
- [ ] Implement `DocumentGrader` class
- [ ] Implement `QueryRewriter` class
- [ ] Add comprehensive docstrings
- [ ] Use existing `LLMGenerator` (no new model loading)

**Test**:
```python
from src.graders import DocumentGrader, QueryRewriter
from src.generator import LLMGenerator

generator = LLMGenerator("Qwen/Qwen2.5-3B-Instruct", load_in_4bit=True)
grader = DocumentGrader(generator)

# Test grading
is_relevant = grader.grade_single(
    "When was Beyoncé born?",
    "Beyoncé was born on September 4, 1981."
)
assert is_relevant == True

# Test batch grading
grades = grader.grade_batch(
    "What is X?",
    ["Doc about X", "Doc about Y", "Doc about X again"]
)
assert len(grades) == 3
```

**Claude Code prompt**:
```
Create src/graders.py with DocumentGrader and QueryRewriter classes.
Use the LLMGenerator from src/generator.py.
Implement batch grading for efficiency (all docs in one prompt).
See prompt templates in docs/06.5_agentic_theory.md section "Prompt Engineering".
```

---

### Task 1.2: Create test file `tests/test_graders.py`

**What to test**:

```python
import pytest
from src.graders import DocumentGrader, QueryRewriter
from unittest.mock import Mock

class TestDocumentGrader:
    @pytest.fixture
    def mock_generator(self):
        """Mock LLMGenerator to avoid loading models in tests."""
        generator = Mock()
        generator.model = Mock()
        generator.tokenizer = Mock()
        generator.device = "cpu"
        return generator

    def test_grade_single_relevant(self, mock_generator):
        """Test grading a clearly relevant document."""
        # Mock LLM to return "yes"
        mock_generator.tokenizer.encode.return_value = [123]
        mock_generator.tokenizer.decode.return_value = "yes"

        grader = DocumentGrader(mock_generator)
        is_relevant = grader.grade_single("Query", "Relevant doc")

        assert is_relevant == True

    def test_grade_batch(self, mock_generator):
        """Test batch grading returns list of bools."""
        mock_generator.tokenizer.decode.return_value = '{"grades": [true, false, true]}'

        grader = DocumentGrader(mock_generator)
        grades = grader.grade_batch("Query", ["Doc1", "Doc2", "Doc3"])

        assert grades == [True, False, True]
```

**Checklist**:
- [ ] Create `tests/test_graders.py`
- [ ] Test `grade_single()` with relevant doc
- [ ] Test `grade_single()` with irrelevant doc
- [ ] Test `grade_batch()` returns correct length
- [ ] Test query rewriting changes query
- [ ] Use mocks (no real LLM calls in tests)

**Run**: `pytest tests/test_graders.py -v`

**Claude Code prompt**:
```
Create tests/test_graders.py with comprehensive unit tests.
Use pytest and Mock to avoid loading models.
Test both single and batch grading.
Test query rewriting.
Aim for >80% coverage.
```

---

### Task 1.3: Create `src/agentic_pipeline.py` (MVP version)

**What to implement** (simplified, no retry yet):

```python
"""
Agentic RAG pipeline using LangGraph.

MVP: retrieve → grade → generate (no self-correction yet)
"""

from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END

from src.retriever import HybridRetriever
from src.reranker import CrossEncoderReranker
from src.generator import LLMGenerator
from src.graders import DocumentGrader

# State definition
class RAGState(TypedDict):
    query: str
    documents: List[Dict]
    graded_documents: List[Dict]
    generation: str
    intermediate_steps: List[str]

class AgenticRAGPipeline:
    """LangGraph-based agentic RAG pipeline."""

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        reranker: CrossEncoderReranker,
        generator: LLMGenerator,
        grader: DocumentGrader,
    ):
        self.retriever = hybrid_retriever
        self.reranker = reranker
        self.generator = generator
        self.grader = grader

        # Build graph
        self.app = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Construct the LangGraph workflow."""
        workflow = StateGraph(RAGState)

        # Add nodes
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("grade_documents", self._grade_documents_node)
        workflow.add_node("generate", self._generate_node)

        # Add edges (linear for MVP)
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_edge("grade_documents", "generate")
        workflow.add_edge("generate", END)

        return workflow.compile()

    def _retrieve_node(self, state: RAGState) -> RAGState:
        """Retrieve and rerank documents."""
        ...

    def _grade_documents_node(self, state: RAGState) -> RAGState:
        """Grade documents for relevance."""
        ...

    def _generate_node(self, state: RAGState) -> RAGState:
        """Generate answer from graded documents."""
        ...

    def query(self, query: str) -> Dict:
        """Run the pipeline on a query."""
        initial_state = {
            "query": query,
            "documents": [],
            "graded_documents": [],
            "generation": "",
            "intermediate_steps": []
        }

        final_state = self.app.invoke(initial_state)

        return {
            "answer": final_state["generation"],
            "steps": final_state["intermediate_steps"],
            "num_docs_graded": len(final_state["graded_documents"])
        }
```

**Checklist**:
- [ ] Create `src/agentic_pipeline.py`
- [ ] Define `RAGState` TypedDict
- [ ] Implement `AgenticRAGPipeline` class
- [ ] Implement 3 nodes (retrieve, grade, generate)
- [ ] Build linear graph (no conditionals yet)
- [ ] Test on 1 query manually

**Test** (manual):
```python
from src.agentic_pipeline import AgenticRAGPipeline
# ... load components
pipeline = AgenticRAGPipeline(retriever, reranker, generator, grader)
result = pipeline.query("When did Beyoncé become famous?")
print(result['answer'])
print(result['steps'])  # Should show: retrieve → grade → generate
```

**Claude Code prompt**:
```
Create src/agentic_pipeline.py with AgenticRAGPipeline class.
Implement a basic LangGraph workflow: retrieve → grade → generate.
Use existing components (HybridRetriever, CrossEncoderReranker, LLMGenerator).
Follow the template in LANGGRAPH_TEMPLATE.md.
Add logging to intermediate_steps at each node.
```

---

### Task 1.4: Integration test

**Goal**: Verify end-to-end on 5 test queries

**Script**: `scripts/test_agentic_mvp.py`

```python
"""Quick integration test for agentic pipeline MVP."""

from src.agentic_pipeline import AgenticRAGPipeline
from src.embeddings import EmbeddingModel, FAISSIndex
from src.retriever import HybridRetriever, DenseRetriever, BM25Retriever
from src.reranker import CrossEncoderReranker
from src.generator import LLMGenerator
from src.graders import DocumentGrader
from src.chunking import Chunk
from loguru import logger

def main():
    # Load components
    logger.info("Loading models...")

    embed_model = EmbeddingModel("BAAI/bge-large-en-v1.5", device="cuda")
    faiss_index = FAISSIndex.load("index/squad")

    dense_retriever = DenseRetriever(faiss_index, embed_model)

    # Rebuild BM25
    chunks = [Chunk(...) for meta in faiss_index.chunk_metadata]
    bm25_retriever = BM25Retriever()
    bm25_retriever.index(chunks)

    hybrid_retriever = HybridRetriever(
        dense_retriever, bm25_retriever,
        k_rrf=60, dense_weight=0.9, sparse_weight=0.1
    )

    reranker = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L6-v2")
    generator = LLMGenerator("Qwen/Qwen2.5-3B-Instruct", load_in_4bit=True, max_new_tokens=80)
    grader = DocumentGrader(generator)

    # Create pipeline
    pipeline = AgenticRAGPipeline(hybrid_retriever, reranker, generator, grader)

    # Test queries
    test_queries = [
        "When did Beyoncé become famous?",
        "What city did she grow up in?",
        "How many Grammy awards for her debut?",
        "Who managed Destiny's Child?",
        "What was her first solo album?",
    ]

    for query in test_queries:
        logger.info(f"\n{'='*60}")
        logger.info(f"Query: {query}")

        result = pipeline.query(query)

        logger.info(f"Answer: {result['answer']}")
        logger.info(f"Graded docs: {result['num_docs_graded']}/10")
        logger.info(f"Steps: {' → '.join(result['steps'])}")

if __name__ == "__main__":
    main()
```

**Checklist**:
- [ ] Create `scripts/test_agentic_mvp.py`
- [ ] Run on 5 queries
- [ ] Verify all return answers
- [ ] Check logs show: retrieve → grade → generate
- [ ] Verify graded_documents < documents (some filtered)

**Run**: `python scripts/test_agentic_mvp.py`

**Expected output**:
```
Query: When did Beyoncé become famous?
Answer: Beyoncé became famous in the late 1990s...
Graded docs: 8/10
Steps: Retrieved 10 docs → Graded: 8/10 relevant → Generated answer
```

### Task 1.5: tests/test_agentic_pipeline.py (baseline)
**Goal**: Unit tests for AgenticRAGPipeline MVP (before Phase 2)
**Implementation**:
- Test RAGState initialization
- Test individual nodes with mocks (retrieve, grade, generate)
- Test graph compilation
- Verify intermediate_steps accumulation
**Files to create**:
- tests/test_agentic_pipeline.py (~150 lines)
**Coverage target**: >80% on src/agentic_pipeline.py

---

## Phase 2: Self-Correction (Priority 2)

**Goal**: Add query rewriting and retry logic

**Estimated time**: 1.5 hours

---

### Task 2.1: Expand `RAGState`

**Add fields**:
```python
class RAGState(TypedDict):
    query: str
    rewritten_query: str          # NEW
    query_history: List[str]       # NEW
    documents: List[Dict]
    graded_documents: List[Dict]
    document_grades: List[bool]    # NEW (for transparency)
    generation: str
    retry_count: int               # NEW
    intermediate_steps: List[str]
```

**Checklist**:
- [ ] Update `RAGState` in `src/agentic_pipeline.py`
- [ ] Initialize new fields in `query()` method

---

### Task 2.2: Add `rewrite_query_node`

**Implement**:
```python
def _rewrite_query_node(self, state: RAGState) -> RAGState:
    """Rewrite query to improve retrieval."""
    current_query = state.get("rewritten_query") or state["query"]
    num_relevant = len(state.get("graded_documents", []))
    num_total = len(state.get("documents", []))

    # Use QueryRewriter
    new_query = self.query_rewriter.rewrite(
        current_query, num_total, num_relevant
    )

    # Update state
    history = state.get("query_history", [])
    history.append(new_query)

    steps = state.get("intermediate_steps", [])
    steps.append(f"Rewrote: '{current_query}' → '{new_query}'")

    return {
        **state,
        "rewritten_query": new_query,
        "query_history": history,
        "retry_count": state.get("retry_count", 0) + 1,
        "intermediate_steps": steps
    }
```

**Checklist**:
- [ ] Add `QueryRewriter` to `__init__`
- [ ] Implement `_rewrite_query_node`
- [ ] Log rewrite in `intermediate_steps`

---

### Task 2.3: Add conditional routing

**Implement `_decide_to_generate`**:
```python
def _decide_to_generate(self, state: RAGState) -> str:
    """Route to generate or rewrite based on doc quality."""
    graded_docs = state.get("graded_documents", [])
    retry_count = state.get("retry_count", 0)
    query_history = state.get("query_history", [])

    # Rule 1: Enough relevant docs
    if len(graded_docs) >= 3:
        return "generate"

    # Rule 2: Max retries
    if retry_count >= 3:
        return "generate"

    # Rule 3: Query unchanged (loop detection)
    if len(query_history) >= 2 and query_history[-1] == query_history[-2]:
        return "generate"

    # Otherwise: retry
    return "rewrite"
```

**Update graph**:
```python
def _build_graph(self):
    workflow = StateGraph(RAGState)

    # Add nodes
    workflow.add_node("retrieve", self._retrieve_node)
    workflow.add_node("grade_documents", self._grade_documents_node)
    workflow.add_node("generate", self._generate_node)
    workflow.add_node("rewrite_query", self._rewrite_query_node)  # NEW

    # Edges
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")

    # Conditional edge
    workflow.add_conditional_edges(
        "grade_documents",
        self._decide_to_generate,
        {
            "generate": "generate",
            "rewrite": "rewrite_query"
        }
    )

    workflow.add_edge("rewrite_query", "retrieve")  # Loop back
    workflow.add_edge("generate", END)

    return workflow.compile()
```

**Checklist**:
- [ ] Implement `_decide_to_generate`
- [ ] Add `rewrite_query` node to graph
- [ ] Add conditional edge from `grade_documents`
- [ ] Add loop edge from `rewrite_query` to `retrieve`

---

### Task 2.4: Extend tests/test_agentic_pipeline.py
**Goal**: Add tests for Phase 2 features (retry logic, routing)
**Implementation**:
- Test decide_to_generate() routing logic
- Test retry_count increment
- Test query_history loop detection
- Test max retry limit (3)
**Files to modify**:
- tests/test_agentic_pipeline.py (add ~100 lines)

```python
def test_retry_on_poor_retrieval():
    """Test that poor retrieval triggers query rewrite."""
    # Mock retriever to return irrelevant docs
    # Mock grader to reject all docs
    # Assert: retry_count > 0 and rewritten_query != original
    ...

def test_max_retries_fallback():
    """Test that max retries triggers generation anyway."""
    # Set retry_count = 3 in initial state
    # Assert: Still returns an answer
    ...
```

**Run**: `pytest tests/test_agentic_pipeline.py -v`

---

## Phase 3: Evaluation (Priority 3)

**Goal**: Compare agentic vs linear on 100 queries

**Estimated time**: 1.5 hours

---

### Task 3.1: Create `scripts/evaluate_agentic.py`

**Based on**: `scripts/ablation_study.py` structure

**What to implement**:

```python
"""
Evaluate agentic RAG pipeline vs linear baseline.

Runs both pipelines on same 100 queries, compares metrics.
"""

import json
from pathlib import Path
from loguru import logger
from tqdm import tqdm

from src.agentic_pipeline import AgenticRAGPipeline
from src.pipeline import RAGPipeline  # Linear baseline
# ... load components

def evaluate_agentic(pipeline, queries):
    """Evaluate agentic pipeline."""
    results = []

    for query_data in tqdm(queries, desc="Agentic"):
        question = query_data['query']
        ground_truth = query_data.get('answer', '')

        # Run agentic pipeline
        result = pipeline.query(question)

        # Compute metrics
        metrics = compute_all(
            result['answer'],
            ground_truth,
            result.get('graded_documents', [])
        )

        results.append({
            'query': question,
            'answer': result['answer'],
            'metrics': metrics,
            'retry_count': result.get('retry_count', 0),
            'steps': result.get('steps', [])
        })

    return results

def evaluate_linear(pipeline, queries):
    """Evaluate linear pipeline (Step 6 baseline)."""
    ...

def compare_results(agentic_results, linear_results):
    """Generate comparison table."""
    comparison = {
        'linear': aggregate(linear_results),
        'agentic': aggregate(agentic_results),
        'delta': {}
    }

    for metric in ['faithfulness', 'f1', 'recall']:
        linear_val = comparison['linear'][metric]['mean']
        agentic_val = comparison['agentic'][metric]['mean']
        comparison['delta'][metric] = agentic_val - linear_val

    return comparison

def main():
    # Load queries
    with open('data/squad/queries_500_with_answers.json') as f:
        all_queries = json.load(f)

    queries = all_queries[:100]

    # Load components
    # ...

    # Create pipelines
    linear_pipeline = RAGPipeline(...)
    agentic_pipeline = AgenticRAGPipeline(...)

    # Evaluate both
    logger.info("Evaluating linear pipeline...")
    linear_results = evaluate_linear(linear_pipeline, queries)

    logger.info("Evaluating agentic pipeline...")
    agentic_results = evaluate_agentic(agentic_pipeline, queries)

    # Compare
    comparison = compare_results(agentic_results, linear_results)

    # Save
    output_dir = Path("outputs/agentic_eval")
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / 'comparison.json', 'w') as f:
        json.dump(comparison, f, indent=2)

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("AGENTIC vs LINEAR COMPARISON")
    logger.info("="*60)
    for metric, delta in comparison['delta'].items():
        sign = "+" if delta > 0 else ""
        logger.info(f"{metric}: {sign}{delta:.2%}")

if __name__ == "__main__":
    main()
```

**Checklist**:
- [ ] Create `scripts/evaluate_agentic.py`
- [ ] Implement evaluation for both pipelines
- [ ] Use same metrics as Step 6 (EM, F1, ROUGE-L, Faithfulness)
- [ ] Add agentic-specific metrics (retry_rate, success_rate)
- [ ] Save individual results + comparison

**Run**: `python scripts/evaluate_agentic.py`

**Expected output**:
```
AGENTIC vs LINEAR COMPARISON
============================================================
faithfulness: +5.2%
f1: +7.8%
recall@5: +2.1%
retry_rate: 18%
avg_retries: 0.24
```

**Note** To see the correct value of parameters refer to docs/experiments/step6_generation_analysis.md

---

### Task 3.2: Generate comparison visualizations

**Create**: `scripts/plot_agentic_comparison.py`

**What to plot**:

1. **Bar chart**: Metrics comparison (Linear vs Agentic)
2. **Histogram**: Retry count distribution
3. **Scatter**: F1 improvement vs retry count

**Use matplotlib**, follow style from `ablation_study.py`

**Checklist**:
- [ ] Create plotting script
- [ ] Generate 3 graphs
- [ ] Save to `outputs/agentic_eval/`

---

## Phase 4: Documentation (Priority 4)

**Goal**: Write analysis document

**Estimated time**: 1 hour

---

### Task 4.1: Write `docs/experiments/06.5_agentic_analysis.md`

**Structure** (similar to `06_generation_analysis.md`):

```markdown
# Step 6.5: Agentic RAG - Experimental Results

## Overview
[Summary of approach and results]

## Results Summary
[Table: Linear vs Agentic metrics]

## Qualitative Analysis
[Examples where agentic helped]
[Examples where it didn't]

## Retry Analysis
[Distribution of retries]
[Success rate by retry count]

## Latency Analysis
[Breakdown by phase]

## Lessons Learned
[What worked, what didn't]

## Future Work
[Multi-hop, web search, etc.]
```

**Checklist**:
- [ ] Write comprehensive analysis doc
- [ ] Include graphs
- [ ] Document edge cases discovered
- [ ] Write lessons learned

---

### Task 4.2: Update main README

**Add section**:

```markdown
### Step 6.5: Agentic RAG ✅
- Self-correcting workflow with LangGraph
- Document grading + query rewriting
- Performance: 98% Recall@5, 85% Faithfulness
- **+6% faithfulness improvement over linear**
```

---

## Testing Strategy

### Unit Tests

**Files**:
- `tests/test_graders.py` - Document grading
- `tests/test_agentic.py` - Pipeline logic
- `tests/test_query_rewriter.py` - Query rewriting

**Coverage goal**: >80%

**Run**: `pytest tests/ -v --cov=src`

---

### Integration Tests

**File**: `scripts/test_agentic_mvp.py`

**Test scenarios**:
- Easy query (no retry needed)
- Medium query (1 retry)
- Hard query (max retries)
- Edge case (all docs irrelevant)

---

### Manual Testing

**Checklist**:
- [ ] Test on 5 easy queries → No retries
- [ ] Test on 5 ambiguous queries → 1-2 retries
- [ ] Test on 5 hard queries → Max retries
- [ ] Verify logs show correct flow
- [ ] Inspect graph visualization (Mermaid)

---

## Debugging Checklist

### Common Issues

**Issue 1**: Grading always returns True/False

**Debug**:
```python
# Print LLM response
logger.debug(f"LLM grading response: {response}")
```

**Fix**: Adjust prompt, add examples

---

**Issue 2**: Query rewriting returns same query

**Debug**:
```python
# Check if LLM understood task
logger.debug(f"Rewrite prompt: {prompt}")
logger.debug(f"LLM response: {response}")
```

**Fix**: Improve prompt clarity

---

**Issue 3**: Infinite loop (query → rewrite → same query)

**Debug**:
```python
# Check query_history
logger.debug(f"Query history: {state['query_history']}")
```

**Fix**: Loop detection already in `_decide_to_generate`

---

**Issue 4**: Out of memory

**Debug**:
```python
allocated = torch.cuda.memory_allocated() / 1e9
logger.info(f"VRAM: {allocated:.2f}GB")
```

**Fix**: Use `load_in_4bit=True` for all models

---

## Git Workflow

### Commits

Commit after each phase:

```bash
# Phase 1
git add src/graders.py tests/test_graders.py
git commit -m "feat(step6.5): Add document grading and query rewriting"

# Phase 2
git add src/agentic_pipeline.py
git commit -m "feat(step6.5): Implement LangGraph workflow with self-correction"

# Phase 3
git add scripts/evaluate_agentic.py outputs/agentic_eval/
git commit -m "feat(step6.5): Add evaluation and comparison with linear baseline"

# Phase 4
git add docs/
git commit -m "docs(step6.5): Add comprehensive analysis and theory documents"
```

### Final merge

```bash
git push origin step6.5-agentic-rag
# Create PR or merge
git checkout main
git merge step6.5-agentic-rag
git push origin main
```

---

## Success Checklist

### Code Quality

- [ ] All functions have docstrings
- [ ] Type hints on all functions
- [ ] No hardcoded paths
- [ ] Logging at INFO level for key steps
- [ ] Error handling for edge cases

### Tests

- [ ] Unit tests pass (`pytest tests/ -v`)
- [ ] Coverage >80%
- [ ] Integration test runs successfully
- [ ] Manual testing on diverse queries

### Documentation

- [ ] Theory document complete
- [ ] Analysis document with results
- [ ] README updated
- [ ] Code comments for complex logic

### Performance

- [ ] Faithfulness ≥ 84%
- [ ] Recall@5 ≥ 97%
- [ ] Latency p95 ≤ 40s
- [ ] Success rate ≥ 92%



---

**Ready to implement! Start with Phase 1, Task 1.1 (graders.py). Good luck! 🚀**
