"""
Unit tests for the agentic RAG pipeline.

Tests:
- RAGState initialization and structure
- Individual node behavior with mocked components
- Graph compilation and structure
- Intermediate steps accumulation across nodes
- query() method output format
- Fallback behavior when grading filters all documents
- Conditional routing (_decide_to_generate)
- Query rewriting node
- Retry loop integration

Note: Uses mocking to avoid loading models or LangGraph execution overhead.
"""

from unittest.mock import MagicMock

import pytest

from src.agentic_pipeline import AgenticRAGPipeline, RAGState


def create_mock_components():
    """Create mocked pipeline components."""
    retriever = MagicMock()
    reranker = MagicMock()
    generator = MagicMock()
    grader = MagicMock()
    query_rewriter = MagicMock()
    return retriever, reranker, generator, grader, query_rewriter


def make_doc(content: str, doc_id: str = "doc1", score: float = 0.9) -> dict:
    """Create a minimal document dict matching retriever output format."""
    return {
        "content": content,
        "doc_id": doc_id,
        "chunk_id": f"{doc_id}_0",
        "score": score,
        "chunk_index": 0,
        "metadata": {},
    }


def make_state(**overrides) -> RAGState:
    """Create a RAGState with defaults, overriding specified fields."""
    state = {
        "query": "test query",
        "rewritten_query": "",
        "query_history": [],
        "documents": [],
        "graded_documents": [],
        "document_grades": [],
        "generation": "",
        "retry_count": 0,
        "intermediate_steps": [],
    }
    state.update(overrides)
    return state


class TestRAGState:
    """Tests for RAGState TypedDict structure."""

    def test_state_accepts_all_fields(self):
        """RAGState should accept all required fields."""
        state = make_state()
        assert state["query"] == "test query"
        assert state["rewritten_query"] == ""
        assert state["query_history"] == []
        assert state["documents"] == []
        assert state["graded_documents"] == []
        assert state["document_grades"] == []
        assert state["generation"] == ""
        assert state["retry_count"] == 0
        assert state["intermediate_steps"] == []

    def test_state_with_populated_fields(self):
        """RAGState should hold populated data including retry fields."""
        docs = [make_doc("content 1"), make_doc("content 2")]
        state = make_state(
            query="When was Beyonce born?",
            rewritten_query="Beyonce birth year date",
            query_history=["Beyonce birth year date"],
            documents=docs,
            graded_documents=[docs[0]],
            document_grades=[True, False],
            generation="Beyonce was born in 1981.",
            retry_count=1,
            intermediate_steps=["Retrieved 2 docs", "Graded: 1/2 relevant"],
        )
        assert len(state["documents"]) == 2
        assert len(state["graded_documents"]) == 1
        assert state["retry_count"] == 1
        assert state["rewritten_query"] == "Beyonce birth year date"
        assert len(state["query_history"]) == 1


class TestPipelineInit:
    """Tests for AgenticRAGPipeline initialization and graph compilation."""

    def test_init_stores_components(self):
        """Pipeline should store all component references."""
        retriever, reranker, generator, grader, rewriter = create_mock_components()

        pipeline = AgenticRAGPipeline(
            hybrid_retriever=retriever,
            reranker=reranker,
            generator=generator,
            grader=grader,
            query_rewriter=rewriter,
        )

        assert pipeline.retriever is retriever
        assert pipeline.reranker is reranker
        assert pipeline.generator is generator
        assert pipeline.grader is grader
        assert pipeline.query_rewriter is rewriter

    def test_init_default_k_values(self):
        """Default k_retrieve=20, k_rerank=10."""
        retriever, reranker, generator, grader, rewriter = create_mock_components()

        pipeline = AgenticRAGPipeline(
            hybrid_retriever=retriever,
            reranker=reranker,
            generator=generator,
            grader=grader,
            query_rewriter=rewriter,
        )

        assert pipeline.k_retrieve == 20
        assert pipeline.k_rerank == 5

    def test_init_default_retry_values(self):
        """Default min_relevant=3, max_retries=3."""
        retriever, reranker, generator, grader, rewriter = create_mock_components()

        pipeline = AgenticRAGPipeline(
            hybrid_retriever=retriever,
            reranker=reranker,
            generator=generator,
            grader=grader,
            query_rewriter=rewriter,
        )

        assert pipeline.min_relevant == 3
        assert pipeline.max_retries == 3

    def test_init_custom_k_values(self):
        """Custom k values should be stored."""
        retriever, reranker, generator, grader, rewriter = create_mock_components()

        pipeline = AgenticRAGPipeline(
            hybrid_retriever=retriever,
            reranker=reranker,
            generator=generator,
            grader=grader,
            query_rewriter=rewriter,
            k_retrieve=30,
            k_rerank=5,
        )

        assert pipeline.k_retrieve == 30
        assert pipeline.k_rerank == 5

    def test_init_custom_retry_values(self):
        """Custom retry values should be stored."""
        retriever, reranker, generator, grader, rewriter = create_mock_components()

        pipeline = AgenticRAGPipeline(
            hybrid_retriever=retriever,
            reranker=reranker,
            generator=generator,
            grader=grader,
            query_rewriter=rewriter,
            min_relevant=5,
            max_retries=2,
        )

        assert pipeline.min_relevant == 5
        assert pipeline.max_retries == 2

    def test_graph_compiles(self):
        """_build_graph() should produce a compiled graph."""
        retriever, reranker, generator, grader, rewriter = create_mock_components()

        pipeline = AgenticRAGPipeline(
            hybrid_retriever=retriever,
            reranker=reranker,
            generator=generator,
            grader=grader,
            query_rewriter=rewriter,
        )

        assert pipeline.app is not None


class TestRetrieveNode:
    """Tests for _retrieve_node."""

    @pytest.fixture
    def pipeline(self):
        retriever, reranker, generator, grader, rewriter = create_mock_components()
        retriever.search.return_value = [
            make_doc("doc A", "a"),
            make_doc("doc B", "b"),
            make_doc("doc C", "c"),
        ]
        reranker.rerank.return_value = [
            make_doc("doc A", "a"),
            make_doc("doc B", "b"),
        ]
        return AgenticRAGPipeline(
            hybrid_retriever=retriever,
            reranker=reranker,
            generator=generator,
            grader=grader,
            query_rewriter=rewriter,
            k_retrieve=20,
            k_rerank=2,
        )

    def test_returns_reranked_documents(self, pipeline):
        """Node should return reranked documents."""
        state = make_state(query="test query")
        result = pipeline._retrieve_node(state)

        assert len(result["documents"]) == 2
        assert result["documents"][0]["doc_id"] == "a"

    def test_calls_retriever_with_correct_k(self, pipeline):
        """Retriever should be called with k_retrieve and k_retriever=50."""
        state = make_state(query="test query")
        pipeline._retrieve_node(state)

        pipeline.retriever.search.assert_called_once_with("test query", k=20, k_retriever=50)

    def test_uses_rewritten_query_on_retry(self, pipeline):
        """On retry, should use rewritten_query instead of original."""
        state = make_state(
            query="original query",
            rewritten_query="better query",
        )
        pipeline._retrieve_node(state)

        pipeline.retriever.search.assert_called_once_with("better query", k=20, k_retriever=50)

    def test_falls_back_to_original_when_no_rewrite(self, pipeline):
        """With empty rewritten_query, should use original query."""
        state = make_state(query="original query", rewritten_query="")
        pipeline._retrieve_node(state)

        pipeline.retriever.search.assert_called_once_with("original query", k=20, k_retriever=50)

    def test_calls_reranker_with_correct_top_k(self, pipeline):
        """Reranker should be called with k_rerank as top_k."""
        state = make_state(query="test query")
        pipeline._retrieve_node(state)

        pipeline.reranker.rerank.assert_called_once()
        call_kwargs = pipeline.reranker.rerank.call_args
        assert call_kwargs[1]["top_k"] == 2 or call_kwargs[0][2] == 2

    def test_logs_intermediate_step(self, pipeline):
        """Node should add a step describing retrieval."""
        state = make_state(query="test query")
        result = pipeline._retrieve_node(state)

        assert len(result["intermediate_steps"]) == 1
        assert "Retrieved" in result["intermediate_steps"][0]
        assert "2" in result["intermediate_steps"][0]


class TestGradeDocumentsNode:
    """Tests for _grade_documents_node."""

    @pytest.fixture
    def pipeline(self):
        retriever, reranker, generator, grader, rewriter = create_mock_components()
        grader.grade_batch.return_value = [True, False, True]
        return AgenticRAGPipeline(
            hybrid_retriever=retriever,
            reranker=reranker,
            generator=generator,
            grader=grader,
            query_rewriter=rewriter,
        )

    def test_filters_irrelevant_documents(self, pipeline):
        """Only documents graded True should remain."""
        docs = [make_doc("relevant A"), make_doc("irrelevant B"), make_doc("relevant C")]
        state = make_state(query="test query", documents=docs)

        result = pipeline._grade_documents_node(state)

        assert len(result["graded_documents"]) == 2
        assert result["graded_documents"][0]["content"] == "relevant A"
        assert result["graded_documents"][1]["content"] == "relevant C"

    def test_returns_document_grades(self, pipeline):
        """Node should return raw boolean grades for transparency."""
        docs = [make_doc("A"), make_doc("B"), make_doc("C")]
        state = make_state(query="test query", documents=docs)

        result = pipeline._grade_documents_node(state)

        assert result["document_grades"] == [True, False, True]

    def test_calls_grader_with_doc_contents(self, pipeline):
        """Grader should receive document content strings, not full dicts."""
        docs = [make_doc("content A"), make_doc("content B"), make_doc("content C")]
        state = make_state(query="test query", documents=docs)

        pipeline._grade_documents_node(state)

        pipeline.grader.grade_batch.assert_called_once_with(
            "test query", ["content A", "content B", "content C"]
        )

    def test_all_relevant(self, pipeline):
        """When all docs are relevant, all should pass through."""
        pipeline.grader.grade_batch.return_value = [True, True]
        docs = [make_doc("A"), make_doc("B")]
        state = make_state(query="q", documents=docs)

        result = pipeline._grade_documents_node(state)

        assert len(result["graded_documents"]) == 2

    def test_all_irrelevant(self, pipeline):
        """When all docs are irrelevant, graded list should be empty."""
        pipeline.grader.grade_batch.return_value = [False, False, False]
        docs = [make_doc("A"), make_doc("B"), make_doc("C")]
        state = make_state(query="q", documents=docs)

        result = pipeline._grade_documents_node(state)

        assert len(result["graded_documents"]) == 0

    def test_logs_intermediate_step(self, pipeline):
        """Node should log how many docs passed grading."""
        docs = [make_doc("A"), make_doc("B"), make_doc("C")]
        state = make_state(query="q", documents=docs)

        result = pipeline._grade_documents_node(state)

        assert len(result["intermediate_steps"]) == 1
        assert "2/3" in result["intermediate_steps"][0]


class TestDecideToGenerate:
    """Tests for _decide_to_generate routing logic."""

    @pytest.fixture
    def pipeline(self):
        retriever, reranker, generator, grader, rewriter = create_mock_components()
        return AgenticRAGPipeline(
            hybrid_retriever=retriever,
            reranker=reranker,
            generator=generator,
            grader=grader,
            query_rewriter=rewriter,
            min_relevant=3,
            max_retries=3,
        )

    def test_generate_when_enough_relevant(self, pipeline):
        """Should route to generate when >= min_relevant docs."""
        graded = [make_doc("A"), make_doc("B"), make_doc("C")]
        state = make_state(graded_documents=graded, retry_count=0)

        assert pipeline._decide_to_generate(state) == "generate"

    def test_generate_when_more_than_enough(self, pipeline):
        """Should route to generate when well above threshold."""
        graded = [make_doc(f"d{i}") for i in range(8)]
        state = make_state(graded_documents=graded, retry_count=0)

        assert pipeline._decide_to_generate(state) == "generate"

    def test_rewrite_when_too_few_relevant(self, pipeline):
        """Should route to rewrite when < min_relevant and retries left."""
        graded = [make_doc("A")]
        state = make_state(graded_documents=graded, retry_count=0)

        assert pipeline._decide_to_generate(state) == "rewrite"

    def test_rewrite_when_zero_relevant(self, pipeline):
        """Should route to rewrite when no docs are relevant."""
        state = make_state(graded_documents=[], retry_count=0)

        assert pipeline._decide_to_generate(state) == "rewrite"

    def test_generate_at_max_retries(self, pipeline):
        """Should generate when max retries reached, even with few docs."""
        graded = [make_doc("A")]
        state = make_state(graded_documents=graded, retry_count=3)

        assert pipeline._decide_to_generate(state) == "generate"

    def test_generate_past_max_retries(self, pipeline):
        """Should generate when past max retries."""
        state = make_state(graded_documents=[], retry_count=5)

        assert pipeline._decide_to_generate(state) == "generate"

    def test_generate_on_query_loop(self, pipeline):
        """Should generate when rewrite produces same query twice."""
        state = make_state(
            graded_documents=[make_doc("A")],
            retry_count=1,
            query_history=["better query", "better query"],
        )

        assert pipeline._decide_to_generate(state) == "generate"

    def test_rewrite_when_queries_differ(self, pipeline):
        """Should rewrite when query history shows different queries."""
        state = make_state(
            graded_documents=[make_doc("A")],
            retry_count=1,
            query_history=["query v1", "query v2"],
        )

        assert pipeline._decide_to_generate(state) == "rewrite"

    def test_rewrite_with_single_history_entry(self, pipeline):
        """With only 1 history entry, loop detection shouldn't trigger."""
        state = make_state(
            graded_documents=[make_doc("A")],
            retry_count=1,
            query_history=["better query"],
        )

        assert pipeline._decide_to_generate(state) == "rewrite"

    def test_generate_threshold_exactly_at_min(self, pipeline):
        """Exactly min_relevant docs should route to generate."""
        graded = [make_doc("A"), make_doc("B"), make_doc("C")]
        state = make_state(graded_documents=graded, retry_count=0)

        assert pipeline._decide_to_generate(state) == "generate"

    def test_rewrite_threshold_one_below_min(self, pipeline):
        """One below min_relevant should route to rewrite."""
        graded = [make_doc("A"), make_doc("B")]
        state = make_state(graded_documents=graded, retry_count=0)

        assert pipeline._decide_to_generate(state) == "rewrite"

    def test_custom_min_relevant(self):
        """Custom min_relevant should be respected."""
        retriever, reranker, generator, grader, rewriter = create_mock_components()
        pipeline = AgenticRAGPipeline(
            hybrid_retriever=retriever,
            reranker=reranker,
            generator=generator,
            grader=grader,
            query_rewriter=rewriter,
            min_relevant=5,
        )

        graded = [make_doc(f"d{i}") for i in range(4)]
        state = make_state(graded_documents=graded, retry_count=0)

        assert pipeline._decide_to_generate(state) == "rewrite"

    def test_custom_max_retries(self):
        """Custom max_retries should be respected."""
        retriever, reranker, generator, grader, rewriter = create_mock_components()
        pipeline = AgenticRAGPipeline(
            hybrid_retriever=retriever,
            reranker=reranker,
            generator=generator,
            grader=grader,
            query_rewriter=rewriter,
            max_retries=1,
        )

        state = make_state(graded_documents=[], retry_count=1)

        assert pipeline._decide_to_generate(state) == "generate"


class TestRewriteQueryNode:
    """Tests for _rewrite_query_node."""

    @pytest.fixture
    def pipeline(self):
        retriever, reranker, generator, grader, rewriter = create_mock_components()
        rewriter.rewrite.return_value = "Beyonce birth year date"
        return AgenticRAGPipeline(
            hybrid_retriever=retriever,
            reranker=reranker,
            generator=generator,
            grader=grader,
            query_rewriter=rewriter,
        )

    def test_returns_rewritten_query(self, pipeline):
        """Node should return the new rewritten query."""
        state = make_state(query="When was she born?")

        result = pipeline._rewrite_query_node(state)

        assert result["rewritten_query"] == "Beyonce birth year date"

    def test_increments_retry_count(self, pipeline):
        """retry_count should increase by 1."""
        state = make_state(retry_count=0)

        result = pipeline._rewrite_query_node(state)

        assert result["retry_count"] == 1

    def test_increments_from_existing_count(self, pipeline):
        """retry_count should build on existing value."""
        state = make_state(retry_count=2)

        result = pipeline._rewrite_query_node(state)

        assert result["retry_count"] == 3

    def test_appends_to_query_history(self, pipeline):
        """New query should be appended to history."""
        state = make_state()

        result = pipeline._rewrite_query_node(state)

        assert result["query_history"] == ["Beyonce birth year date"]

    def test_calls_rewriter_with_stats(self, pipeline):
        """Rewriter should receive current query and retrieval stats."""
        docs = [make_doc("A"), make_doc("B"), make_doc("C")]
        graded = [make_doc("A")]
        state = make_state(
            query="original query",
            documents=docs,
            graded_documents=graded,
        )

        pipeline._rewrite_query_node(state)

        pipeline.query_rewriter.rewrite.assert_called_once_with("original query", 3, 1)

    def test_always_rewrites_from_original_query(self, pipeline):
        """On subsequent retries, should rewrite from original query (not previous rewrite)."""
        state = make_state(
            query="original query",
            rewritten_query="already rewritten",
            documents=[make_doc("A")],
            graded_documents=[],
        )

        pipeline._rewrite_query_node(state)

        pipeline.query_rewriter.rewrite.assert_called_once_with("original query", 1, 0)

    def test_logs_intermediate_step(self, pipeline):
        """Node should log the rewrite action."""
        state = make_state(query="original query")

        result = pipeline._rewrite_query_node(state)

        assert len(result["intermediate_steps"]) == 1
        assert "Rewrote" in result["intermediate_steps"][0]
        assert "Beyonce birth year date" in result["intermediate_steps"][0]


class TestGenerateNode:
    """Tests for _generate_node."""

    @pytest.fixture
    def pipeline(self):
        retriever, reranker, generator, grader, rewriter = create_mock_components()
        generator.generate.return_value = {"answer": "Beyonce was born in 1981."}
        return AgenticRAGPipeline(
            hybrid_retriever=retriever,
            reranker=reranker,
            generator=generator,
            grader=grader,
            query_rewriter=rewriter,
        )

    def test_generates_from_all_reranked_docs(self, pipeline):
        """Node should generate answer from all reranked documents, not just graded."""
        docs = [make_doc("d1"), make_doc("d2")]
        graded = [make_doc("d1")]  # grading filtered one, but generate uses all
        state = make_state(
            query="When was Beyonce born?",
            documents=docs,
            graded_documents=graded,
        )

        result = pipeline._generate_node(state)

        assert result["generation"] == "Beyonce was born in 1981."
        # Verify all reranked docs were passed, not just graded
        call_args = pipeline.generator.generate.call_args[0]
        assert call_args[1] == docs

    def test_calls_generator_with_query_and_all_docs(self, pipeline):
        """Generator should receive original query and all reranked docs."""
        docs = [make_doc("d1"), make_doc("d2"), make_doc("d3")]
        state = make_state(
            query="When was Beyonce born?",
            documents=docs,
            graded_documents=[make_doc("d1")],
        )

        pipeline._generate_node(state)

        pipeline.generator.generate.assert_called_once_with(
            "When was Beyonce born?", docs, max_chunks=5
        )

    def test_uses_original_query_not_rewritten(self, pipeline):
        """Generator should always use the original query, not the rewritten one."""
        docs = [make_doc("relevant doc")]
        state = make_state(
            query="When was she born?",
            rewritten_query="Beyonce birth year",
            documents=docs,
            graded_documents=docs,
        )

        pipeline._generate_node(state)

        call_args = pipeline.generator.generate.call_args[0]
        assert call_args[0] == "When was she born?"

    def test_generates_even_when_no_graded_docs(self, pipeline):
        """When graded_documents is empty, should still use all reranked docs."""
        retrieved = [
            make_doc("d1", "1"),
            make_doc("d2", "2"),
            make_doc("d3", "3"),
            make_doc("d4", "4"),
        ]
        state = make_state(query="hard query", documents=retrieved, graded_documents=[])

        pipeline._generate_node(state)

        call_args = pipeline.generator.generate.call_args
        docs_used = call_args[0][1]
        assert len(docs_used) == 4
        assert docs_used[0]["doc_id"] == "1"

    def test_generates_with_few_retrieved(self, pipeline):
        """Should pass all available docs even if fewer than max_chunks."""
        retrieved = [make_doc("d1", "1"), make_doc("d2", "2")]
        state = make_state(query="query", documents=retrieved, graded_documents=[])

        pipeline._generate_node(state)

        call_args = pipeline.generator.generate.call_args
        docs_used = call_args[0][1]
        assert len(docs_used) == 2

    def test_logs_intermediate_step(self, pipeline):
        """Node should log total docs and graded count."""
        docs = [make_doc("A"), make_doc("B"), make_doc("C")]
        graded = [make_doc("A"), make_doc("B")]
        state = make_state(query="q", documents=docs, graded_documents=graded)

        result = pipeline._generate_node(state)

        assert len(result["intermediate_steps"]) == 1
        assert "3" in result["intermediate_steps"][0]  # 3 total docs
        assert "2" in result["intermediate_steps"][0]  # 2 graded relevant


class TestQueryMethodNoRetry:
    """Tests for query() when grading passes on first attempt (no retry)."""

    @pytest.fixture
    def pipeline(self):
        retriever, reranker, generator, grader, rewriter = create_mock_components()

        reranked = [make_doc("A", "1"), make_doc("B", "2"), make_doc("C", "3")]

        retriever.search.return_value = reranked
        reranker.rerank.return_value = reranked
        grader.grade_batch.return_value = [True, True, True]  # >= 3 relevant
        generator.generate.return_value = {"answer": "Test answer"}

        return AgenticRAGPipeline(
            hybrid_retriever=retriever,
            reranker=reranker,
            generator=generator,
            grader=grader,
            query_rewriter=rewriter,
        )

    def test_returns_required_keys(self, pipeline):
        """Result dict should contain all expected keys."""
        result = pipeline.query("test query")

        assert "query" in result
        assert "answer" in result
        assert "steps" in result
        assert "num_docs_retrieved" in result
        assert "num_docs_graded" in result
        assert "retry_count" in result

    def test_returns_original_query(self, pipeline):
        """Result should contain the original query string."""
        result = pipeline.query("When was Beyonce born?")

        assert result["query"] == "When was Beyonce born?"

    def test_returns_generated_answer(self, pipeline):
        """Result should contain the generated answer."""
        result = pipeline.query("test query")

        assert result["answer"] == "Test answer"

    def test_steps_has_three_entries(self, pipeline):
        """No-retry pipeline should produce exactly 3 steps."""
        result = pipeline.query("test query")

        assert len(result["steps"]) == 3

    def test_num_docs_retrieved(self, pipeline):
        """num_docs_retrieved should match reranked count."""
        result = pipeline.query("test query")

        assert result["num_docs_retrieved"] == 3

    def test_num_docs_graded(self, pipeline):
        """num_docs_graded should match filtered count."""
        result = pipeline.query("test query")

        assert result["num_docs_graded"] == 3

    def test_retry_count_zero(self, pipeline):
        """No retry should mean retry_count=0."""
        result = pipeline.query("test query")

        assert result["retry_count"] == 0

    def test_intermediate_steps_accumulate(self, pipeline):
        """Steps should accumulate across all 3 nodes."""
        result = pipeline.query("test query")

        steps = result["steps"]
        assert any("Retrieved" in s for s in steps)
        assert any("Graded" in s for s in steps)
        assert any("Generated" in s for s in steps)


class TestQueryMethodWithRetry:
    """Tests for query() when retry is triggered."""

    @pytest.fixture
    def pipeline(self):
        """Pipeline where first grading fails (<3), retry succeeds."""
        retriever, reranker, generator, grader, rewriter = create_mock_components()

        reranked = [make_doc("A", "1"), make_doc("B", "2"), make_doc("C", "3")]
        retriever.search.return_value = reranked
        reranker.rerank.return_value = reranked

        # First call: 1 relevant (triggers rewrite), second call: 3 relevant
        grader.grade_batch.side_effect = [
            [True, False, False],
            [True, True, True],
        ]
        generator.generate.return_value = {"answer": "Retry answer"}
        rewriter.rewrite.return_value = "better query"

        return AgenticRAGPipeline(
            hybrid_retriever=retriever,
            reranker=reranker,
            generator=generator,
            grader=grader,
            query_rewriter=rewriter,
        )

    def test_retry_produces_answer(self, pipeline):
        """Pipeline should still produce an answer after retry."""
        result = pipeline.query("vague query")

        assert result["answer"] == "Retry answer"

    def test_retry_count_is_one(self, pipeline):
        """Single retry should give retry_count=1."""
        result = pipeline.query("vague query")

        assert result["retry_count"] == 1

    def test_steps_include_rewrite(self, pipeline):
        """Steps should include the rewrite action."""
        result = pipeline.query("vague query")

        steps = result["steps"]
        assert any("Rewrote" in s for s in steps)

    def test_more_steps_than_no_retry(self, pipeline):
        """Retry adds extra steps (rewrite + retrieve + grade)."""
        result = pipeline.query("vague query")

        # 3 (first pass) + 1 (rewrite) + 2 (retrieve + grade) + 1 (generate) = 7
        assert len(result["steps"]) > 3

    def test_rewriter_called(self, pipeline):
        """QueryRewriter should be called during retry."""
        pipeline.query("vague query")

        pipeline.query_rewriter.rewrite.assert_called_once()


class TestQueryMethodMaxRetries:
    """Tests for query() when max retries is reached."""

    @pytest.fixture
    def pipeline(self):
        """Pipeline where grading always fails, hitting max retries."""
        retriever, reranker, generator, grader, rewriter = create_mock_components()

        reranked = [make_doc("A", "1"), make_doc("B", "2"), make_doc("C", "3")]
        retriever.search.return_value = reranked
        reranker.rerank.return_value = reranked

        # Always returns only 1 relevant (below threshold of 3)
        grader.grade_batch.return_value = [True, False, False]
        generator.generate.return_value = {"answer": "Fallback answer"}
        rewriter.rewrite.return_value = "rewritten query"

        return AgenticRAGPipeline(
            hybrid_retriever=retriever,
            reranker=reranker,
            generator=generator,
            grader=grader,
            query_rewriter=rewriter,
            max_retries=2,
        )

    def test_still_generates_answer(self, pipeline):
        """Should generate an answer even after max retries."""
        result = pipeline.query("impossible query")

        assert result["answer"] == "Fallback answer"

    def test_retry_count_matches_max(self, pipeline):
        """retry_count should equal max_retries."""
        result = pipeline.query("impossible query")

        assert result["retry_count"] == 2

    def test_rewriter_called_max_times(self, pipeline):
        """Rewriter should be called exactly max_retries times."""
        pipeline.query("impossible query")

        assert pipeline.query_rewriter.rewrite.call_count == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
