"""
Unit tests for the agentic RAG pipeline (v3: adaptive retrieval + answer grading).

Tests:
- RAGState initialization and structure
- Individual node behavior with mocked components
- Graph compilation and structure
- Feature flags (adaptive retrieval, answer grading, rerank threshold)
- Conditional routing (_decide_retrieval_quality, _decide_after_grading)
- Retry generation with stricter prompt
- query() method output format
- Configurations: Linear, Adaptive, AnswerGrading, Full

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
    answer_grader = MagicMock()
    return retriever, reranker, generator, answer_grader


def make_doc(content: str, doc_id: str = "doc1", score: float = 0.9) -> dict:
    """Create a minimal document dict matching retriever output format."""
    return {
        "content": content,
        "doc_id": doc_id,
        "chunk_id": f"{doc_id}_0",
        "score": score,
        "rerank_score": score,
        "chunk_index": 0,
        "metadata": {},
    }


def make_state(**overrides) -> RAGState:
    """Create a RAGState with defaults, overriding specified fields."""
    state = {
        "query": "test query",
        "documents": [],
        "generation": "",
        "retry_count": 0,
        "min_rerank_score": 0.0,
        "used_fallback_retrieval": False,
        "used_web_search": False,
        "answer_is_acceptable": True,
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
        assert state["documents"] == []
        assert state["generation"] == ""
        assert state["retry_count"] == 0
        assert state["min_rerank_score"] == 0.0
        assert state["used_fallback_retrieval"] is False
        assert state["used_web_search"] is False
        assert state["answer_is_acceptable"] is True
        assert state["intermediate_steps"] == []

    def test_state_with_populated_fields(self):
        """RAGState should hold populated data."""
        docs = [make_doc("content 1"), make_doc("content 2")]
        state = make_state(
            query="When was Beyonce born?",
            documents=docs,
            generation="Beyonce was born in 1981.",
            retry_count=1,
            min_rerank_score=2.5,
            used_fallback_retrieval=True,
            answer_is_acceptable=False,
            intermediate_steps=["Retrieved 2 docs", "Generated answer"],
        )
        assert len(state["documents"]) == 2
        assert state["retry_count"] == 1
        assert state["min_rerank_score"] == 2.5
        assert state["used_fallback_retrieval"] is True
        assert state["answer_is_acceptable"] is False


class TestPipelineInit:
    """Tests for AgenticRAGPipeline initialization and graph compilation."""

    def test_init_stores_components(self):
        """Pipeline should store all component references."""
        retriever, reranker, generator, answer_grader = create_mock_components()

        pipeline = AgenticRAGPipeline(
            hybrid_retriever=retriever,
            reranker=reranker,
            generator=generator,
            answer_grader=answer_grader,
            enable_answer_grading=True,
        )

        assert pipeline.retriever is retriever
        assert pipeline.reranker is reranker
        assert pipeline.generator is generator
        assert pipeline.answer_grader is answer_grader

    def test_init_default_values(self):
        """Default k_retrieve=20, k_rerank=5, max_retries=1."""
        retriever, reranker, generator, _ = create_mock_components()

        pipeline = AgenticRAGPipeline(
            hybrid_retriever=retriever,
            reranker=reranker,
            generator=generator,
        )

        assert pipeline.k_retrieve == 20
        assert pipeline.k_rerank == 5
        assert pipeline.max_retries == 1

    def test_init_default_feature_flags(self):
        """Default: all features disabled (linear mode)."""
        retriever, reranker, generator, _ = create_mock_components()

        pipeline = AgenticRAGPipeline(
            hybrid_retriever=retriever,
            reranker=reranker,
            generator=generator,
        )

        assert pipeline.enable_adaptive_retrieval is False
        assert pipeline.enable_web_search is False
        assert pipeline.enable_answer_grading is False
        assert pipeline.enable_rerank_threshold is False

    def test_init_custom_values(self):
        """Custom k and retry values should be stored."""
        retriever, reranker, generator, answer_grader = create_mock_components()

        pipeline = AgenticRAGPipeline(
            hybrid_retriever=retriever,
            reranker=reranker,
            generator=generator,
            answer_grader=answer_grader,
            enable_answer_grading=True,
            k_retrieve=30,
            k_rerank=10,
            max_retries=2,
        )

        assert pipeline.k_retrieve == 30
        assert pipeline.k_rerank == 10
        assert pipeline.max_retries == 2

    def test_init_linear_mode(self):
        """Linear mode: no grader or fallback needed when all disabled."""
        retriever, reranker, generator, _ = create_mock_components()

        pipeline = AgenticRAGPipeline(
            hybrid_retriever=retriever,
            reranker=reranker,
            generator=generator,
        )

        assert pipeline.answer_grader is None
        assert pipeline.fallback_retriever is retriever  # defaults to primary

    def test_init_raises_without_grader(self):
        """Should raise ValueError when grading enabled but no grader."""
        retriever, reranker, generator, _ = create_mock_components()

        with pytest.raises(ValueError, match="answer_grader is required"):
            AgenticRAGPipeline(
                hybrid_retriever=retriever,
                reranker=reranker,
                generator=generator,
                enable_answer_grading=True,
            )

    def test_init_adaptive_defaults_fallback_to_primary(self):
        """Adaptive without explicit fallback should use primary retriever."""
        retriever, reranker, generator, _ = create_mock_components()

        pipeline = AgenticRAGPipeline(
            hybrid_retriever=retriever,
            reranker=reranker,
            generator=generator,
            enable_adaptive_retrieval=True,
        )
        assert pipeline.fallback_retriever is retriever

    def test_init_web_search_creates_default_tool(self):
        """Web search enabled without tool should create default DuckDuckGoSearchTool."""
        retriever, reranker, generator, _ = create_mock_components()

        pipeline = AgenticRAGPipeline(
            hybrid_retriever=retriever,
            reranker=reranker,
            generator=generator,
            enable_web_search=True,
        )

        assert pipeline.web_search_tool is not None
        assert pipeline.enable_web_search is True

    def test_init_web_search_uses_provided_tool(self):
        """Web search enabled with explicit tool should use it."""
        retriever, reranker, generator, _ = create_mock_components()
        web_tool = MagicMock()

        pipeline = AgenticRAGPipeline(
            hybrid_retriever=retriever,
            reranker=reranker,
            generator=generator,
            web_search_tool=web_tool,
            enable_web_search=True,
        )

        assert pipeline.web_search_tool is web_tool

    def test_init_web_search_disabled_no_default_tool(self):
        """Web search disabled should not create a tool."""
        retriever, reranker, generator, _ = create_mock_components()

        pipeline = AgenticRAGPipeline(
            hybrid_retriever=retriever,
            reranker=reranker,
            generator=generator,
        )

        assert pipeline.web_search_tool is None

    def test_graph_compiles(self):
        """_build_graph() should produce a compiled graph."""
        retriever, reranker, generator, answer_grader = create_mock_components()

        pipeline = AgenticRAGPipeline(
            hybrid_retriever=retriever,
            reranker=reranker,
            generator=generator,
            answer_grader=answer_grader,
            enable_answer_grading=True,
        )

        assert pipeline.app is not None

    def test_graph_compiles_linear(self):
        """Linear graph should compile without extra nodes."""
        retriever, reranker, generator, _ = create_mock_components()

        pipeline = AgenticRAGPipeline(
            hybrid_retriever=retriever,
            reranker=reranker,
            generator=generator,
        )

        assert pipeline.app is not None

    def test_graph_compiles_adaptive(self):
        """Adaptive graph should compile with fallback_retrieve node."""
        retriever, reranker, generator, _ = create_mock_components()
        fallback = MagicMock()

        pipeline = AgenticRAGPipeline(
            hybrid_retriever=retriever,
            reranker=reranker,
            generator=generator,
            fallback_retriever=fallback,
            enable_adaptive_retrieval=True,
        )

        assert pipeline.app is not None
        assert pipeline.fallback_retriever is fallback

    def test_graph_compiles_web_search(self):
        """Web search graph should compile with check_web and web_search nodes."""
        retriever, reranker, generator, _ = create_mock_components()

        pipeline = AgenticRAGPipeline(
            hybrid_retriever=retriever,
            reranker=reranker,
            generator=generator,
            enable_web_search=True,
        )

        assert pipeline.app is not None

    def test_graph_compiles_adaptive_plus_web_search(self):
        """Adaptive + web search graph should compile."""
        retriever, reranker, generator, _ = create_mock_components()
        fallback = MagicMock()

        pipeline = AgenticRAGPipeline(
            hybrid_retriever=retriever,
            reranker=reranker,
            generator=generator,
            fallback_retriever=fallback,
            enable_adaptive_retrieval=True,
            enable_web_search=True,
        )

        assert pipeline.app is not None


class TestRetrieveNode:
    """Tests for _retrieve_node."""

    @pytest.fixture
    def pipeline(self):
        retriever, reranker, generator, _ = create_mock_components()
        retriever.search.return_value = [
            make_doc("doc A", "a"),
            make_doc("doc B", "b"),
            make_doc("doc C", "c"),
        ]
        reranker.rerank.return_value = [
            make_doc("doc A", "a", 3.5),
            make_doc("doc B", "b", 1.2),
        ]
        return AgenticRAGPipeline(
            hybrid_retriever=retriever,
            reranker=reranker,
            generator=generator,
            k_retrieve=20,
            k_rerank=2,
        )

    def test_returns_reranked_documents(self, pipeline):
        """Node should return reranked documents."""
        state = make_state(query="test query")
        result = pipeline._retrieve_node(state)

        assert len(result["documents"]) == 2
        assert result["documents"][0]["doc_id"] == "a"

    def test_computes_min_rerank_score(self, pipeline):
        """Node should compute min rerank score from documents."""
        state = make_state(query="test query")
        result = pipeline._retrieve_node(state)

        assert result["min_rerank_score"] == 1.2

    def test_calls_retriever_with_correct_k(self, pipeline):
        """Retriever should be called with k_retrieve and k_retriever=50."""
        state = make_state(query="test query")
        pipeline._retrieve_node(state)

        pipeline.retriever.search.assert_called_once_with("test query", k=20, k_retriever=50)

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

    def test_empty_reranked_defaults_min_score_zero(self, pipeline):
        """Empty reranked list should default min_rerank_score to 0.0."""
        pipeline.reranker.rerank.return_value = []
        state = make_state(query="test query")
        result = pipeline._retrieve_node(state)

        assert result["min_rerank_score"] == 0.0


class TestDecideRetrievalQuality:
    """Tests for _decide_retrieval_quality routing logic."""

    @pytest.fixture
    def pipeline(self):
        retriever, reranker, generator, _ = create_mock_components()
        fallback = MagicMock()
        return AgenticRAGPipeline(
            hybrid_retriever=retriever,
            reranker=reranker,
            generator=generator,
            fallback_retriever=fallback,
            enable_adaptive_retrieval=True,
            retrieval_threshold=0.0,
        )

    def test_acceptable_when_above_threshold(self, pipeline):
        """Good rerank scores should route to generate."""
        state = make_state(min_rerank_score=2.0)
        assert pipeline._decide_retrieval_quality(state) == "acceptable"

    def test_fallback_when_below_threshold(self, pipeline):
        """Poor rerank scores should route to fallback retrieval."""
        state = make_state(min_rerank_score=-3.0)
        assert pipeline._decide_retrieval_quality(state) == "fallback"

    def test_acceptable_at_exact_threshold(self, pipeline):
        """Score equal to threshold should be acceptable (>= check)."""
        state = make_state(min_rerank_score=0.0)
        assert pipeline._decide_retrieval_quality(state) == "acceptable"

    def test_fallback_just_below_threshold(self, pipeline):
        """Score just below threshold should trigger fallback."""
        state = make_state(min_rerank_score=-0.001)
        assert pipeline._decide_retrieval_quality(state) == "fallback"

    def test_custom_threshold(self):
        """Custom threshold value should be used for routing."""
        retriever, reranker, generator, _ = create_mock_components()
        fallback = MagicMock()
        pipeline = AgenticRAGPipeline(
            hybrid_retriever=retriever,
            reranker=reranker,
            generator=generator,
            fallback_retriever=fallback,
            enable_adaptive_retrieval=True,
            retrieval_threshold=2.0,
        )

        # 1.5 < 2.0 → fallback
        state = make_state(min_rerank_score=1.5)
        assert pipeline._decide_retrieval_quality(state) == "fallback"

        # 3.0 >= 2.0 → acceptable
        state = make_state(min_rerank_score=3.0)
        assert pipeline._decide_retrieval_quality(state) == "acceptable"

    def test_negative_threshold(self):
        """Negative threshold should still work correctly."""
        retriever, reranker, generator, _ = create_mock_components()
        fallback = MagicMock()
        pipeline = AgenticRAGPipeline(
            hybrid_retriever=retriever,
            reranker=reranker,
            generator=generator,
            fallback_retriever=fallback,
            enable_adaptive_retrieval=True,
            retrieval_threshold=-5.0,
        )

        # -3.0 >= -5.0 → acceptable
        state = make_state(min_rerank_score=-3.0)
        assert pipeline._decide_retrieval_quality(state) == "acceptable"

        # -6.0 < -5.0 → fallback
        state = make_state(min_rerank_score=-6.0)
        assert pipeline._decide_retrieval_quality(state) == "fallback"


class TestFallbackRetrieveNode:
    """Tests for _fallback_retrieve_node."""

    @pytest.fixture
    def pipeline(self):
        retriever, reranker, generator, _ = create_mock_components()
        fallback = MagicMock()
        fallback.search.return_value = [
            make_doc("fallback A", "fa"),
            make_doc("fallback B", "fb"),
        ]
        reranker.rerank.return_value = [
            make_doc("fallback A", "fa", 2.0),
            make_doc("fallback B", "fb", 0.5),
        ]
        return AgenticRAGPipeline(
            hybrid_retriever=retriever,
            reranker=reranker,
            generator=generator,
            fallback_retriever=fallback,
            enable_adaptive_retrieval=True,
        )

    def test_uses_fallback_retriever(self, pipeline):
        """Should call fallback_retriever, not primary retriever."""
        state = make_state(query="test query")
        pipeline._fallback_retrieve_node(state)

        pipeline.fallback_retriever.search.assert_called_once()
        pipeline.retriever.search.assert_not_called()

    def test_returns_reranked_documents(self, pipeline):
        """Should return fallback-retrieved and reranked documents."""
        state = make_state(query="test query")
        result = pipeline._fallback_retrieve_node(state)

        assert len(result["documents"]) == 2
        assert result["documents"][0]["doc_id"] == "fa"

    def test_sets_fallback_flag(self, pipeline):
        """Should set used_fallback_retrieval to True."""
        state = make_state(query="test query")
        result = pipeline._fallback_retrieve_node(state)

        assert result["used_fallback_retrieval"] is True

    def test_computes_min_rerank_score(self, pipeline):
        """Should compute min rerank score from fallback docs."""
        state = make_state(query="test query")
        result = pipeline._fallback_retrieve_node(state)

        assert result["min_rerank_score"] == 0.5

    def test_logs_fallback_step(self, pipeline):
        """Intermediate step should mention 'Fallback'."""
        state = make_state(query="test query")
        result = pipeline._fallback_retrieve_node(state)

        assert "Fallback" in result["intermediate_steps"][0]

    def test_calls_fallback_with_wider_k(self, pipeline):
        """Fallback retriever should use wider k (2x primary by default)."""
        state = make_state(query="test query")
        pipeline._fallback_retrieve_node(state)

        pipeline.fallback_retriever.search.assert_called_once_with(
            "test query", k=40, k_retriever=100
        )


class TestCheckWebNode:
    """Tests for _check_web_node passthrough."""

    @pytest.fixture
    def pipeline(self):
        retriever, reranker, generator, _ = create_mock_components()
        return AgenticRAGPipeline(
            hybrid_retriever=retriever,
            reranker=reranker,
            generator=generator,
            enable_web_search=True,
        )

    def test_returns_empty_dict(self, pipeline):
        """Passthrough node should return empty dict (no state updates)."""
        state = make_state()
        result = pipeline._check_web_node(state)
        assert result == {}


class TestDecideWebSearch:
    """Tests for _decide_web_search routing logic."""

    @pytest.fixture
    def pipeline(self):
        retriever, reranker, generator, _ = create_mock_components()
        return AgenticRAGPipeline(
            hybrid_retriever=retriever,
            reranker=reranker,
            generator=generator,
            enable_web_search=True,
            web_search_threshold=-5.0,
        )

    def test_skip_when_above_threshold(self, pipeline):
        """Good rerank scores should skip web search."""
        state = make_state(min_rerank_score=2.0)
        assert pipeline._decide_web_search(state) == "skip_web"

    def test_web_search_when_below_threshold(self, pipeline):
        """Very poor rerank scores should trigger web search."""
        state = make_state(min_rerank_score=-7.0)
        assert pipeline._decide_web_search(state) == "web_search"

    def test_skip_at_exact_threshold(self, pipeline):
        """Score equal to threshold should skip (>= check)."""
        state = make_state(min_rerank_score=-5.0)
        assert pipeline._decide_web_search(state) == "skip_web"

    def test_web_search_just_below_threshold(self, pipeline):
        """Score just below threshold should trigger web search."""
        state = make_state(min_rerank_score=-5.001)
        assert pipeline._decide_web_search(state) == "web_search"

    def test_custom_threshold(self):
        """Custom web_search_threshold should be used for routing."""
        retriever, reranker, generator, _ = create_mock_components()
        pipeline = AgenticRAGPipeline(
            hybrid_retriever=retriever,
            reranker=reranker,
            generator=generator,
            enable_web_search=True,
            web_search_threshold=0.0,
        )

        state = make_state(min_rerank_score=-0.5)
        assert pipeline._decide_web_search(state) == "web_search"

        state = make_state(min_rerank_score=1.0)
        assert pipeline._decide_web_search(state) == "skip_web"

    def test_default_inf_skips_web(self, pipeline):
        """Missing min_rerank_score should default to inf (skip web)."""
        state = {"query": "q", "intermediate_steps": []}
        assert pipeline._decide_web_search(state) == "skip_web"


class TestWebSearchNode:
    """Tests for _web_search_node."""

    @pytest.fixture
    def pipeline(self):
        retriever, reranker, generator, _ = create_mock_components()
        web_tool = MagicMock()
        web_tool.search.return_value = [
            {"content": "Web result 1", "source": "https://example.com/1"},
            {"content": "Web result 2", "source": "https://example.com/2"},
        ]
        return AgenticRAGPipeline(
            hybrid_retriever=retriever,
            reranker=reranker,
            generator=generator,
            web_search_tool=web_tool,
            enable_web_search=True,
        )

    def test_calls_web_search_tool(self, pipeline):
        """Should call web_search_tool.search with the query."""
        state = make_state(query="test query")
        pipeline._web_search_node(state)

        pipeline.web_search_tool.search.assert_called_once_with("test query")

    def test_merges_web_docs_with_existing(self, pipeline):
        """Should merge web results with existing local documents."""
        local_docs = [make_doc("local A", "la")]
        state = make_state(query="test query", documents=local_docs)
        result = pipeline._web_search_node(state)

        assert len(result["documents"]) == 3  # 1 local + 2 web
        assert result["documents"][0]["doc_id"] == "la"
        assert result["documents"][1]["content"] == "Web result 1"
        assert result["documents"][2]["content"] == "Web result 2"

    def test_sets_web_search_flag(self, pipeline):
        """Should set used_web_search to True."""
        state = make_state(query="test query")
        result = pipeline._web_search_node(state)

        assert result["used_web_search"] is True

    def test_logs_intermediate_step(self, pipeline):
        """Should log a step with result count."""
        state = make_state(query="test query")
        result = pipeline._web_search_node(state)

        assert len(result["intermediate_steps"]) == 1
        assert "Web search" in result["intermediate_steps"][0]
        assert "2" in result["intermediate_steps"][0]

    def test_empty_web_results(self, pipeline):
        """Should handle empty web results (still merges)."""
        pipeline.web_search_tool.search.return_value = []
        local_docs = [make_doc("local", "l")]
        state = make_state(query="test query", documents=local_docs)
        result = pipeline._web_search_node(state)

        assert len(result["documents"]) == 1  # only local
        assert result["used_web_search"] is True

    def test_web_results_with_empty_local(self, pipeline):
        """Should work when local docs are empty."""
        state = make_state(query="test query", documents=[])
        result = pipeline._web_search_node(state)

        assert len(result["documents"]) == 2  # only web


class TestGenerateNode:
    """Tests for _generate_node."""

    @pytest.fixture
    def pipeline(self):
        retriever, reranker, generator, answer_grader = create_mock_components()
        generator.generate.return_value = {"answer": "Beyonce was born in 1981."}
        generator._format_context.return_value = "[1] context text"
        generator._generate_text.return_value = "Precise answer text."
        generator._parse_answer.return_value = "Precise answer text."
        return AgenticRAGPipeline(
            hybrid_retriever=retriever,
            reranker=reranker,
            generator=generator,
            answer_grader=answer_grader,
            enable_answer_grading=True,
        )

    def test_first_attempt_uses_standard_generate(self, pipeline):
        """First attempt (retry_count=0) should use generator.generate()."""
        docs = [make_doc("d1"), make_doc("d2")]
        state = make_state(
            query="When was Beyonce born?",
            documents=docs,
            retry_count=0,
        )

        result = pipeline._generate_node(state)

        assert result["generation"] == "Beyonce was born in 1981."
        pipeline.generator.generate.assert_called_once_with(
            "When was Beyonce born?", docs, max_chunks=5
        )

    def test_retry_uses_stricter_prompt(self, pipeline):
        """Retry (retry_count > 0) should use RETRY_PROMPT and _generate_text."""
        docs = [make_doc("d1")]
        state = make_state(
            query="When was Beyonce born?",
            documents=docs,
            retry_count=1,
        )

        result = pipeline._generate_node(state)

        assert result["generation"] == "Precise answer text."
        pipeline.generator._generate_text.assert_called_once()
        pipeline.generator._parse_answer.assert_called_once()
        # generator.generate should NOT be called on retry
        pipeline.generator.generate.assert_not_called()

    def test_retry_prompt_contains_question(self, pipeline):
        """Retry prompt should contain the original question."""
        state = make_state(
            query="test question?",
            documents=[make_doc("d1")],
            retry_count=1,
        )

        pipeline._generate_node(state)

        call_args = pipeline.generator._generate_text.call_args[0][0]
        assert "test question?" in call_args

    def test_logs_attempt_number(self, pipeline):
        """Step log should include the attempt number."""
        state = make_state(retry_count=0)
        result = pipeline._generate_node(state)
        assert "attempt 1" in result["intermediate_steps"][0]

        state = make_state(retry_count=1, documents=[make_doc("d1")])
        result = pipeline._generate_node(state)
        assert "attempt 2" in result["intermediate_steps"][0]

    def test_generates_with_empty_docs(self, pipeline):
        """Should handle empty documents list gracefully."""
        state = make_state(query="query", documents=[], retry_count=0)
        result = pipeline._generate_node(state)

        assert "generation" in result


class TestGradeAnswerNode:
    """Tests for _grade_answer_node."""

    @pytest.fixture
    def pipeline(self):
        retriever, reranker, generator, answer_grader = create_mock_components()
        answer_grader.grade.return_value = True
        return AgenticRAGPipeline(
            hybrid_retriever=retriever,
            reranker=reranker,
            generator=generator,
            answer_grader=answer_grader,
            enable_answer_grading=True,
        )

    def test_acceptable_answer(self, pipeline):
        """Acceptable answer should set answer_is_acceptable=True."""
        pipeline.answer_grader.grade.return_value = True
        docs = [make_doc("context A")]
        state = make_state(
            query="test query",
            generation="good answer",
            documents=docs,
        )

        result = pipeline._grade_answer_node(state)

        assert result["answer_is_acceptable"] is True

    def test_unacceptable_answer(self, pipeline):
        """Unacceptable answer should set answer_is_acceptable=False."""
        pipeline.answer_grader.grade.return_value = False
        docs = [make_doc("context A")]
        state = make_state(
            query="test query",
            generation="bad answer",
            documents=docs,
        )

        result = pipeline._grade_answer_node(state)

        assert result["answer_is_acceptable"] is False

    def test_increments_retry_count_on_failure(self, pipeline):
        """retry_count should increment when answer is not acceptable."""
        pipeline.answer_grader.grade.return_value = False
        state = make_state(
            query="q",
            generation="bad",
            documents=[make_doc("d")],
            retry_count=0,
        )

        result = pipeline._grade_answer_node(state)

        assert result["retry_count"] == 1

    def test_does_not_increment_on_success(self, pipeline):
        """retry_count should NOT be in update when answer is acceptable."""
        pipeline.answer_grader.grade.return_value = True
        state = make_state(
            query="q",
            generation="good",
            documents=[make_doc("d")],
            retry_count=0,
        )

        result = pipeline._grade_answer_node(state)

        assert "retry_count" not in result

    def test_calls_grader_with_doc_contents(self, pipeline):
        """Grader should receive document content strings."""
        docs = [make_doc("content A"), make_doc("content B")]
        state = make_state(
            query="test query",
            generation="test answer",
            documents=docs,
        )

        pipeline._grade_answer_node(state)

        pipeline.answer_grader.grade.assert_called_once_with(
            "test query", "test answer", ["content A", "content B"]
        )

    def test_limits_to_5_docs(self, pipeline):
        """Should only send first 5 docs to grader."""
        docs = [make_doc(f"doc {i}", f"d{i}") for i in range(10)]
        state = make_state(
            query="q",
            generation="answer",
            documents=docs,
        )

        pipeline._grade_answer_node(state)

        call_args = pipeline.answer_grader.grade.call_args[0]
        assert len(call_args[2]) == 5

    def test_logs_intermediate_step(self, pipeline):
        """Node should log the grading result."""
        pipeline.answer_grader.grade.return_value = True
        state = make_state(query="q", generation="a", documents=[make_doc("d")])
        result = pipeline._grade_answer_node(state)

        assert "acceptable" in result["intermediate_steps"][0]


class TestDecideAfterGrading:
    """Tests for _decide_after_grading routing logic."""

    @pytest.fixture
    def pipeline(self):
        retriever, reranker, generator, answer_grader = create_mock_components()
        return AgenticRAGPipeline(
            hybrid_retriever=retriever,
            reranker=reranker,
            generator=generator,
            answer_grader=answer_grader,
            enable_answer_grading=True,
            max_retries=1,
        )

    def test_accept_when_acceptable(self, pipeline):
        """Should accept when answer is acceptable."""
        state = make_state(answer_is_acceptable=True, retry_count=0)
        assert pipeline._decide_after_grading(state) == "accept"

    def test_retry_when_not_acceptable(self, pipeline):
        """Should retry when answer is not acceptable and retries remain."""
        state = make_state(answer_is_acceptable=False, retry_count=1)
        assert pipeline._decide_after_grading(state) == "retry"

    def test_accept_at_max_retries(self, pipeline):
        """Should accept when max retries exceeded."""
        state = make_state(answer_is_acceptable=False, retry_count=2)
        assert pipeline._decide_after_grading(state) == "accept"

    def test_accept_past_max_retries(self, pipeline):
        """Should accept when well past max retries."""
        state = make_state(answer_is_acceptable=False, retry_count=5)
        assert pipeline._decide_after_grading(state) == "accept"

    def test_default_acceptable_is_true(self, pipeline):
        """Missing answer_is_acceptable should default to True (accept)."""
        state = {"query": "q", "retry_count": 0, "intermediate_steps": []}
        assert pipeline._decide_after_grading(state) == "accept"


class TestDecideWithRerankThreshold:
    """Tests for _decide_after_grading with rerank threshold enabled."""

    @pytest.fixture
    def pipeline(self):
        retriever, reranker, generator, answer_grader = create_mock_components()
        return AgenticRAGPipeline(
            hybrid_retriever=retriever,
            reranker=reranker,
            generator=generator,
            answer_grader=answer_grader,
            enable_answer_grading=True,
            max_retries=2,
            enable_rerank_threshold=True,
            rerank_threshold=1.0,
        )

    def test_accept_when_low_retrieval_quality(self, pipeline):
        """Should accept (skip retry) when rerank score below threshold."""
        state = make_state(
            answer_is_acceptable=False,
            retry_count=1,
            min_rerank_score=0.5,  # below threshold of 1.0
        )
        assert pipeline._decide_after_grading(state) == "accept"

    def test_retry_when_good_retrieval_quality(self, pipeline):
        """Should retry when rerank score above threshold."""
        state = make_state(
            answer_is_acceptable=False,
            retry_count=1,
            min_rerank_score=2.0,  # above threshold of 1.0
        )
        assert pipeline._decide_after_grading(state) == "retry"

    def test_accept_when_score_equals_threshold(self, pipeline):
        """Score exactly at threshold should still trigger retry (< check)."""
        state = make_state(
            answer_is_acceptable=False,
            retry_count=1,
            min_rerank_score=1.0,  # equal to threshold
        )
        # min_score < threshold is False (1.0 < 1.0), so should retry
        assert pipeline._decide_after_grading(state) == "retry"

    def test_threshold_ignored_when_acceptable(self, pipeline):
        """Threshold should not matter when answer is acceptable."""
        state = make_state(
            answer_is_acceptable=True,
            retry_count=0,
            min_rerank_score=-5.0,  # terrible score, but answer is fine
        )
        assert pipeline._decide_after_grading(state) == "accept"


class TestDecideCustomMaxRetries:
    """Tests for _decide_after_grading with custom max_retries."""

    def test_custom_max_retries_2(self):
        """max_retries=2 should allow 2 retries."""
        retriever, reranker, generator, answer_grader = create_mock_components()
        pipeline = AgenticRAGPipeline(
            hybrid_retriever=retriever,
            reranker=reranker,
            generator=generator,
            answer_grader=answer_grader,
            enable_answer_grading=True,
            max_retries=2,
        )

        # retry_count=1 -> retry (1 failure, still under max)
        state = make_state(answer_is_acceptable=False, retry_count=1)
        assert pipeline._decide_after_grading(state) == "retry"

        # retry_count=2 -> retry (2 failures, check is > not >=)
        state = make_state(answer_is_acceptable=False, retry_count=2)
        assert pipeline._decide_after_grading(state) == "retry"

        # retry_count=3 -> accept (3 > 2, max retries exceeded)
        state = make_state(answer_is_acceptable=False, retry_count=3)
        assert pipeline._decide_after_grading(state) == "accept"


class TestQueryMethodLinear:
    """Tests for query() in linear mode (no features enabled)."""

    @pytest.fixture
    def pipeline(self):
        retriever, reranker, generator, _ = create_mock_components()

        reranked = [make_doc("A", "1"), make_doc("B", "2"), make_doc("C", "3")]
        retriever.search.return_value = reranked
        reranker.rerank.return_value = reranked
        generator.generate.return_value = {"answer": "Test answer"}

        return AgenticRAGPipeline(
            hybrid_retriever=retriever,
            reranker=reranker,
            generator=generator,
        )

    def test_returns_required_keys(self, pipeline):
        """Result dict should contain all expected keys."""
        result = pipeline.query("test query")

        assert "query" in result
        assert "answer" in result
        assert "steps" in result
        assert "num_docs_retrieved" in result
        assert "retry_count" in result
        assert "min_rerank_score" in result
        assert "used_fallback_retrieval" in result
        assert "used_web_search" in result
        assert "answer_is_acceptable" in result

    def test_web_search_not_used_by_default(self, pipeline):
        """Linear mode should not use web search."""
        result = pipeline.query("test query")
        assert result["used_web_search"] is False

    def test_returns_original_query(self, pipeline):
        """Result should contain the original query string."""
        result = pipeline.query("When was Beyonce born?")
        assert result["query"] == "When was Beyonce born?"

    def test_returns_generated_answer(self, pipeline):
        """Result should contain the generated answer."""
        result = pipeline.query("test query")
        assert result["answer"] == "Test answer"

    def test_steps_has_two_entries(self, pipeline):
        """Linear pipeline should produce exactly 2 steps (retrieve + generate)."""
        result = pipeline.query("test query")
        assert len(result["steps"]) == 2

    def test_retry_count_zero(self, pipeline):
        """Linear mode should have retry_count=0."""
        result = pipeline.query("test query")
        assert result["retry_count"] == 0

    def test_fallback_not_used(self, pipeline):
        """Linear mode should not use fallback retrieval."""
        result = pipeline.query("test query")
        assert result["used_fallback_retrieval"] is False


class TestQueryMethodAdaptive:
    """Tests for query() with adaptive retrieval."""

    def _make_pipeline(self, primary_scores, fallback_scores=None):
        """Helper to create pipeline with configurable rerank scores."""
        retriever, reranker, generator, _ = create_mock_components()
        fallback = MagicMock()

        primary_docs = [make_doc(f"P{i}", f"p{i}", s) for i, s in enumerate(primary_scores)]
        retriever.search.return_value = primary_docs
        generator.generate.return_value = {"answer": "Test answer"}

        if fallback_scores is not None:
            fallback_docs = [make_doc(f"F{i}", f"f{i}", s) for i, s in enumerate(fallback_scores)]
            fallback.search.return_value = fallback_docs
            # Reranker: return primary on first call, fallback on second
            reranker.rerank.side_effect = [primary_docs, fallback_docs]
        else:
            reranker.rerank.return_value = primary_docs

        return AgenticRAGPipeline(
            hybrid_retriever=retriever,
            reranker=reranker,
            generator=generator,
            fallback_retriever=fallback,
            enable_adaptive_retrieval=True,
            retrieval_threshold=0.0,
        )

    def test_good_retrieval_skips_fallback(self):
        """Good rerank scores should skip fallback retrieval."""
        pipeline = self._make_pipeline([3.0, 1.5, 0.5])
        result = pipeline.query("test")

        assert result["used_fallback_retrieval"] is False
        assert result["answer"] == "Test answer"
        assert len(result["steps"]) == 2  # retrieve + generate

    def test_poor_retrieval_triggers_fallback(self):
        """Poor rerank scores should trigger fallback retrieval."""
        pipeline = self._make_pipeline([-2.0, -3.0], [2.0, 1.0])
        result = pipeline.query("test")

        assert result["used_fallback_retrieval"] is True
        assert result["answer"] == "Test answer"
        assert len(result["steps"]) == 3  # retrieve + fallback + generate
        assert any("Fallback" in s for s in result["steps"])

    def test_fallback_uses_fallback_retriever(self):
        """Fallback path should use fallback_retriever, not primary."""
        pipeline = self._make_pipeline([-2.0], [1.0])
        pipeline.query("test")

        pipeline.fallback_retriever.search.assert_called_once()


class TestQueryMethodWithAnswerGrading:
    """Tests for query() when answer grading is enabled and answer passes."""

    @pytest.fixture
    def pipeline(self):
        retriever, reranker, generator, answer_grader = create_mock_components()

        reranked = [make_doc("A", "1"), make_doc("B", "2"), make_doc("C", "3")]
        retriever.search.return_value = reranked
        reranker.rerank.return_value = reranked
        generator.generate.return_value = {"answer": "Test answer"}
        answer_grader.grade.return_value = True

        return AgenticRAGPipeline(
            hybrid_retriever=retriever,
            reranker=reranker,
            generator=generator,
            answer_grader=answer_grader,
            enable_answer_grading=True,
        )

    def test_returns_answer(self, pipeline):
        """Should return the generated answer when grading passes."""
        result = pipeline.query("test query")
        assert result["answer"] == "Test answer"

    def test_steps_has_three_entries(self, pipeline):
        """Pipeline with grading should produce 3 steps."""
        result = pipeline.query("test query")
        assert len(result["steps"]) == 3

    def test_answer_is_acceptable(self, pipeline):
        """answer_is_acceptable should be True when grading passes."""
        result = pipeline.query("test query")
        assert result["answer_is_acceptable"] is True

    def test_retry_count_zero(self, pipeline):
        """No retry needed when grading passes."""
        result = pipeline.query("test query")
        assert result["retry_count"] == 0

    def test_steps_include_grading(self, pipeline):
        """Steps should include answer grading step."""
        result = pipeline.query("test query")
        steps = result["steps"]
        assert any("Answer grade" in s for s in steps)


class TestQueryMethodWithRetry:
    """Tests for query() when answer grading fails and triggers retry."""

    @pytest.fixture
    def pipeline(self):
        """Pipeline where first grading fails, retry succeeds."""
        retriever, reranker, generator, answer_grader = create_mock_components()

        reranked = [make_doc("A", "1"), make_doc("B", "2")]
        retriever.search.return_value = reranked
        reranker.rerank.return_value = reranked

        generator.generate.return_value = {"answer": "Bad first answer"}
        generator._format_context.return_value = "[1] context"
        generator._generate_text.return_value = "Good retry answer."
        generator._parse_answer.return_value = "Good retry answer."

        # First call: not acceptable, second call: acceptable
        answer_grader.grade.side_effect = [False, True]

        return AgenticRAGPipeline(
            hybrid_retriever=retriever,
            reranker=reranker,
            generator=generator,
            answer_grader=answer_grader,
            enable_answer_grading=True,
            max_retries=1,
        )

    def test_retry_produces_answer(self, pipeline):
        """Pipeline should produce answer from retry."""
        result = pipeline.query("test query")
        assert result["answer"] == "Good retry answer."

    def test_retry_count_is_one(self, pipeline):
        """Single retry should give retry_count=1."""
        result = pipeline.query("test query")
        assert result["retry_count"] == 1

    def test_steps_include_retry_generation(self, pipeline):
        """Steps should show both generation attempts."""
        result = pipeline.query("test query")
        steps = result["steps"]
        assert any("attempt 1" in s for s in steps)
        assert any("attempt 2" in s for s in steps)

    def test_more_steps_than_linear(self, pipeline):
        """Retry should produce more steps than linear (3 base + 2 retry)."""
        result = pipeline.query("test query")
        # retrieve + generate + grade(fail) + generate(retry) + grade(pass) = 5
        assert len(result["steps"]) == 5

    def test_grader_called_twice(self, pipeline):
        """Answer grader should be called twice (original + retry)."""
        pipeline.query("test query")
        assert pipeline.answer_grader.grade.call_count == 2


class TestQueryMethodMaxRetries:
    """Tests for query() when max retries is reached."""

    @pytest.fixture
    def pipeline(self):
        """Pipeline where answer grading always fails."""
        retriever, reranker, generator, answer_grader = create_mock_components()

        reranked = [make_doc("A", "1"), make_doc("B", "2")]
        retriever.search.return_value = reranked
        reranker.rerank.return_value = reranked

        generator.generate.return_value = {"answer": "First answer"}
        generator._format_context.return_value = "[1] context"
        generator._generate_text.return_value = "Retry answer."
        generator._parse_answer.return_value = "Retry answer."

        # Always fails
        answer_grader.grade.return_value = False

        return AgenticRAGPipeline(
            hybrid_retriever=retriever,
            reranker=reranker,
            generator=generator,
            answer_grader=answer_grader,
            enable_answer_grading=True,
            max_retries=1,
        )

    def test_still_generates_answer(self, pipeline):
        """Should return an answer even after max retries."""
        result = pipeline.query("test query")
        assert result["answer"] == "Retry answer."

    def test_retry_count_matches_max(self, pipeline):
        """retry_count should reflect grading failures."""
        result = pipeline.query("test query")
        # grade fails → retry_count=1, retry, grade fails → retry_count=2
        # 2 > max_retries(1) → accept
        assert result["retry_count"] == 2

    def test_grader_called_twice(self, pipeline):
        """Grader called for initial + 1 retry = 2 times."""
        pipeline.query("test query")
        assert pipeline.answer_grader.grade.call_count == 2


class TestFeatureFlags:
    """Tests for pipeline behavior with different feature flag combinations."""

    def test_linear_config(self):
        """All disabled: should work as simple retrieve->generate."""
        retriever, reranker, generator, _ = create_mock_components()
        reranked = [make_doc("A", "1")]
        retriever.search.return_value = reranked
        reranker.rerank.return_value = reranked
        generator.generate.return_value = {"answer": "Linear answer"}

        pipeline = AgenticRAGPipeline(
            hybrid_retriever=retriever,
            reranker=reranker,
            generator=generator,
        )

        result = pipeline.query("test")
        assert result["answer"] == "Linear answer"
        assert len(result["steps"]) == 2

    def test_adaptive_only_config(self):
        """Adaptive retrieval only, no answer grading."""
        retriever, reranker, generator, _ = create_mock_components()
        fallback = MagicMock()

        reranked = [make_doc("A", "1", 2.0)]
        retriever.search.return_value = reranked
        reranker.rerank.return_value = reranked
        generator.generate.return_value = {"answer": "Adaptive answer"}

        pipeline = AgenticRAGPipeline(
            hybrid_retriever=retriever,
            reranker=reranker,
            generator=generator,
            fallback_retriever=fallback,
            enable_adaptive_retrieval=True,
            retrieval_threshold=0.0,
        )

        result = pipeline.query("test")
        assert result["answer"] == "Adaptive answer"
        assert result["used_fallback_retrieval"] is False
        # retrieve + generate (no fallback because score > threshold)
        assert len(result["steps"]) == 2

    def test_rerank_threshold_only(self):
        """Rerank threshold without answer grading: metadata only."""
        retriever, reranker, generator, _ = create_mock_components()
        reranked = [make_doc("A", "1", 0.5)]
        retriever.search.return_value = reranked
        reranker.rerank.return_value = reranked
        generator.generate.return_value = {"answer": "Answer"}

        pipeline = AgenticRAGPipeline(
            hybrid_retriever=retriever,
            reranker=reranker,
            generator=generator,
            enable_rerank_threshold=True,
            rerank_threshold=1.0,
        )

        result = pipeline.query("test")
        assert result["answer"] == "Answer"
        assert result["min_rerank_score"] == 0.5
        # No retry since answer grading is disabled
        assert result["retry_count"] == 0

    def test_web_search_only_config(self):
        """Web search only (no adaptive, no grading)."""
        retriever, reranker, generator, _ = create_mock_components()
        web_tool = MagicMock()
        web_tool.search.return_value = [
            {"content": "Web result", "source": "https://example.com"},
        ]

        reranked = [make_doc("A", "1", -6.0)]
        retriever.search.return_value = reranked
        reranker.rerank.return_value = reranked
        generator.generate.return_value = {"answer": "Web answer"}

        pipeline = AgenticRAGPipeline(
            hybrid_retriever=retriever,
            reranker=reranker,
            generator=generator,
            web_search_tool=web_tool,
            enable_web_search=True,
            web_search_threshold=-5.0,
        )

        result = pipeline.query("test")
        assert result["answer"] == "Web answer"
        assert result["used_web_search"] is True
        web_tool.search.assert_called_once()

    def test_web_search_skipped_when_good_scores(self):
        """Web search should be skipped when rerank scores are good."""
        retriever, reranker, generator, _ = create_mock_components()
        web_tool = MagicMock()

        reranked = [make_doc("A", "1", 3.0)]
        retriever.search.return_value = reranked
        reranker.rerank.return_value = reranked
        generator.generate.return_value = {"answer": "Local answer"}

        pipeline = AgenticRAGPipeline(
            hybrid_retriever=retriever,
            reranker=reranker,
            generator=generator,
            web_search_tool=web_tool,
            enable_web_search=True,
            web_search_threshold=-5.0,
        )

        result = pipeline.query("test")
        assert result["answer"] == "Local answer"
        assert result["used_web_search"] is False
        web_tool.search.assert_not_called()

    def test_adaptive_plus_web_search_config(self):
        """Adaptive + web search: both fallbacks chained."""
        retriever, reranker, generator, _ = create_mock_components()
        fallback = MagicMock()
        web_tool = MagicMock()
        web_tool.search.return_value = [
            {"content": "Web hit", "source": "https://example.com"},
        ]

        # Primary retrieval: very poor scores → triggers fallback
        primary_docs = [make_doc("A", "1", -8.0)]
        fallback_docs = [make_doc("F1", "f1", -6.0)]
        retriever.search.return_value = primary_docs
        fallback.search.return_value = fallback_docs
        # First rerank: primary docs, second rerank: fallback docs
        reranker.rerank.side_effect = [primary_docs, fallback_docs]
        generator.generate.return_value = {"answer": "Combined answer"}

        pipeline = AgenticRAGPipeline(
            hybrid_retriever=retriever,
            reranker=reranker,
            generator=generator,
            fallback_retriever=fallback,
            web_search_tool=web_tool,
            enable_adaptive_retrieval=True,
            enable_web_search=True,
            retrieval_threshold=0.0,
            web_search_threshold=-5.0,
        )

        result = pipeline.query("test")
        assert result["answer"] == "Combined answer"
        assert result["used_fallback_retrieval"] is True
        assert result["used_web_search"] is True
        web_tool.search.assert_called_once()

    def test_full_agentic_config(self):
        """Adaptive + answer grading: complete agentic pipeline."""
        retriever, reranker, generator, answer_grader = create_mock_components()
        fallback = MagicMock()

        reranked = [make_doc("A", "1", 2.0)]
        retriever.search.return_value = reranked
        reranker.rerank.return_value = reranked
        generator.generate.return_value = {"answer": "Full answer"}
        answer_grader.grade.return_value = True

        pipeline = AgenticRAGPipeline(
            hybrid_retriever=retriever,
            reranker=reranker,
            generator=generator,
            answer_grader=answer_grader,
            fallback_retriever=fallback,
            enable_adaptive_retrieval=True,
            retrieval_threshold=0.0,
            enable_answer_grading=True,
        )

        result = pipeline.query("test")
        assert result["answer"] == "Full answer"
        assert result["min_rerank_score"] == 2.0
        assert result["answer_is_acceptable"] is True
        assert result["used_fallback_retrieval"] is False


class TestRetryPrompt:
    """Tests for RETRY_PROMPT template."""

    def test_retry_prompt_has_placeholders(self):
        """RETRY_PROMPT must contain {context} and {question}."""
        assert "{context}" in AgenticRAGPipeline.RETRY_PROMPT
        assert "{question}" in AgenticRAGPipeline.RETRY_PROMPT

    def test_retry_prompt_mentions_precision(self):
        """RETRY_PROMPT should emphasize precision."""
        prompt_lower = AgenticRAGPipeline.RETRY_PROMPT.lower()
        assert "precise" in prompt_lower


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
