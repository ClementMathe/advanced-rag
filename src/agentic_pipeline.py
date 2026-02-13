"""
Agentic RAG pipeline with adaptive retrieval, web search fallback, and optional answer grading.

Workflow:
  retrieve → [check quality → fallback] → [check web → web search] → generate → [grade → retry] → END

This module provides AgenticRAGPipeline, a LangGraph-based workflow with
self-correction mechanisms (feature flags for ablation testing):

- Adaptive retrieval: if primary retrieval produces low rerank scores,
  falls back to an alternative retrieval strategy (wider candidate pool).
- Web search: if retrieval quality is still poor after local retrieval,
  supplements context with DuckDuckGo web search results.
- Answer grading: LLM checks if the generated answer addresses the query.
  On failure, retries generation with a stricter prompt.
"""

import operator
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph
from loguru import logger

from src.generator import LLMGenerator
from src.graders import AnswerGrader
from src.reranker import CrossEncoderReranker
from src.retriever import HybridRetriever
from src.utils import Timer
from src.web_search import DuckDuckGoSearchTool


class RAGState(TypedDict):
    """State that flows through the agentic RAG graph.

    Attributes:
        query: User's search query.
        documents: Retrieved and reranked documents.
        generation: Generated answer text.
        retry_count: Number of generation retries performed.
        min_rerank_score: Lowest rerank score among retrieved docs.
        used_fallback_retrieval: Whether fallback retrieval strategy was used.
        used_web_search: Whether web search was used to supplement context.
        answer_is_acceptable: Whether answer passed quality grading.
        intermediate_steps: Log of actions taken at each node (auto-appended).
    """

    query: str
    documents: List[Dict[str, Any]]
    generation: str
    retry_count: int
    min_rerank_score: float
    used_fallback_retrieval: bool
    used_web_search: bool
    answer_is_acceptable: bool
    intermediate_steps: Annotated[List[str], operator.add]


class AgenticRAGPipeline:
    """
    LangGraph-based agentic RAG pipeline with adaptive retrieval and web search.

    Workflow varies by enabled features:
    - Linear: retrieve -> generate -> END
    - Adaptive: retrieve -> check quality -> [generate | fallback -> generate] -> END
    - Web search: ... -> check web -> [generate | web search -> generate] -> END
    - With answer grading: ... -> generate -> grade -> [accept | retry] -> END

    Feature flags for ablation testing:
    - Adaptive retrieval: rerank scores trigger fallback to wider retrieval.
    - Web search: supplements context with DuckDuckGo results on poor retrieval.
    - Answer grading: LLM checks answer quality, retries with stricter prompt.

    Attributes:
        retriever: Primary HybridRetriever (dense-heavy by default).
        fallback_retriever: HybridRetriever for fallback (defaults to primary).
        reranker: CrossEncoderReranker for relevance-based reranking.
        generator: LLMGenerator for answer generation.
        answer_grader: AnswerGrader for answer quality checks (optional).
        web_search_tool: DuckDuckGoSearchTool for web search fallback (optional).
        k_retrieve: Number of documents after hybrid fusion.
        k_rerank: Number of documents after cross-encoder reranking.
        fallback_k_retrieve: Number of documents for fallback retrieval.
        max_retries: Maximum number of generation retries.
        enable_adaptive_retrieval: Whether to use adaptive retrieval fallback.
        retrieval_threshold: Minimum rerank score for acceptable retrieval.
        enable_web_search: Whether to use web search fallback.
        web_search_threshold: Minimum rerank score to skip web search.
        enable_answer_grading: Whether to use LLM answer quality checks.
        enable_rerank_threshold: Whether to use rerank score to skip retries.
        rerank_threshold: Minimum rerank score for retry decisions.
        app: Compiled LangGraph workflow.
    """

    RETRY_PROMPT = (
        "The previous answer was not satisfactory. "
        "Using ONLY the provided context, give a precise and factual answer.\n"
        "\n"
        "Rules:\n"
        "- Quote specific facts from the context\n"
        "- If the context doesn't clearly answer the question, say so\n"
        "- Be concise (1-2 sentences)\n"
        "\n"
        "Context:\n"
        "{context}\n"
        "\n"
        "Question: {question}\n"
        "\n"
        "Precise answer:"
    )

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        reranker: CrossEncoderReranker,
        generator: LLMGenerator,
        answer_grader: Optional[AnswerGrader] = None,
        fallback_retriever: Optional[HybridRetriever] = None,
        web_search_tool: Optional[DuckDuckGoSearchTool] = None,
        k_retrieve: int = 20,
        k_rerank: int = 5,
        fallback_k_retrieve: Optional[int] = None,
        max_retries: int = 1,
        enable_adaptive_retrieval: bool = False,
        retrieval_threshold: float = 0.0,
        enable_web_search: bool = False,
        web_search_threshold: float = -5.0,
        enable_answer_grading: bool = False,
        enable_rerank_threshold: bool = False,
        rerank_threshold: float = 0.0,
    ):
        """
        Initialize the agentic RAG pipeline.

        Args:
            hybrid_retriever: Primary HybridRetriever instance.
            reranker: CrossEncoderReranker instance for reranking.
            generator: LLMGenerator instance for answer generation.
            answer_grader: AnswerGrader instance (required if answer grading enabled).
            fallback_retriever: Alternative HybridRetriever for fallback strategy.
                Defaults to hybrid_retriever (wider retrieval with same strategy).
            web_search_tool: DuckDuckGoSearchTool for web search fallback.
                Created automatically if enable_web_search=True and not provided.
            k_retrieve: Number of documents to retrieve via hybrid search.
            k_rerank: Number of documents to keep after reranking.
            fallback_k_retrieve: Number of documents for fallback retrieval.
                Defaults to k_retrieve * 2 (wider candidate pool).
            max_retries: Maximum generation retry attempts (default 1).
            enable_adaptive_retrieval: Enable retrieval quality fallback.
            retrieval_threshold: Min rerank score for acceptable retrieval quality.
            enable_web_search: Enable web search fallback on poor retrieval.
            web_search_threshold: Min rerank score to skip web search (default -5.0).
            enable_answer_grading: Enable LLM answer quality checks.
            enable_rerank_threshold: Enable rerank score threshold for retry decisions.
            rerank_threshold: Min rerank score for retry decisions.
        """
        self.retriever = hybrid_retriever
        self.fallback_retriever = fallback_retriever or hybrid_retriever
        self.reranker = reranker
        self.generator = generator
        self.answer_grader = answer_grader
        self.k_retrieve = k_retrieve
        self.k_rerank = k_rerank
        self.fallback_k_retrieve = fallback_k_retrieve or k_retrieve * 2
        self.max_retries = max_retries
        self.enable_adaptive_retrieval = enable_adaptive_retrieval
        self.retrieval_threshold = retrieval_threshold
        self.enable_web_search = enable_web_search
        self.web_search_threshold = web_search_threshold
        self.enable_answer_grading = enable_answer_grading
        self.enable_rerank_threshold = enable_rerank_threshold
        self.rerank_threshold = rerank_threshold

        if enable_answer_grading and answer_grader is None:
            raise ValueError("answer_grader is required when enable_answer_grading=True")

        if enable_web_search:
            self.web_search_tool = web_search_tool or DuckDuckGoSearchTool()
        else:
            self.web_search_tool = web_search_tool

        self.app = self._build_graph()

        flags = []
        if enable_adaptive_retrieval:
            flags.append(
                f"adaptive_retrieval(threshold={retrieval_threshold}, "
                f"fallback_k={self.fallback_k_retrieve})"
            )
        if enable_web_search:
            flags.append(f"web_search(threshold={web_search_threshold})")
        if enable_answer_grading:
            flags.append("answer_grading")
        if enable_rerank_threshold:
            flags.append(f"rerank_threshold={rerank_threshold}")

        logger.info(
            f"AgenticRAGPipeline initialized: "
            f"k_retrieve={k_retrieve}, k_rerank={k_rerank}, "
            f"max_retries={max_retries}, "
            f"features=[{', '.join(flags) or 'none (linear)'}]"
        )

    def _build_graph(self):
        """Build LangGraph workflow based on enabled features.

        Graph structure varies by configuration:
        - Linear: retrieve -> generate -> END
        - Adaptive: retrieve -> decide_quality -> [generate | fallback -> generate]
        - Web search: ... -> check_web -> [generate | web_search -> generate]
        - With answer grading: ... -> generate -> grade_answer -> [END | generate]

        Returns:
            Compiled LangGraph application.
        """
        workflow = StateGraph(RAGState)

        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("generate", self._generate_node)

        workflow.set_entry_point("retrieve")

        # Determine node before generate (chain: retrieve → [adaptive] → [web] → generate)
        next_after_retrieval = "generate"

        if self.enable_web_search:
            workflow.add_node("check_web", self._check_web_node)
            workflow.add_node("web_search", self._web_search_node)
            workflow.add_conditional_edges(
                "check_web",
                self._decide_web_search,
                {
                    "skip_web": "generate",
                    "web_search": "web_search",
                },
            )
            workflow.add_edge("web_search", "generate")
            next_after_retrieval = "check_web"

        # Retrieval routing
        if self.enable_adaptive_retrieval:
            workflow.add_node("fallback_retrieve", self._fallback_retrieve_node)
            workflow.add_conditional_edges(
                "retrieve",
                self._decide_retrieval_quality,
                {
                    "acceptable": next_after_retrieval,
                    "fallback": "fallback_retrieve",
                },
            )
            workflow.add_edge("fallback_retrieve", next_after_retrieval)
        else:
            workflow.add_edge("retrieve", next_after_retrieval)

        # Generation routing
        if self.enable_answer_grading:
            workflow.add_node("grade_answer", self._grade_answer_node)
            workflow.add_edge("generate", "grade_answer")
            workflow.add_conditional_edges(
                "grade_answer",
                self._decide_after_grading,
                {
                    "accept": END,
                    "retry": "generate",
                },
            )
        else:
            workflow.add_edge("generate", END)

        return workflow.compile()

    def _retrieve_node(self, state: RAGState) -> dict:
        """Retrieve and rerank documents using primary strategy.

        Uses HybridRetriever (dense + BM25 with RRF fusion) followed by
        CrossEncoderReranker for precise relevance scoring. Computes
        min rerank score for retrieval quality assessment.

        Args:
            state: Current graph state with query.

        Returns:
            State update with retrieved documents, rerank score, and step log.
        """
        query = state["query"]

        with Timer("Hybrid retrieval"):
            candidates = self.retriever.search(query, k=self.k_retrieve, k_retriever=50)

        with Timer("Re-ranking"):
            reranked = self.reranker.rerank(query, candidates, top_k=self.k_rerank)

        min_score = min((doc.get("rerank_score", 0.0) for doc in reranked), default=0.0)

        step = f"Retrieved {len(reranked)} docs " f"(min_rerank={min_score:.2f})"
        logger.info(step)

        return {
            "documents": reranked,
            "min_rerank_score": min_score,
            "intermediate_steps": [step],
        }

    def _decide_retrieval_quality(self, state: RAGState) -> str:
        """Route based on retrieval quality (rerank scores).

        If the minimum rerank score is below the threshold, routes to
        fallback retrieval. Otherwise proceeds to generation.

        Args:
            state: Current graph state after primary retrieval.

        Returns:
            "acceptable" or "fallback" routing decision.
        """
        min_score = state.get("min_rerank_score", 0.0)

        if min_score >= self.retrieval_threshold:
            logger.info(
                f"Retrieval quality acceptable "
                f"(min_score={min_score:.2f} >= "
                f"threshold={self.retrieval_threshold})"
            )
            return "acceptable"

        logger.info(
            f"Retrieval quality poor, trying fallback "
            f"(min_score={min_score:.2f} < "
            f"threshold={self.retrieval_threshold})"
        )
        return "fallback"

    def _fallback_retrieve_node(self, state: RAGState) -> dict:
        """Retrieve and rerank using fallback strategy (wider retrieval).

        Called when primary retrieval produces low rerank scores.
        Uses fallback_k_retrieve (default 2x primary) for a larger
        candidate pool, giving the reranker more documents to choose from.

        Args:
            state: Current graph state with query.

        Returns:
            State update with new documents, rerank score, and step log.
        """
        query = state["query"]

        with Timer("Fallback retrieval"):
            candidates = self.fallback_retriever.search(
                query, k=self.fallback_k_retrieve, k_retriever=100
            )

        with Timer("Fallback re-ranking"):
            reranked = self.reranker.rerank(query, candidates, top_k=self.k_rerank)

        min_score = min((doc.get("rerank_score", 0.0) for doc in reranked), default=0.0)

        step = (
            f"Fallback retrieved {len(reranked)} docs "
            f"(k={self.fallback_k_retrieve}, min_rerank={min_score:.2f})"
        )
        logger.info(step)

        return {
            "documents": reranked,
            "min_rerank_score": min_score,
            "used_fallback_retrieval": True,
            "intermediate_steps": [step],
        }

    def _check_web_node(self, state: RAGState) -> dict:
        """Passthrough node for web search routing convergence.

        Exists as a convergence point after primary/fallback retrieval
        where the web search decision is made via conditional edges.

        Returns:
            Empty dict (no state updates).
        """
        return {}

    def _decide_web_search(self, state: RAGState) -> str:
        """Route to web search or skip based on retrieval quality.

        If the minimum rerank score is below web_search_threshold,
        routes to web search to supplement local retrieval context.

        Args:
            state: Current graph state after local retrieval.

        Returns:
            "skip_web" or "web_search" routing decision.
        """
        min_score = state.get("min_rerank_score", float("inf"))

        if min_score >= self.web_search_threshold:
            logger.info(
                f"Retrieval sufficient, skipping web search "
                f"(min_score={min_score:.2f} >= "
                f"threshold={self.web_search_threshold})"
            )
            return "skip_web"

        logger.info(
            f"Retrieval quality poor, triggering web search "
            f"(min_score={min_score:.2f} < "
            f"threshold={self.web_search_threshold})"
        )
        return "web_search"

    def _web_search_node(self, state: RAGState) -> dict:
        """Search the web and merge results with local documents.

        Called when local retrieval (including fallback) produces poor
        rerank scores. Uses DuckDuckGoSearchTool to find additional
        context, then merges web results with existing local documents.

        Args:
            state: Current graph state with query and documents.

        Returns:
            State update with merged documents, web search flag, and step log.
        """
        query = state["query"]
        existing_docs = state.get("documents", [])

        with Timer("Web search"):
            web_docs = self.web_search_tool.search(query)

        merged = existing_docs + web_docs

        step = f"Web search returned {len(web_docs)} results"
        logger.info(step)

        return {
            "documents": merged,
            "used_web_search": True,
            "intermediate_steps": [step],
        }

    def _generate_node(self, state: RAGState) -> dict:
        """Generate answer from retrieved documents.

        On first attempt, uses the standard generation prompt.
        On retry (retry_count > 0), uses a stricter prompt that
        emphasizes precision and factual grounding.

        Args:
            state: Current graph state with documents.

        Returns:
            State update with generated answer and step log.
        """
        query = state["query"]
        docs = state.get("documents", [])
        retry_count = state.get("retry_count", 0)

        if retry_count > 0:
            context = self.generator._format_context(docs[:5])
            prompt = self.RETRY_PROMPT.format(context=context, question=query)

            with Timer(f"Generation (retry {retry_count})"):
                raw_answer = self.generator._generate_text(prompt)
            answer = self.generator._parse_answer(raw_answer)
        else:
            with Timer("Generation"):
                result = self.generator.generate(query, docs, max_chunks=5)
            answer = result["answer"]

        step = f"Generated answer (attempt {retry_count + 1})"
        logger.info(step)

        return {
            "generation": answer,
            "intermediate_steps": [step],
        }

    def _grade_answer_node(self, state: RAGState) -> dict:
        """Grade the generated answer for quality.

        Uses the AnswerGrader to check if the answer correctly addresses
        the query based on the retrieved context. Increments retry_count
        when the answer is not acceptable.

        Args:
            state: Current graph state with generation and documents.

        Returns:
            State update with grade result and optional retry count increment.
        """
        query = state["query"]
        answer = state.get("generation", "")
        docs = state.get("documents", [])

        doc_contents = [doc.get("content", "") for doc in docs[:5]]

        with Timer("Answer grading"):
            is_acceptable = self.answer_grader.grade(query, answer, doc_contents)

        step = f"Answer grade: " f"{'acceptable' if is_acceptable else 'not acceptable'}"
        logger.info(step)

        update = {
            "answer_is_acceptable": is_acceptable,
            "intermediate_steps": [step],
        }

        if not is_acceptable:
            update["retry_count"] = state.get("retry_count", 0) + 1

        return update

    def _decide_after_grading(self, state: RAGState) -> str:
        """Route to accept or retry based on answer quality.

        Decision rules (in order):
        1. Answer is acceptable -> accept
        2. Max retries reached -> accept (use best available answer)
        3. Rerank threshold enabled and retrieval is poor -> accept
           (retrying won't help with bad documents)
        4. Otherwise -> retry generation with stricter prompt

        Args:
            state: Current graph state after answer grading.

        Returns:
            "accept" or "retry" routing decision.
        """
        if state.get("answer_is_acceptable", True):
            logger.info("Routing -> accept (answer is acceptable)")
            return "accept"

        retry_count = state.get("retry_count", 0)
        if retry_count > self.max_retries:
            logger.info(f"Routing -> accept (max retries {self.max_retries} reached)")
            return "accept"

        # Don't retry if retrieval quality is low (regenerating won't help)
        if self.enable_rerank_threshold:
            min_score = state.get("min_rerank_score", float("inf"))
            if min_score < self.rerank_threshold:
                logger.info(
                    f"Routing -> accept (low retrieval quality: "
                    f"min_score={min_score:.2f} < "
                    f"threshold={self.rerank_threshold})"
                )
                return "accept"

        logger.info(f"Routing -> retry (attempt {retry_count})")
        return "retry"

    def query(self, query: str) -> Dict[str, Any]:
        """Run the agentic pipeline on a query.

        Args:
            query: User's search query.

        Returns:
            Dictionary with:
                - query: Original query
                - answer: Generated answer
                - steps: List of intermediate steps taken
                - num_docs_retrieved: Total docs after reranking
                - context_documents: Documents used for generation
                - retry_count: Number of generation retries performed
                - min_rerank_score: Lowest rerank score (retrieval quality)
                - used_fallback_retrieval: Whether fallback strategy was used
                - used_web_search: Whether web search was used
                - answer_is_acceptable: Whether final answer passed grading
        """
        initial_state: RAGState = {
            "query": query,
            "documents": [],
            "generation": "",
            "retry_count": 0,
            "min_rerank_score": 0.0,
            "used_fallback_retrieval": False,
            "used_web_search": False,
            "answer_is_acceptable": True,
            "intermediate_steps": [],
        }

        logger.info(f"{'='*60}")
        logger.info(f"Agentic RAG query: '{query}'")
        logger.info(f"{'='*60}")

        with Timer("Agentic pipeline total"):
            final_state = self.app.invoke(initial_state)

        logger.info(f"Steps: {' -> '.join(final_state['intermediate_steps'])}")

        gen_docs = final_state["documents"][:5]

        return {
            "query": query,
            "answer": final_state["generation"],
            "steps": final_state["intermediate_steps"],
            "num_docs_retrieved": len(final_state["documents"]),
            "context_documents": gen_docs,
            "retry_count": final_state.get("retry_count", 0),
            "min_rerank_score": final_state.get("min_rerank_score", 0.0),
            "used_fallback_retrieval": final_state.get("used_fallback_retrieval", False),
            "used_web_search": final_state.get("used_web_search", False),
            "answer_is_acceptable": final_state.get("answer_is_acceptable", True),
        }
