"""
Agentic RAG pipeline using LangGraph.

Self-correcting workflow: retrieve -> grade -> [generate | rewrite -> retry].

This module provides AgenticRAGPipeline, a LangGraph-based workflow that adds
document relevance grading between retrieval and generation. When too few
documents pass grading (<3), the pipeline rewrites the query and retries
retrieval (up to 3 times) before generating an answer.
"""

import operator
from typing import Annotated, Any, Dict, List, TypedDict

from langgraph.graph import END, StateGraph
from loguru import logger

from src.generator import LLMGenerator
from src.graders import DocumentGrader, QueryRewriter
from src.reranker import CrossEncoderReranker
from src.retriever import HybridRetriever
from src.utils import Timer


class RAGState(TypedDict):
    """State that flows through the agentic RAG graph.

    Attributes:
        query: Original user query (never modified).
        rewritten_query: LLM-rewritten query for retry retrieval.
        query_history: History of rewritten queries (auto-appended).
        documents: Retrieved and reranked documents (replaced each cycle).
        graded_documents: Documents that passed relevance grading.
        document_grades: Raw boolean grades from batch grading.
        generation: Final generated answer.
        retry_count: Number of query rewrites performed so far.
        intermediate_steps: Log of actions taken at each node (auto-appended).
    """

    query: str
    rewritten_query: str
    query_history: Annotated[List[str], operator.add]
    documents: List[Dict[str, Any]]
    graded_documents: List[Dict[str, Any]]
    document_grades: List[bool]
    generation: str
    retry_count: int
    intermediate_steps: Annotated[List[str], operator.add]


class AgenticRAGPipeline:
    """
    LangGraph-based agentic RAG pipeline with self-correction.

    Workflow: retrieve -> grade -> [generate | rewrite -> retrieve (retry)].

    When fewer than 3 documents pass relevance grading, the pipeline
    rewrites the query and retries retrieval. This self-correction loop
    runs up to 3 times before falling back to generation with whatever
    documents are available.

    Attributes:
        retriever: HybridRetriever for document retrieval.
        reranker: CrossEncoderReranker for relevance-based reranking.
        generator: LLMGenerator for answer generation.
        grader: DocumentGrader for relevance filtering.
        query_rewriter: QueryRewriter for improving queries on retry.
        k_retrieve: Number of documents after hybrid fusion.
        k_rerank: Number of documents after cross-encoder reranking.
        min_relevant: Minimum relevant docs to proceed to generation.
        max_retries: Maximum number of query rewrite attempts.
        app: Compiled LangGraph workflow.
    """

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        reranker: CrossEncoderReranker,
        generator: LLMGenerator,
        grader: DocumentGrader,
        query_rewriter: QueryRewriter,
        k_retrieve: int = 20,
        k_rerank: int = 5,
        min_relevant: int = 3,
        max_retries: int = 3,
    ):
        """
        Initialize the agentic RAG pipeline.

        Args:
            hybrid_retriever: HybridRetriever instance for document retrieval.
            reranker: CrossEncoderReranker instance for reranking.
            generator: LLMGenerator instance for answer generation.
            grader: DocumentGrader instance for relevance grading.
            query_rewriter: QueryRewriter instance for query improvement.
            k_retrieve: Number of documents to retrieve via hybrid search.
            k_rerank: Number of documents to keep after reranking.
            min_relevant: Minimum relevant docs to skip rewriting (default 3).
            max_retries: Maximum query rewrite attempts (default 3).
        """
        self.retriever = hybrid_retriever
        self.reranker = reranker
        self.generator = generator
        self.grader = grader
        self.query_rewriter = query_rewriter
        self.k_retrieve = k_retrieve
        self.k_rerank = k_rerank
        self.min_relevant = min_relevant
        self.max_retries = max_retries

        self.app = self._build_graph()

        logger.info(
            f"AgenticRAGPipeline initialized: "
            f"k_retrieve={k_retrieve}, k_rerank={k_rerank}, "
            f"min_relevant={min_relevant}, max_retries={max_retries}"
        )

    def _build_graph(self):
        """Construct the LangGraph workflow with self-correction.

        Graph:
            retrieve -> grade_documents -> [generate | rewrite_query -> retrieve]

        Returns:
            Compiled LangGraph application.
        """
        workflow = StateGraph(RAGState)

        # Add nodes
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("grade_documents", self._grade_documents_node)
        workflow.add_node("generate", self._generate_node)
        workflow.add_node("rewrite_query", self._rewrite_query_node)

        # Edges
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "grade_documents")

        # Conditional: generate if enough relevant docs, else rewrite
        workflow.add_conditional_edges(
            "grade_documents",
            self._decide_to_generate,
            {
                "generate": "generate",
                "rewrite": "rewrite_query",
            },
        )

        workflow.add_edge("rewrite_query", "retrieve")  # Loop back
        workflow.add_edge("generate", END)

        return workflow.compile()

    def _retrieve_node(self, state: RAGState) -> dict:
        """Retrieve and rerank documents.

        Uses HybridRetriever (dense + BM25 with RRF fusion) followed by
        CrossEncoderReranker for precise relevance scoring. On retry,
        uses the rewritten query instead of the original.

        Args:
            state: Current graph state with query.

        Returns:
            State update with retrieved documents and step log.
        """
        query = state.get("rewritten_query") or state["query"]

        with Timer("Hybrid retrieval"):
            candidates = self.retriever.search(query, k=self.k_retrieve, k_retriever=50)

        with Timer("Re-ranking"):
            reranked = self.reranker.rerank(query, candidates, top_k=self.k_rerank)

        step = f"Retrieved {len(reranked)} docs for: '{query}'"
        logger.info(step)

        return {
            "documents": reranked,
            "intermediate_steps": [step],
        }

    def _grade_documents_node(self, state: RAGState) -> dict:
        """Grade documents for relevance using LLM.

        Uses batch grading for efficiency (single LLM call for all docs).
        Documents that fail grading are filtered out.

        Args:
            state: Current graph state with documents.

        Returns:
            State update with graded documents and step log.
        """
        query = state["query"]
        documents = state["documents"]

        doc_contents = [doc["content"] for doc in documents]

        with Timer("Document grading"):
            grades = self.grader.grade_batch(query, doc_contents)

        graded = [doc for doc, is_relevant in zip(documents, grades) if is_relevant]

        step = f"Graded: {len(graded)}/{len(documents)} relevant"
        logger.info(step)

        return {
            "graded_documents": graded,
            "document_grades": grades,
            "intermediate_steps": [step],
        }

    def _decide_to_generate(self, state: RAGState) -> str:
        """Route to generate or rewrite based on graded document quality.

        Decision rules (in order):
        1. Enough relevant docs (>= min_relevant) -> generate
        2. Max retries reached -> generate (with whatever docs available)
        3. Query unchanged after rewrite (loop detection) -> generate
        4. Otherwise -> rewrite and retry

        Args:
            state: Current graph state after grading.

        Returns:
            "generate" or "rewrite" routing decision.
        """
        graded_docs = state.get("graded_documents", [])
        retry_count = state.get("retry_count", 0)
        query_history = state.get("query_history", [])

        # Rule 1: Enough relevant docs
        if len(graded_docs) >= self.min_relevant:
            logger.info(
                f"Routing -> generate ({len(graded_docs)} relevant "
                f">= {self.min_relevant} threshold)"
            )
            return "generate"

        # Rule 2: Max retries reached
        if retry_count >= self.max_retries:
            logger.info(f"Routing -> generate (max retries {self.max_retries} reached)")
            return "generate"

        # Rule 3: Query unchanged (loop detection)
        if len(query_history) >= 2 and query_history[-1] == query_history[-2]:
            logger.info("Routing -> generate (query unchanged after rewrite)")
            return "generate"

        # Otherwise: rewrite
        logger.info(
            f"Routing -> rewrite ({len(graded_docs)} relevant "
            f"< {self.min_relevant}, retry {retry_count + 1})"
        )
        return "rewrite"

    def _rewrite_query_node(self, state: RAGState) -> dict:
        """Rewrite the query to improve retrieval on retry.

        Always rewrites from the original query to prevent compound drift
        (each retry gets a fresh take rather than rewriting a rewrite).

        Args:
            state: Current graph state with grading results.

        Returns:
            State update with rewritten query, updated history, and retry count.
        """
        original_query = state["query"]
        num_relevant = len(state.get("graded_documents", []))
        num_total = len(state.get("documents", []))

        with Timer("Query rewriting"):
            new_query = self.query_rewriter.rewrite(original_query, num_total, num_relevant)

        step = f"Rewrote: '{original_query}' -> '{new_query}'"
        logger.info(step)

        return {
            "rewritten_query": new_query,
            "query_history": [new_query],
            "retry_count": state.get("retry_count", 0) + 1,
            "intermediate_steps": [step],
        }

    def _generate_node(self, state: RAGState) -> dict:
        """Generate answer from all reranked documents.

        Uses the original query for generation. Grading is used only for
        the retry decision, not for filtering the generation context —
        the reranker already selected the most relevant docs.

        Args:
            state: Current graph state with documents.

        Returns:
            State update with generated answer and step log.
        """
        query = state["query"]
        docs = state.get("documents", [])

        num_graded = len(state.get("graded_documents", []))

        with Timer("Generation"):
            result = self.generator.generate(query, docs, max_chunks=5)

        step = f"Generated answer using {len(docs)} docs ({num_graded} graded relevant)"
        logger.info(step)

        return {
            "generation": result["answer"],
            "intermediate_steps": [step],
        }

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
                - num_docs_graded: Docs that passed relevance grading
                - context_documents: Documents actually used for generation
                - retry_count: Number of query rewrites performed
        """
        initial_state: RAGState = {
            "query": query,
            "rewritten_query": "",
            "query_history": [],
            "documents": [],
            "graded_documents": [],
            "document_grades": [],
            "generation": "",
            "retry_count": 0,
            "intermediate_steps": [],
        }

        logger.info(f"{'='*60}")
        logger.info(f"Agentic RAG query: '{query}'")
        logger.info(f"{'='*60}")

        with Timer("Agentic pipeline total"):
            final_state = self.app.invoke(initial_state)

        logger.info(f"Steps: {' -> '.join(final_state['intermediate_steps'])}")

        # Reconstruct docs used for generation (same logic as _generate_node)
        gen_docs = final_state["documents"][:5]  # max_chunks=5 in generate

        return {
            "query": query,
            "answer": final_state["generation"],
            "steps": final_state["intermediate_steps"],
            "num_docs_retrieved": len(final_state["documents"]),
            "num_docs_graded": len(final_state["graded_documents"]),
            "context_documents": gen_docs,
            "retry_count": final_state.get("retry_count", 0),
        }
