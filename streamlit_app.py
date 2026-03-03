"""
Advanced RAG Demo: Streamlit app.

Supports three pipeline modes (Linear / Adaptive / Agentic) and three
generator options (Qwen local, Mistral Small, Mistral Large).

Usage:
    streamlit run streamlit_app.py

Prerequisites:
    - Build the SQuAD index first:
        python scripts/prepare_squad.py
        python scripts/build_index.py
    - For Mistral generators: set MISTRAL_API_KEY env var or enter it in the sidebar.
"""

import os
import re
import time
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st
import torch

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

INDEX_PATH = "index/squad"

# USD per 1M tokens (source: mistral.ai pricing, 2025)
PRICING = {
    "mistral-small-latest": {"input": 0.20, "output": 0.60},
    "mistral-large-latest": {"input": 2.00, "output": 6.00},
}

GENERATOR_TO_MODEL = {
    "Mistral Small": "mistral-small-latest",
    "Mistral Large": "mistral-large-latest",
}

# ─────────────────────────────────────────────────────────────────────────────
# Page config (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Advanced RAG Demo",
    page_icon="🔍",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# Index availability check
# ─────────────────────────────────────────────────────────────────────────────

if not Path(INDEX_PATH).exists():
    st.error(
        "**SQuAD index not found.** Build it first:\n\n"
        "```bash\n"
        "python scripts/prepare_squad.py\n"
        "python scripts/build_index.py\n"
        "```"
    )
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Cached loaders — heavy components loaded once per session
# ─────────────────────────────────────────────────────────────────────────────


@st.cache_resource(show_spinner="Loading index & retriever (first run only)...")
def load_retriever(index_path: str):
    """Load FAISS index, rebuild BM25, build HybridRetriever."""
    from src.chunking import Chunk
    from src.embeddings import EmbeddingModel, FAISSIndex
    from src.retriever import BM25Retriever, DenseRetriever, HybridRetriever

    device = "cuda" if torch.cuda.is_available() else "cpu"

    faiss_index = FAISSIndex.load(index_path)

    chunks = [
        Chunk(
            content=m["content"],
            chunk_id=m["chunk_id"],
            doc_id=m["doc_id"],
            start_char=0,
            end_char=len(m["content"]),
            chunk_index=m["chunk_index"],
            metadata=m.get("metadata", {}),
        )
        for m in faiss_index.chunk_metadata
    ]

    bm25 = BM25Retriever()
    bm25.index(chunks)

    embed_model = EmbeddingModel(model_name="BAAI/bge-large-en-v1.5", device=device)
    dense = DenseRetriever(faiss_index, embed_model)
    return HybridRetriever(dense, bm25, dense_weight=0.9, sparse_weight=0.1)


@st.cache_resource(show_spinner="Loading reranker...")
def load_reranker():
    """Load cross-encoder reranker."""
    from src.reranker import CrossEncoderReranker

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return CrossEncoderReranker(
        model_name="cross-encoder/ms-marco-MiniLM-L6-v2",
        device=device,
    )


@st.cache_resource(show_spinner="Loading Qwen 2.5-3B — this may take ~2 min...")
def load_qwen():
    """Load Qwen2.5-3B-Instruct with 4-bit quantization (GPU) or fp32 (CPU)."""
    from src.generator import LLMGenerator

    return LLMGenerator(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        load_in_4bit=torch.cuda.is_available(),
        temperature=0.1,
        max_new_tokens=80,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline builder — called at query time (cheap: components come from cache)
# ─────────────────────────────────────────────────────────────────────────────


def build_pipeline(
    mode: str,
    generator_choice: str,
    api_key: str,
    enable_doc_grading: bool,
):
    """Assemble pipeline from cached components and current sidebar settings."""
    from src.agentic_pipeline import AgenticRAGPipeline
    from src.pipeline import RAGPipeline

    retriever = load_retriever(INDEX_PATH)
    reranker = load_reranker()

    # ── Generator ─────────────────────────────────────────────────────────────
    if generator_choice == "Qwen 2.5-3B (local)":
        generator = load_qwen()
    else:
        from src.mistral_generator import MistralAPIGenerator

        model_name = GENERATOR_TO_MODEL[generator_choice]
        generator = MistralAPIGenerator(model_name=model_name, api_key=api_key)

    # ── Linear pipeline ───────────────────────────────────────────────────────
    if mode == "Linear":
        return RAGPipeline(retriever, reranker, generator)

    # ── Adaptive / Agentic ────────────────────────────────────────────────────
    kwargs = dict(
        hybrid_retriever=retriever,
        reranker=reranker,
        generator=generator,
        enable_adaptive_retrieval=True,
        k_retrieve=20,
        k_rerank=5,
    )

    if mode == "Agentic":
        if generator_choice == "Qwen 2.5-3B (local)":
            from src.graders import AnswerGrader

            kwargs["answer_grader"] = AnswerGrader(generator)
        else:
            from mistralai import Mistral

            from src.mistral_grader import MistralAnswerGrader

            kwargs["answer_grader"] = MistralAnswerGrader(
                Mistral(api_key=api_key), "mistral-small-latest"
            )
        kwargs["enable_answer_grading"] = True

        if enable_doc_grading and api_key:
            from mistralai import Mistral

            from src.mistral_grader import MistralDocumentGrader

            client = Mistral(api_key=api_key)
            kwargs["doc_grader"] = MistralDocumentGrader(
                client=client, model_name="mistral-small-latest"
            )
            kwargs["enable_doc_grading"] = True

    return AgenticRAGPipeline(**kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: cost estimation
# ─────────────────────────────────────────────────────────────────────────────


def estimate_cost(query: str, answer: str, model_name: str) -> tuple[int, int, float]:
    """Rough cost estimate: 1 token ~= 4 characters."""
    input_chars = len(query) + 5 * 500  # context prompt ~= 5 chunks * 500 chars
    output_chars = len(answer)
    input_tok = input_chars // 4
    output_tok = output_chars // 4
    prices = PRICING.get(model_name, PRICING["mistral-small-latest"])
    cost = (input_tok * prices["input"] + output_tok * prices["output"]) / 1_000_000
    return input_tok, output_tok, cost


# ─────────────────────────────────────────────────────────────────────────────
# Helper: answer highlighting
# ─────────────────────────────────────────────────────────────────────────────


def highlight_answer_in_text(chunk_text: str, answer: str) -> str:
    """Wrap answer words (4+ chars) found in chunk_text with <mark> tags."""
    answer_words = {
        w.lower()
        for w in re.findall(r"\b\w{4,}\b", answer)
        if w.lower()
        not in {
            "that",
            "this",
            "with",
            "from",
            "have",
            "were",
            "they",
            "been",
            "which",
            "their",
            "also",
            "more",
            "than",
            "into",
            "when",
            "what",
            "will",
            "some",
            "time",
            "would",
            "there",
        }
    }
    if not answer_words:
        return chunk_text

    def maybe_mark(match: re.Match) -> str:
        word = match.group(0)
        if word.lower() in answer_words:
            return (
                f'<mark style="background:#ffd166;padding:1px 3px;'
                f'border-radius:3px;font-weight:500">{word}</mark>'
            )
        return word

    return re.sub(r"\b\w+\b", maybe_mark, chunk_text)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Plotly charts
# ─────────────────────────────────────────────────────────────────────────────

_BLUE_PALETTE = [
    "#1565C0",
    "#1976D2",
    "#1E88E5",
    "#2196F3",
    "#42A5F5",
]
_CHART_LAYOUT = dict(
    margin=dict(l=10, r=10, t=40, b=10),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(size=12),
)


def plot_rerank_scores(docs: list) -> go.Figure:
    """Horizontal bar chart of cross-encoder rerank scores."""
    labels = [f"Doc {i + 1} · {str(d.get('doc_id', ''))[-14:]}" for i, d in enumerate(docs)]
    scores = [d.get("rerank_score", d.get("score", 0)) or 0 for d in docs]

    fig = go.Figure(
        go.Bar(
            x=scores,
            y=labels,
            orientation="h",
            marker=dict(
                color=_BLUE_PALETTE[: len(docs)],
                line=dict(width=0),
            ),
            text=[f"{s:.3f}" for s in scores],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Cross-encoder rerank scores",
        xaxis=dict(title="Score", range=[min(0, min(scores)) - 0.05, max(scores) + 0.1]),
        yaxis=dict(autorange="reversed"),
        height=220,
        **_CHART_LAYOUT,
    )
    return fig


def plot_latency_pie(timings: dict) -> go.Figure:
    """Donut chart breaking down pipeline latency."""
    labels, values, colors = [], [], []
    color_map = {
        "retrieval_ms": "#42A5F5",
        "reranking_ms": "#FF9800",
        "generation_ms": "#4CAF50",
    }
    display_names = {
        "retrieval_ms": "Retrieval",
        "reranking_ms": "Reranking",
        "generation_ms": "Generation",
    }
    for key, color in color_map.items():
        if key in timings and timings[key] > 0:
            labels.append(display_names[key])
            values.append(round(timings[key]))
            colors.append(color)

    if not values:
        return go.Figure()

    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.45,
            marker=dict(colors=colors, line=dict(color="white", width=2)),
            textinfo="label+percent",
            hovertemplate="%{label}: %{value} ms<extra></extra>",
        )
    )
    total = sum(values)
    fig.update_layout(
        title="Latency breakdown",
        annotations=[dict(text=f"{total} ms", x=0.5, y=0.5, showarrow=False, font_size=14)],
        height=220,
        showlegend=False,
        **_CHART_LAYOUT,
    )
    return fig


def render_reranking_impact(pre_docs: list, post_docs: list) -> None:
    """Show rank changes from hybrid retrieval to cross-encoder reranking."""

    def chunk_key(d: dict) -> str:
        return str(d.get("doc_id", "")) + "|" + str(d.get("chunk_index", d.get("content", "")[:30]))

    pre_keys = [chunk_key(d) for d in pre_docs]

    rows = []
    for post_rank, doc in enumerate(post_docs):
        key = chunk_key(doc)
        try:
            pre_rank = pre_keys.index(key) + 1  # 1-based
        except ValueError:
            pre_rank = None

        score = doc.get("rerank_score")
        short_id = str(doc.get("doc_id", f"doc_{post_rank + 1}"))[-18:]
        rows.append((post_rank + 1, pre_rank, score, short_id))

    # ── Slope / dumbbell chart ─────────────────────────────────────────────
    fig = go.Figure()
    for ce_rank, pre_rank, score, _short_id in rows:
        if pre_rank is None:
            continue
        delta = pre_rank - ce_rank  # positive = improved (was further back, now closer to #1)
        color = "#4CAF50" if delta > 0 else ("#F44336" if delta < 0 else "#9E9E9E")
        label = f"Doc {ce_rank}"

        # Connecting line
        fig.add_trace(
            go.Scatter(
                x=[pre_rank, ce_rank],
                y=[label, label],
                mode="lines",
                line=dict(color=color, width=2.5),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        # Hybrid rank dot
        fig.add_trace(
            go.Scatter(
                x=[pre_rank],
                y=[label],
                mode="markers+text",
                marker=dict(size=18, color="#90CAF9", line=dict(width=2, color="#1565C0")),
                text=[f"#{pre_rank}"],
                textfont=dict(color="#1565C0", size=11),
                textposition="middle right",
                showlegend=False,
                hovertemplate=f"{label}: hybrid rank #{pre_rank}<extra></extra>",
            )
        )
        # CE rank dot
        fig.add_trace(
            go.Scatter(
                x=[ce_rank],
                y=[label],
                mode="markers+text",
                marker=dict(size=18, color=color, line=dict(width=2, color=color)),
                text=[f"#{ce_rank}"],
                textfont=dict(color=color, size=11),
                textposition="middle left",
                showlegend=False,
                hovertemplate=(
                    f"{label}: CE rank #{ce_rank}"
                    + (f" (score {score:.3f})" if score is not None else "")
                    + "<extra></extra>"
                ),
            )
        )

    # Invisible legend traces
    for label, color in [
        ("Hybrid rank", "#90CAF9"),
        ("Improved ↑", "#4CAF50"),
        ("Declined ↓", "#F44336"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=12, color=color),
                name=label,
                showlegend=True,
            )
        )

    max_rank = max(
        max((r[0] for r in rows), default=5),
        max((r[1] for r in rows if r[1] is not None), default=5),
    )
    fig.update_layout(
        xaxis=dict(
            title="Rank position  (← better  ·  worse →)",
            dtick=1,
            range=[0.5, max_rank + 0.5],
        ),
        yaxis=dict(title="", autorange="reversed"),
        height=240,
        legend=dict(orientation="h", yanchor="bottom", y=-0.35),
        **_CHART_LAYOUT,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Readable summary ───────────────────────────────────────────────────
    for ce_rank, pre_rank, score, short_id in rows:
        if pre_rank is None:
            continue
        delta = pre_rank - ce_rank
        if delta > 0:
            arrow, color_css = f"↑ +{delta}", "color:#2e7d32;font-weight:600"
        elif delta < 0:
            arrow, color_css = f"↓ {delta}", "color:#c62828;font-weight:600"
        else:
            arrow, color_css = "→ unchanged", "color:#666"
        score_str = f" · score `{score:.3f}`" if score is not None else ""
        st.markdown(
            f"**Doc {ce_rank}** `{short_id}` — "
            f"hybrid rank **#{pre_rank}** → CE rank **#{ce_rank}** "
            f'<span style="{color_css}">{arrow}</span>{score_str}',
            unsafe_allow_html=True,
        )


def render_faithfulness_metrics(docs: list, answer: str) -> None:
    """Show local faithfulness proxy metrics."""
    if not docs or not answer:
        return

    answer_words = set(re.findall(r"\b\w{4,}\b", answer.lower()))

    chunk_overlaps = []
    for doc in docs:
        content = doc.get("content", "")
        content_words = set(re.findall(r"\b\w{4,}\b", content.lower()))
        overlap = len(answer_words & content_words) / max(len(answer_words), 1)
        chunk_overlaps.append(overlap)

    # Faithfulness proxy: % of answer words found in ANY chunk
    all_context_words = set()
    for doc in docs:
        all_context_words |= set(re.findall(r"\b\w{4,}\b", doc.get("content", "").lower()))
    grounded_words = answer_words & all_context_words
    faithfulness = len(grounded_words) / max(len(answer_words), 1)

    # Context utilization: how many chunks actually contributed
    contributing = sum(1 for o in chunk_overlaps if o > 0.05)

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Faithfulness proxy",
        f"{faithfulness:.0%}",
        help="% of answer words (4+ chars) found in the retrieved context. "
        "High = answer is grounded in retrieved docs.",
    )
    col2.metric(
        "Context utilization",
        f"{contributing}/{len(docs)} chunks",
        help="Number of retrieved chunks that share at least 5% vocabulary with the answer.",
    )
    col3.metric(
        "Answer length",
        f"{len(answer.split())} words",
        help="Number of words in the generated answer.",
    )

    # Per-chunk overlap mini-bar
    overlap_fig = go.Figure(
        go.Bar(
            x=[f"Doc {i + 1}" for i in range(len(docs))],
            y=[round(o * 100, 1) for o in chunk_overlaps],
            marker=dict(
                color=[f"rgba(33,150,243,{0.4 + 0.6 * o})" for o in chunk_overlaps],
                line=dict(width=0),
            ),
            text=[f"{o:.0%}" for o in chunk_overlaps],
            textposition="outside",
        )
    )
    overlap_fig.update_layout(
        title="Answer word overlap per chunk",
        yaxis=dict(title="Overlap (%)", range=[0, 110]),
        height=200,
        **_CHART_LAYOUT,
    )
    st.plotly_chart(overlap_fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("Configuration")

    st.subheader("Pipeline Mode")
    mode = st.radio(
        "Pipeline Mode",
        ["Linear", "Adaptive", "Agentic"],
        captions=[
            "Retrieve → Rerank → Generate",
            "+ confidence-based adaptive retrieval",
            "+ answer grading & doc filtering",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    st.subheader("Generator")
    generator_choice = st.radio(
        "Generator",
        ["Qwen 2.5-3B (local)", "Mistral Small", "Mistral Large"],
        captions=[
            "No API key · ~2 GB VRAM · ~20 s/query",
            "317 ms avg · $0.20/1M input tokens",
            "Highest quality · $2.00/1M input tokens",
        ],
        label_visibility="collapsed",
    )

    api_key = ""
    if generator_choice != "Qwen 2.5-3B (local)":
        st.subheader("Mistral API Key")
        api_key = st.text_input(
            "Mistral API Key",
            type="password",
            value=os.getenv("MISTRAL_API_KEY", ""),
            placeholder="Enter key or set MISTRAL_API_KEY",
            label_visibility="collapsed",
        )
        if not api_key:
            st.warning("API key required for Mistral generators.")

    st.divider()

    enable_doc_grading = False
    if mode == "Agentic":
        st.subheader("Agentic Options")
        if generator_choice != "Qwen 2.5-3B (local)":
            enable_doc_grading = st.checkbox(
                "Document Grading",
                value=False,
                help=(
                    "Filter irrelevant chunks before generation using Mistral "
                    "function calling. Adds latency but improves precision on "
                    "ambiguous queries."
                ),
            )
        else:
            st.caption("Document grading requires a Mistral generator.")
        st.divider()

    st.caption("**Index:** SQuAD v2 · 16,423 chunks")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        st.caption(f"**GPU:** {gpu_name} ({vram_gb:.1f} GB)")
    else:
        st.caption("No GPU detected — Qwen will run on CPU (~30 s/query).")


# ─────────────────────────────────────────────────────────────────────────────
# Main area
# ─────────────────────────────────────────────────────────────────────────────

st.title("Advanced RAG Demo")
st.caption(
    "Ask a factual question — the pipeline retrieves relevant SQuAD v2 passages "
    "and generates a grounded answer."
)

st.markdown("**Try an example:**")
example_cols = st.columns(3)
examples = [
    "When did Beyoncé start becoming popular?",
    "Which artist did Beyoncé marry?",
    "Who influenced Beyoncé?",
]
for col, ex in zip(example_cols, examples):
    if col.button(ex, use_container_width=True):
        st.session_state["query_text"] = ex
        st.rerun()

query = st.text_area(
    "Question",
    key="query_text",
    placeholder="e.g. When did the French Revolution begin?",
    height=80,
    label_visibility="collapsed",
)

run_disabled = bool(
    not query.strip() or (generator_choice != "Qwen 2.5-3B (local)" and not api_key)
)

run_btn = st.button("Run Query", type="primary", disabled=run_disabled)

# ─────────────────────────────────────────────────────────────────────────────
# Query execution
# ─────────────────────────────────────────────────────────────────────────────

if run_btn and query.strip():
    with st.spinner("Running pipeline..."):
        try:
            t0 = time.time()
            pipeline = build_pipeline(mode, generator_choice, api_key, enable_doc_grading)

            if mode == "Linear":
                raw = pipeline.query(query, return_intermediate=True)
                result = {
                    "query": raw["query"],
                    "answer": raw.get("answer", ""),
                    "context_documents": raw.get("reranked_chunks", raw.get("chunks", [])),
                    "retrieved_chunks_raw": raw.get("retrieved_chunks", []),  # pre-reranking
                    "steps": ["Hybrid retrieval + reranking", "LLM generation"],
                    "retry_count": 0,
                    "used_fallback_retrieval": False,
                    "used_web_search": False,
                    "answer_is_acceptable": None,
                    "total_time_ms": raw.get("total_time_ms", 0),
                    "timings": raw.get("timings", {}),
                }
            else:
                result = pipeline.query(query)
                result.setdefault("total_time_ms", 0)
                result.setdefault("retrieved_chunks_raw", [])

            elapsed_ms = (time.time() - t0) * 1000

        except Exception as exc:
            st.error(f"**Pipeline error:** {exc}")
            st.stop()

    answer_text = result.get("answer") or "No answer generated."
    docs = result.get("context_documents", [])
    timings = result.get("timings", {})
    pre_docs = result.get("retrieved_chunks_raw", [])

    # ── Answer ────────────────────────────────────────────────────────────────
    st.subheader("Answer")
    st.success(answer_text)

    # ── Metrics row ───────────────────────────────────────────────────────────
    st.subheader("Metrics")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Latency", f"{elapsed_ms:.0f} ms")

    if generator_choice != "Qwen 2.5-3B (local)":
        model_name = GENERATOR_TO_MODEL[generator_choice]
        in_tok, out_tok, cost = estimate_cost(query, answer_text, model_name)
        with col2:
            st.metric("Est. Tokens", f"~{in_tok} in / ~{out_tok} out")
        with col3:
            st.metric("Est. Cost", f"~${cost:.5f}")
    else:
        gen_ms = timings.get("generation_ms", 0)
        with col2:
            st.metric("Generator", "Qwen 2.5-3B (local)")
        with col3:
            if gen_ms:
                st.metric("Generation time", f"{gen_ms:.0f} ms")
            elif torch.cuda.is_available():
                used = torch.cuda.memory_allocated() / 1e9
                st.metric("VRAM used", f"{used:.1f} GB")

    # ── Charts row: rerank scores + latency ───────────────────────────────────
    if docs or timings:
        chart_left, chart_right = st.columns(2)
        with chart_left:
            if docs:
                st.plotly_chart(plot_rerank_scores(docs), use_container_width=True)
        with chart_right:
            if timings and mode == "Linear":
                st.plotly_chart(plot_latency_pie(timings), use_container_width=True)

    # ── Faithfulness metrics ──────────────────────────────────────────────────
    with st.expander("Faithfulness & Context Metrics", expanded=False):
        render_faithfulness_metrics(docs, answer_text)

    # ── Pipeline trace (Adaptive / Agentic) ───────────────────────────────────
    if mode != "Linear":
        with st.expander("Pipeline Trace", expanded=True):
            steps = result.get("steps", [])
            for step in steps:
                st.markdown(f"- {step}")

            st.divider()
            flag_cols = st.columns(3)
            flag_cols[0].markdown(
                f"**Fallback retrieval:** {'Yes ⚠️' if result.get('used_fallback_retrieval') else 'No'}"
            )
            flag_cols[1].markdown(
                f"**Web search:** {'Yes 🌐' if result.get('used_web_search') else 'No'}"
            )
            flag_cols[2].markdown(f"**Retries:** {result.get('retry_count', 0)}")

            if result.get("answer_is_acceptable") is not None:
                quality = (
                    "Acceptable ✅" if result["answer_is_acceptable"] else "Below threshold ❌"
                )
                st.markdown(f"**Answer quality check:** {quality}")

            if enable_doc_grading:
                num_graded = result.get("num_docs_graded", "—")
                st.markdown(f"**Docs graded:** {num_graded}")

    # ── Reranking impact (Linear only, needs pre-reranking data) ──────────────
    if mode == "Linear" and pre_docs and docs:
        with st.expander("Reranking Impact", expanded=True):
            st.caption(
                "Each row is one document. "
                "Blue dot = rank from hybrid search · Coloured dot = rank after cross-encoder reranking. "
                "Rank 1 (leftmost) is best."
            )
            render_reranking_impact(pre_docs, docs)

    # ── Retrieved documents with answer highlighting ───────────────────────────
    with st.expander(f"Retrieved Documents ({len(docs)})", expanded=False):
        if docs:
            st.caption("Words highlighted in yellow appear in the generated answer.")
        for i, doc in enumerate(docs):
            score = doc.get("rerank_score", doc.get("score"))
            doc_id = doc.get("doc_id", f"doc_{i + 1}")
            content = doc.get("content", "")
            relevant = doc.get("relevant")

            header_parts = [f"**{i + 1}. `{doc_id}`**"]
            if score is not None:
                header_parts.append(f"Score: `{score:.3f}`")
            if relevant is not None:
                header_parts.append("Relevant ✅" if relevant else "Filtered out ❌")
            st.markdown(" · ".join(header_parts))

            highlighted = highlight_answer_in_text(content, answer_text)
            st.markdown(
                f"<div style='background:#f8f9fa;padding:8px 12px;"
                f"border-radius:4px;font-size:0.88em;color:#333;"
                f"border-left:3px solid #dee2e6;line-height:1.6'>{highlighted}</div>",
                unsafe_allow_html=True,
            )
            if i < len(docs) - 1:
                st.divider()
