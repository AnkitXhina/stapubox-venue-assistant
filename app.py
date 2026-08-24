"""
StapuBox Venue Assistant
========================
A RAG-powered Streamlit application that answers user queries about sports
venues using FAISS for vector search and Google Gemini for generation.

Architecture:
    1. Load & chunk the knowledge base by section headers.
    2. Embed chunks with Gemini `text-embedding-004`.
    3. Index embeddings in a FAISS inner-product index (cosine similarity).
    4. On query: embed the query, retrieve top-k chunks, generate a
       structured JSON answer with citations via `gemini-2.5-flash`.
    5. Render answer + citations in the Streamlit UI.

Author:  StapuBox Engineering
License: MIT
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import faiss
import numpy as np
import streamlit as st
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# ═══════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION & CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

KNOWLEDGE_BASE_PATH = Path(__file__).parent / "venue_knowledge_base.txt"

EMBEDDING_MODEL: str = "gemini-embedding-2"
GENERATION_MODEL: str = "gemini-3.6-flash"
EMBEDDING_DIMENSION: int = 768  # gemini-embedding-2 output dimension
TOP_K: int = 5                  # Number of chunks to retrieve
EMBEDDING_BATCH_SIZE: int = 100 # API batch limit

# Regex patterns for parsing the knowledge base structure
_VENUE_HEADER = re.compile(r"^={3,}\s*(.+?)\s*={3,}$")
_SECTION_HEADER = re.compile(r"^-{3,}\s*(.+?)\s*-{3,}$")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. PYDANTIC RESPONSE MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class Citation(BaseModel):
    """A single citation referencing a knowledge-base source."""
    source: str = Field(description="The venue and section the information came from.")
    excerpt: str = Field(description="The exact relevant quote from the source context.")


class RAGResponse(BaseModel):
    """Structured response returned by the generation model."""
    answer: str = Field(description="A detailed answer to the user's question.")
    citations: List[Citation] = Field(
        description="List of citations supporting the answer."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TextChunk:
    """A chunk of text extracted from the knowledge base with source metadata."""
    content: str
    venue: str
    section: str
    source_label: str = field(init=False)

    def __post_init__(self) -> None:
        self.source_label = f"{self.venue} › {self.section}"


# ═══════════════════════════════════════════════════════════════════════════════
# 4. GEMINI CLIENT INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def _resolve_api_key() -> str | None:
    """
    Resolve the Gemini API key using a three-tier fallback strategy:

    1. ``GOOGLE_API_KEY`` environment variable.
    2. Streamlit secrets (``st.secrets["GOOGLE_API_KEY"]``).
    3. Sidebar text-input stored in session state.
    """
    # Priority 1 — Environment variable
    if key := os.environ.get("GOOGLE_API_KEY"):
        return key

    # Priority 2 — Streamlit secrets (.streamlit/secrets.toml)
    try:
        if key := st.secrets.get("GOOGLE_API_KEY"):
            return key
    except FileNotFoundError:
        pass

    # Priority 3 — Sidebar input (stored in session state)
    return st.session_state.get("_sidebar_api_key") or None


@st.cache_resource
def _get_genai_client(api_key: str) -> genai.Client:
    """Create and cache a ``google.genai.Client``."""
    return genai.Client(api_key=api_key)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. KNOWLEDGE BASE LOADER & CHUNKER
# ═══════════════════════════════════════════════════════════════════════════════

def load_knowledge_base(path: Path = KNOWLEDGE_BASE_PATH) -> str:
    """Read the knowledge base text file from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Knowledge base not found at {path}")
    return path.read_text(encoding="utf-8")


def chunk_text(raw_text: str) -> list[TextChunk]:
    """
    Parse the knowledge base into :class:`TextChunk` objects.

    Splitting strategy: each ``--- Section Header ---`` block under its
    parent ``=== Venue Header ===`` becomes one chunk.  This preserves
    coherent, self-contained pieces of information that map naturally to
    citation sources.
    """
    chunks: list[TextChunk] = []
    current_venue = "Unknown Venue"
    current_section = "General"
    buffer: list[str] = []

    def _flush() -> None:
        """Flush accumulated lines into a TextChunk if non-empty."""
        text = "\n".join(buffer).strip()
        if text:
            chunks.append(TextChunk(
                content=text,
                venue=current_venue,
                section=current_section,
            ))

    for line in raw_text.splitlines():
        stripped = line.strip()

        # Check for venue-level header  (=== Venue Name ===)
        if m := _VENUE_HEADER.match(stripped):
            _flush()
            current_venue = m.group(1).strip()
            current_section = "General"
            buffer = []
            continue

        # Check for section-level header  (--- Section Name ---)
        if m := _SECTION_HEADER.match(stripped):
            _flush()
            current_section = m.group(1).strip()
            buffer = []
            continue

        buffer.append(line)

    _flush()  # Don't forget the last section
    return chunks


# ═══════════════════════════════════════════════════════════════════════════════
# 6. EMBEDDING ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def compute_embeddings(
    client: genai.Client,
    texts: list[str],
) -> np.ndarray:
    """
    Generate embeddings for *texts* using the Gemini embedding model.

    Args:
        client: An initialised ``genai.Client``.
        texts: The strings to embed.

    Returns:
        A ``(len(texts), EMBEDDING_DIMENSION)`` float32 NumPy array.
    """
    all_embeddings: list[list[float]] = []

    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[start : start + EMBEDDING_BATCH_SIZE]
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=batch,
        )
        all_embeddings.extend(emb.values for emb in response.embeddings)

    return np.array(all_embeddings, dtype="float32")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. FAISS INDEX MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build a FAISS inner-product index from L2-normalised embeddings.

    Using ``IndexFlatIP`` on L2-normalised vectors is equivalent to cosine
    similarity, which is the recommended metric for Gemini embeddings.
    """
    # Normalise in-place so inner product == cosine similarity
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def search_faiss_index(
    index: faiss.IndexFlatIP,
    query_embedding: np.ndarray,
    k: int = TOP_K,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Search the FAISS index for the *k* nearest neighbours.

    Returns:
        ``(scores, indices)`` — 1-D arrays of length *k*.
    """
    faiss.normalize_L2(query_embedding)
    scores, indices = index.search(query_embedding, k)
    return scores[0], indices[0]


# ═══════════════════════════════════════════════════════════════════════════════
# 8. RAG PIPELINE ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="📚 Indexing knowledge base …")
def build_rag_index(
    _client: genai.Client,
) -> tuple[faiss.IndexFlatIP, list[TextChunk]]:
    """
    End-to-end pipeline: load → chunk → embed → index.

    The result is cached by Streamlit so the expensive embedding call only
    happens once per app session.
    """
    raw_text = load_knowledge_base()
    chunks = chunk_text(raw_text)
    embeddings = compute_embeddings(
        _client, [c.content for c in chunks]
    )
    index = build_faiss_index(embeddings)
    return index, chunks


def retrieve_context(
    client: genai.Client,
    index: faiss.IndexFlatIP,
    chunks: list[TextChunk],
    query: str,
    k: int = TOP_K,
) -> list[tuple[TextChunk, float]]:
    """Retrieve the top-*k* most relevant chunks for a user query."""
    query_emb = compute_embeddings(client, [query])
    scores, indices = search_faiss_index(index, query_emb, k=k)
    return [
        (chunks[int(idx)], float(score))
        for score, idx in zip(scores, indices)
        if idx >= 0  # FAISS returns -1 for unfilled slots
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# 9. RESPONSE GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

_SYSTEM_INSTRUCTION = """\
You are **StapuBox Venue Assistant**, an expert on sports-venue rules, \
schedules, pricing, and policies.

RULES:
1. Answer ONLY from the provided context. If the context lacks sufficient \
   information, say so honestly.
2. Be specific — cite venue names, times, prices, and rules exactly as stated.
3. Include citations for every factual claim.
4. Respond with a JSON object matching the given schema. Do NOT add markdown \
   fences, commentary, or text outside the JSON.
"""


def generate_answer(
    client: genai.Client,
    query: str,
    context_chunks: list[tuple[TextChunk, float]],
) -> RAGResponse:
    """
    Generate a structured JSON answer using Gemini with retrieved context.

    The ``response_schema`` parameter enforces strict Pydantic-validated
    output from the model.
    """
    # ── Assemble context block ──────────────────────────────────────────────
    context_parts: list[str] = []
    for i, (chunk, _score) in enumerate(context_chunks, 1):
        context_parts.append(
            f"[Source {i}: {chunk.source_label}]\n{chunk.content}"
        )
    context_block = "\n\n---\n\n".join(context_parts)

    # ── Build the user prompt ───────────────────────────────────────────────
    user_prompt = (
        f"CONTEXT:\n{context_block}\n\n"
        f"USER QUESTION:\n{query}\n\n"
        "Respond with a JSON object containing \"answer\" and \"citations\"."
    )

    # ── Call Gemini with structured output ──────────────────────────────────
    response = client.models.generate_content(
        model=GENERATION_MODEL,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=_SYSTEM_INSTRUCTION,
            temperature=0.2,
            response_mime_type="application/json",
            response_schema=RAGResponse,
        ),
    )

    return RAGResponse.model_validate_json(response.text)


# ═══════════════════════════════════════════════════════════════════════════════
# 10. STREAMLIT UI
# ═══════════════════════════════════════════════════════════════════════════════

_EXAMPLE_QUERIES: list[str] = [
    "What are the tennis court rules at StapuBox Arena Central?",
    "When is Summit Athletic Club open on weekends?",
    "What is the dress code for the swimming pool?",
    "How much does badminton court booking cost at Greenfield?",
    "Can I rent squash equipment at Summit Athletic Club?",
    "What is the cancellation policy at Greenfield Sports Complex?",
]


def _render_sidebar() -> None:
    """Render sidebar: API-key input, example queries, and app info."""
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.text_input(
            "Gemini API Key",
            type="password",
            placeholder="Paste your GOOGLE_API_KEY …",
            help=(
                "Required if the `GOOGLE_API_KEY` environment variable "
                "or `.streamlit/secrets.toml` is not configured."
            ),
            key="_sidebar_api_key",
        )

        st.divider()
        st.header("💡 Try an Example")
        for idx, example in enumerate(_EXAMPLE_QUERIES):
            if st.button(example, key=f"_ex_{idx}", use_container_width=True):
                st.session_state["user_query"] = example
                st.rerun()

        st.divider()
        st.caption(
            "Built with ❤️ using **Streamlit**, **FAISS**, and **Google Gemini**"
        )


def _render_results(
    response: RAGResponse,
    context_chunks: list[tuple[TextChunk, float]],
) -> None:
    """Render the answer, citations, and debug panels."""

    # ── Answer ──────────────────────────────────────────────────────────────
    st.subheader("📋 Answer")
    st.success(response.answer)

    # ── Citations ───────────────────────────────────────────────────────────
    st.subheader("📚 Citations")
    if response.citations:
        for i, cit in enumerate(response.citations, 1):
            with st.expander(f"**Citation {i}** — {cit.source}", expanded=True):
                st.markdown(f"> *\"{cit.excerpt}\"*")
    else:
        st.info("No citations were returned for this response.")

    # ── Debug: Raw JSON ─────────────────────────────────────────────────────
    with st.expander("🔧 Raw JSON Response", expanded=False):
        st.json(response.model_dump())

    # ── Debug: Retrieved chunks ─────────────────────────────────────────────
    with st.expander("📄 Retrieved Context Chunks", expanded=False):
        for chunk, score in context_chunks:
            st.markdown(f"**{chunk.source_label}** — similarity: `{score:.4f}`")
            st.code(chunk.content[:500], language=None)
            st.divider()


def main() -> None:
    """Main Streamlit application entry point."""

    # ── Page configuration ──────────────────────────────────────────────────
    st.set_page_config(
        page_title="StapuBox Venue Assistant",
        page_icon="🏟️",
        layout="wide",
    )

    _render_sidebar()

    # ── Header ──────────────────────────────────────────────────────────────
    st.title("🏟️ StapuBox Venue Assistant")
    st.markdown(
        "Ask anything about our sports venues — **rules**, **hours**, "
        "**pricing**, **dress codes**, and more.  \n"
        "Powered by **RAG** with FAISS vector search & Google Gemini."
    )
    st.divider()

    # ── Resolve API key ─────────────────────────────────────────────────────
    api_key = _resolve_api_key()
    if not api_key:
        st.warning(
            "🔑 Please provide a Gemini API key via the **sidebar**, the "
            "`GOOGLE_API_KEY` environment variable, or "
            "`.streamlit/secrets.toml`."
        )
        st.stop()

    client = _get_genai_client(api_key)

    # ── Build / load the FAISS index (cached) ───────────────────────────────
    try:
        index, chunks = build_rag_index(client)
    except Exception as exc:
        st.error(f"❌ Failed to build the knowledge-base index: {exc}")
        st.stop()

    st.caption(f"Knowledge base loaded: **{len(chunks)}** chunks indexed.")

    # ── Query input ─────────────────────────────────────────────────────────
    query = st.text_input(
        "🔍 Your question",
        key="user_query",
        placeholder="e.g. What are the tennis court rules at StapuBox Arena Central?",
    )

    if not query:
        st.info(
            "Type a question above or click an example in the sidebar to "
            "get started."
        )
        st.stop()

    # ── Run RAG pipeline ────────────────────────────────────────────────────
    with st.spinner("🔎 Searching knowledge base and generating answer …"):
        try:
            context_chunks = retrieve_context(client, index, chunks, query)
            rag_response = generate_answer(client, query, context_chunks)
        except Exception as exc:
            st.error(f"❌ An error occurred: {exc}")
            st.stop()

    # ── Display results ─────────────────────────────────────────────────────
    _render_results(rag_response, context_chunks)


# ── Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
