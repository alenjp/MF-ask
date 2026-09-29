"""
Tiny chat UI:  streamlit run app/app.py

Welcome line + 3 example questions + "Facts-only. No investment advice."
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from config import DISCLAIMER, GROQ_API_KEY_ENV
from rag import ask, best_citation, index_ready

EXAMPLES = [
    "What is the expense ratio of HDFC Large Cap Fund Direct Growth?",
    "What is the minimum SIP for HDFC ELSS Tax Saver Fund Direct Plan Growth?",
    "What is the exit load and benchmark of HDFC Small Cap Fund Direct Growth?",
]

st.set_page_config(page_title="HDFC MF FAQ Assistant (RAG)", page_icon="📄", layout="centered")

ready, n_chunks = index_ready()
has_key = bool(os.getenv(GROQ_API_KEY_ENV))

with st.sidebar:
    st.markdown("### Corpus status")
    if ready and n_chunks:
        st.success(f"Indexed **{n_chunks}** chunks from **5** public HDFC scheme pages")
        st.caption("Embedding: sentence-transformers/all-MiniLM-L6-v2 · Vector DB: ChromaDB (on disk)")
    else:
        st.error("Vector store is empty. Run `python app/ingest.py` in a terminal.")
    st.markdown("### LLM")
    if has_key:
        st.success("GROQ_API_KEY loaded from .env")
    else:
        st.error("GROQ_API_KEY missing - retrieval works, generation is disabled.")
    st.markdown("### Data handling")
    st.caption(
        "Public pages only. The bot refuses PAN, Aadhaar, account numbers, "
        "OTPs, emails and phone numbers, and stores nothing about you."
    )

st.title("HDFC Mutual Fund FAQ Assistant")
st.caption("A RAG chatbot: question → embed → retrieve top-5 chunks → LLM → answer with one source link.")
st.info("**Facts-only. No investment advice.**")

if not (ready and n_chunks):
    st.stop()

for ex in EXAMPLES:
    if st.button(ex, key=ex):
        st.session_state["q"] = ex

question = st.text_input(
    "Ask a factual question about an HDFC scheme",
    key="q",
    placeholder="e.g. What is the exit load of HDFC Balanced Advantage Fund?",
)

if st.button("Ask", type="primary") and question and question.strip():
    with st.spinner("Retrieving and answering..."):
        result = ask(question)

    icon = {"answer": "✅", "unknown": "🔎", "out_of_scope": "🚫", "pii": "🔒",
            "performance": "📉", "error": "⚠️"}[result.kind]
    st.markdown(f"### {icon} Answer")
    st.write(result.text)

    if result.hits:
        top = best_citation(result.hits)
        st.markdown("#### Source")
        st.markdown(f"[{top.citation}]({top.source_url})")
        if result.kind == "answer":
            with st.expander("Retrieved context (top 5)"):
                for i, hit in enumerate(result.hits, 1):
                    st.markdown(
                        f"**{i}. {hit.scheme} — {hit.section}** · similarity `{hit.score:.3f}` · "
                        f"[link]({hit.source_url})"
                    )
                    st.caption(hit.text[:400].replace("\n", " · "))

    st.caption("Last updated from sources: 2026-09-29")

st.divider()
st.caption(DISCLAIMER)
