"""
Tiny chat UI:  streamlit run app/app.py

Welcome line + 5 quick question buttons + "Facts-only. No investment advice."
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from config import DISCLAIMER, GROQ_API_KEY_ENV, GROQ_MODEL, TOP_K
from rag import Answer, ask, best_citation, index_ready

ROOT = Path(__file__).resolve().parent.parent

# Examples chosen to demo a grounded hit, an honest miss, an advice refusal,
# and a PII refusal, plus a second grounded hit.
EXAMPLES = [
    "What is the expense ratio of HDFC Large Cap Fund Direct Growth?",
    "What is the minimum SIP amount for HDFC ELSS Tax Saver Fund Direct Plan Growth?",
    "What is the lock-in period for HDFC ELSS Tax Saver Fund?",
    "Should I buy HDFC Small Cap Fund for my portfolio?",
    "My PAN is ABCDE1234F - what is the exit load?",
]

ICONS = {
    "answer": "✅",
    "unknown": "🔎",
    "out_of_scope": "🚫",
    "pii": "🔒",
    "performance": "📉",
    "error": "⚠️",
}


def retrieved_on() -> str:
    try:
        with (ROOT / "sources.csv").open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        return rows[0]["retrieved_on"] if rows else ""
    except Exception:  # noqa: BLE001
        return ""


def render_answer(result: Answer) -> None:
    st.markdown(f"**{ICONS.get(result.kind, '')}** {result.text}")
    if result.hits:
        top = best_citation(result.hits)
        st.markdown(f"[{top.citation}]({top.source_url})")
        if result.kind == "answer":
            with st.expander(f"Retrieved context (top {TOP_K})"):
                for i, hit in enumerate(result.hits, 1):
                    st.markdown(
                        f"**{i}. {hit.scheme} — {hit.section}** · similarity "
                        f"`{hit.score:.3f}` · [link]({hit.source_url})"
                    )
                    st.caption(hit.text[:400].replace("\n", " · "))
    st.caption(f"Last updated from sources: {retrieved_on()}")


def submit(text: str) -> None:
    text = text.strip()
    if not text:
        return
    with st.spinner("Retrieving and answering..."):
        result = ask(text)
    st.session_state["messages"].append({"role": "user", "content": text})
    st.session_state["messages"].append({"role": "assistant", "result": result})


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
        st.success("GROQ_API_KEY loaded from environment")
    else:
        st.error("GROQ_API_KEY missing - retrieval works, generation is disabled. "
                 "Add it in Render → Environment, then redeploy.")
    st.caption(f"Model: {GROQ_MODEL}")
    st.markdown("### Data handling")
    st.caption(
        "Public pages only. The bot refuses PAN, Aadhaar, account numbers, "
        "OTPs, emails and phone numbers, and stores nothing about you."
    )

st.title("HDFC Mutual Fund FAQ Assistant")
st.caption(f"A RAG chatbot: question → embed → retrieve top-{TOP_K} chunks → LLM → answer with one source link.")
st.info("**Facts-only. No investment advice.**")

if not (ready and n_chunks):
    st.stop()

st.session_state.setdefault("messages", [])

for ex in EXAMPLES:
    if st.button(ex):
        submit(ex)

question = st.text_input(
    "Ask a factual question about an HDFC scheme",
    key="q_input",
    placeholder="e.g. What is the exit load of HDFC Balanced Advantage Fund?",
)

if st.button("Ask", type="primary") and question and question.strip():
    submit(question)

for m in st.session_state["messages"]:
    if m["role"] == "user":
        with st.chat_message("user"):
            st.write(m["content"])
    else:
        with st.chat_message("assistant"):
            render_answer(m["result"])

st.divider()
st.caption(DISCLAIMER)