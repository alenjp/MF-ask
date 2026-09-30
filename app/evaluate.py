"""
Smoke test + deliverable generator.

Without GROQ_API_KEY it still exercises LOAD -> CHUNK -> EMBED -> RETRIEVE and
prints the top hits per question. With a key it writes SAMPLE_QA.md.

Usage:  python app/evaluate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import os

from config import DISCLAIMER, GROQ_API_KEY_ENV
from rag import ask, best_citation

ROOT = Path(__file__).resolve().parent.parent

QUERIES = [
    ("What is the expense ratio of HDFC Large Cap Fund Direct Growth?", "answer"),
    ("What is the minimum SIP amount for HDFC ELSS Tax Saver Fund Direct Plan Growth?", "answer"),
    ("What is the exit load on HDFC Small Cap Fund Direct Growth?", "answer"),
    ("What is the benchmark of HDFC Flexi Cap Fund Direct Growth?", "answer"),
    ("What is the risk category / riskometer level of HDFC Balanced Advantage Fund?", "answer"),
    ("Who is the fund manager of HDFC Large Cap Fund Direct Growth?", "answer"),
    ("What is the stamp duty and tax implication on redemption?", "answer"),
    ("What is the lock-in period for HDFC ELSS Tax Saver Fund?", "unknown"),
    ("Should I buy HDFC Small Cap Fund for my portfolio?", "out_of_scope"),
    ("What are the 5 year returns of HDFC Large Cap Fund versus the category?", "performance"),
    ("My PAN is ABCDE1234F, what is the exit load?", "pii"),
    ("What is the weather in Mumbai?", "unknown"),
]


def main() -> int:
    has_key = bool(os.getenv(GROQ_API_KEY_ENV))
    rows = []
    for q, expected in QUERIES:
        res = ask(q)
        top = best_citation(res.hits) if res.hits else None
        print(f"\nQ: {q}\n   kind={res.kind} (expected {expected})")
        print(f"   {res.text[:230]}")
        if top:
            print(f"   top: {top.citation} score={top.score:.3f}")
        rows.append(
            f"### `{q}`\n\n"
            f"**Answer ({res.kind}):** {res.text}\n\n"
            + (f"**Source:** [{top.citation}]({top.source_url})\n\n" if top else "")
        )

    passed = sum(
        1 for q, exp in QUERIES if ask(q).kind == exp
    )
    print(f"\n[eval] guard routing: {passed}/{len(QUERIES)} as expected")

    out = [
        "# Sample Q&A - HDFC Mutual Fund FAQ Assistant (RAG)",
        "",
        f"Model: Groq `{os.getenv('GROQ_MODEL', 'qwen/qwen3.8-27b')}` · "
        "Embeddings: sentence-transformers/all-MiniLM-L6-v2 · Vector DB: ChromaDB (on disk)",
        "",
        "Corpus: 5 public HDFC Mutual Fund scheme pages (see `sources.csv`), 36 chunks. "
        "Last updated from sources: 2026-09-29.",
        "",
        f"> **Disclaimer used in the UI:** {DISCLAIMER}",
        "",
        "---",
        "",
    ] + rows
    if not has_key:
        out[8:8] = [
            "> NOTE: this transcript was generated in **extractive mode** (no `GROQ_API_KEY` set), "
            "so answers quote source lines verbatim instead of Groq prose. Set the key and re-run "
            "`python app/evaluate.py` to regenerate with LLM answers. Retrieval scores, guard "
            "routing and citations are identical either way.",
            "",
            "---",
            "",
        ]
    (ROOT / "SAMPLE_QA.md").write_text("\n".join(out), encoding="utf-8")
    print(f"[eval] wrote SAMPLE_QA.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
