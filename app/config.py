"""Shared paths, constants and prompt text for the MF FAQ RAG assistant."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
SOURCES_CSV = ROOT / "sources.csv"
RAW_DIR = ROOT / "data" / "raw"
CHUNKS_TXT = ROOT / "data" / "chunks.txt"
CHROMA_DIR = ROOT / "data" / "chroma"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim, local, no API key
COLLECTION = "hdfc_mf_faq"
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
GROQ_API_KEY_ENV = "GROQ_API_KEY"

# --- retrieval ---------------------------------------------------------------
TOP_K = 10            # context window: retrieve up to 10 chunks per question
TOP_K_GENERIC = TOP_K  # kept as a separate constant so the adaptive-k logic reads plainly
MIN_SIM = 0.22       # cosine floor; below this we say "not in the sources"

# Sections that define terms rather than state scheme facts - never cite these
# when a fact-bearing chunk was also retrieved.
NON_CITABLE_SECTIONS = {"Glossary (Understand terms)"}

# --- chunking ----------------------------------------------------------------
# 900 chars (~225 tokens). Sized after inspecting the pages: it is large enough to
# hold a whole "About the scheme" block (objective + risk + minimums + benchmark
# for one scheme) so the benchmark is not orphaned in a second chunk, and small
# enough that top-10 retrieval does not dump a whole page into the prompt.
CHUNK_CHARS = 900
CHUNK_OVERLAP_CHARS = 140

# --- answer style ------------------------------------------------------------
MAX_SENTENCES = 3

DISCLAIMER = (
    "Facts-only. No investment advice. Answers are summarised from the public "
    "scheme pages listed below and are not a recommendation to buy, sell or hold "
    "any security. Mutual fund investments are subject to market risks; read "
    "all scheme related documents carefully before investing."
)

NO_ADVICE_REPLY = (
    "I can only share published facts about these HDFC Mutual Fund schemes - I "
    "don't give buy/sell/hold opinions or portfolio suggestions. For a neutral "
    "explainer on how to evaluate a fund, see the official investor education "
    "material."
)

EDUCATION_LINK = "https://groww.in/mutual-funds"
EDUCATION_LABEL = "Groww Mutual Funds overview (public)"

# Phrases that identify a scheme in a question. Retrieval is scoped to the
# matching scheme so an answer about HDFC Small Cap can never be assembled from
# HDFC Large Cap chunks (which read almost identically).
SCHEME_ALIASES = {
    "HDFC Large Cap Fund (Direct Growth)": ["large cap", "largecap"],
    "HDFC Flexi Cap Fund (Direct Growth)": ["flexi cap", "flexicap", "hdfc equity fund", "hdfc flexi"],
    "HDFC ELSS Tax Saver Fund (Direct Plan Growth)": ["elss", "tax saver", "taxsaver"],
    "HDFC Small Cap Fund (Direct Growth)": ["small cap", "smallcap"],
    "HDFC Balanced Advantage Fund (Direct Growth)": ["balanced advantage", "balancedadvantage"],
}
