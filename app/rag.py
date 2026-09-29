"""
Query path: question -> embed -> retrieve top-k -> guard -> LLM -> answer + citation.

Guards run BEFORE the LLM so we never send advice-seeking or PII-bearing text to
the model, and a post-check caps the answer at 3 sentences.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    DISCLAIMER,
    EDUCATION_LABEL,
    EDUCATION_LINK,
    GROQ_API_KEY_ENV,
    GROQ_MODEL,
    MAX_SENTENCES,
    MIN_SIM,
    NO_ADVICE_REPLY,
    NON_CITABLE_SECTIONS,
    SCHEME_ALIASES,
    TOP_K,
    TOP_K_GENERIC,
)

# --- guard 1: never accept PII ---------------------------------------------
PII_PATTERNS = [
    re.compile(r"\b[ABCDEFGHJKLMNPQRSTUVW]{5}\d{4}[A-Z]\b"),                 # PAN
    re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"),                                 # Aadhaar
    re.compile(r"\b[6-9]\d{11}\b"),                                          # bank acct
    re.compile(r"\b\d{6}\b"),                                               # OTP
    re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"),                             # email
    re.compile(r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{9}(?!\d)\b"),              # phone
    re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\b"),                        # card
]
PII_REFUSAL = (
    "I don't accept or store personal identifiers. Please remove any PAN, "
    "Aadhaar number, account number, OTP, email address or phone number and "
    "ask your question again without them."
)

# --- guard 2: no advice ----------------------------------------------------
ADVICE_PATTERNS = [
    r"\bshould (i|we|you)\b", r"\bcan i (buy|sell|invest|put|start|pick)\b",
    r"\bis it (a )?good\b", r"\bworth (buying|investing|it)\b",
    r"\bwhich (fund|scheme|mutual fund|one) (should|do i|is best|would you)\b",
    r"\bbest (fund|scheme|mutual fund|elss|small cap|large cap|flexi cap) (for|to)\b",
    r"\bbook (profits?|losses?)\b", r"\bsuggest\b", r"\brecommend\w*\b",
    r"\b(advice|opinion)\b", r"\bportfolio (suggest|advice|allocation|construction)\b",
    r"\bhow much (should|do i)\b", r"\b(good|right) time to\b",
    r"\b(i|we) (want|wish|plan|thinking) to (invest|allocate|start|park)\b",
    r"\bshould i (exit|redeem|switch|sell|book)\b",
    r"\bexit (my|the) (investment|position|fund|money|portfolio)\b",
    r"\b(buy|sell|purchase) (it|this|that)\b", r"\bswitch (to|from)\b",
    r"\bcompare .{0,25}(better|best|outperform|vs|versus)\b",
    r"\bwhich one\b", r"\bhelp me (choose|pick|select|decide)\b",
]
ADVICE_RE = re.compile("|".join(ADVICE_PATTERNS), re.I)

# --- guard 3: no performance claims ----------------------------------------
PERFORMANCE_RE = re.compile(
    r"\b(outperform\w*|will return|expected return|future return|projected return|"
    r"guarantee\w*|cagr|best performing|performance|how much (will|did|would) .{0,25}"
    r"(return|earn|make|get)|how much will i (make|earn|get)|sends? me (the )?"
    r"(1y|3y|5y|10y)|money multiplier|sip calculator)\b"
    r"|\b\d+\s*(year|yr|month)s?\b.{0,30}\breturns?\b"
    r"|\breturns?\b.{0,40}\b(vs|versus|compared|comparison|outperform|rank|category average)\b",
    re.I,
)
PERFORMANCE_REFUSAL = (
    "I don't calculate, compare or forecast returns - that would be a "
    "performance claim. Please read the scheme's official monthly factsheet or "
    "the fund page linked below for published figures."
)

SYSTEM_PROMPT = f"""You are a mutual fund FAQ assistant for a classroom demo.

ABSOLUTE RULES
1. Answer ONLY from the CONTEXT blocks below. They are verbatim extracts from
   public HDFC Mutual Fund scheme pages. If the answer is not in the context,
   say exactly: "Not stated in the sources I have."
2. Never use outside knowledge. Never guess a number.
3. NO investment advice. Never say should/ought/best/buy/sell/hold/allocate.
   If the user asks for an opinion, say you only share published facts.
4. NO performance claims. Do not compute, compare, rank or forecast returns.
   If asked, say you don't forecast returns and point to the official factsheet.
5. At most {MAX_SENTENCES} sentences. Be plain and factual.
6. If the question names a scheme, use ONLY context blocks for that
   scheme. Never blend facts across schemes - the five HDFC pages use
   near-identical wording and mixing them produces wrong numbers.
7. Name the scheme explicitly in the answer (e.g. "HDFC ELSS Tax Saver Fund
   Direct Plan Growth").
8. If the question does NOT name a scheme but the context states the fact for
   one or more schemes, answer anyway: name the scheme(s) you used, and if the
   value is identical across them say it applies to all of them. Do NOT reply
   "not stated" merely because no scheme was named. Reserve "not stated" for
   facts that are genuinely absent from the context.
9. Do not add a source URL - the app renders the citation for you.
10. Always keep the tone factual and neutral.

{DISCLAIMER}"""


@dataclass
class Hit:
    text: str
    score: float
    source_url: str
    scheme: str
    section: str
    chunk_id: str

    @property
    def citation(self) -> str:
        return f"{self.scheme} - {self.section}"


@dataclass
class Answer:
    text: str
    hits: list[Hit]
    kind: str  # "answer" | "out_of_scope" | "unknown" | "pii" | "performance" | "error"


# --- resources --------------------------------------------------------------
@lru_cache(maxsize=1)
def _model():
    from ingest import get_model

    return get_model()


@lru_cache(maxsize=1)
def _collection():
    import chromadb

    from config import CHROMA_DIR, COLLECTION

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=chromadb.config.Settings(anonymized_telemetry=False, allow_reset=True),
    )
    return client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})


def index_ready() -> tuple[bool, int]:
    try:
        return True, _collection().count()
    except Exception:  # noqa: BLE001
        return False, 0


def detect_scheme(question: str) -> str | None:
    """Return the full scheme name if the question names exactly one scheme."""
    low = question.lower()
    matched = [
        name for name, aliases in SCHEME_ALIASES.items() if any(a in low for a in aliases)
    ]
    return matched[0] if len(matched) == 1 else None


def retrieve(question: str, k: int = TOP_K) -> list[Hit]:
    vec = _model().encode([question], normalize_embeddings=True)[0]
    res = _collection().query(
        query_embeddings=[vec.tolist()], n_results=k, include=["documents", "metadatas", "distances"]
    )
    hits: list[Hit] = []
    for doc, meta, dist in zip(
        res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        score = 1.0 - float(dist)
        hits.append(
            Hit(
                text=doc,
                score=score,
                source_url=meta.get("source_url", ""),
                scheme=meta.get("scheme", ""),
                section=meta.get("section", ""),
                chunk_id=meta.get("chunk_id", meta.get("id", "")),
            )
        )

    scheme = detect_scheme(question)
    if scheme:
        scoped = [h for h in hits if h.scheme == scheme]
        if scoped:  # never return an empty context just because scoping filtered all
            return scoped
    return hits


def _trim_sentences(text: str, limit: int = MAX_SENTENCES) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?<=[.!?])\s+", text)
    return " ".join(parts[:limit]).strip()


def _llm_answer(question: str, hits: list[Hit]) -> str:
    from groq import Groq

    client = Groq(api_key=os.environ[GROQ_API_KEY_ENV])
    context = "\n\n".join(
        f"[{i + 1}] SOURCE: {h.scheme} | SECTION: {h.section}\n{h.text}"
        for i, h in enumerate(hits)
    )
    user = f"CONTEXT\n{context}\n\nQUESTION\n{question}\n\nANSWER"
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0.0,
        max_tokens=220,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
    )
    return _trim_sentences(resp.choices[0].message.content or "")


_STOP = {
    "the", "a", "an", "of", "is", "are", "what", "which", "how", "much", "many", "on", "in",
    "for", "to", "and", "or", "my", "me", "i", "it", "this", "that", "does", "do", "can",
    "fund", "funds", "hdfc", "direct", "growth", "plan", "scheme", "mutual",
    "cap", "small", "large", "flexi", "balanced", "advantage", "saver", "tax", "elss", "period",
}

# Lay words -> the wording the source pages actually use.
_SYNONYMS = {
    "riskometer": ["risk"], "risk": ["risk"], "charges": ["expense", "load"],
    "fees": ["expense"], "ter": ["expense"], "sip": ["min. for sip", "sip"],
    "manager": ["manager"], "benchmark": ["benchmark"], "cost": ["expense"],
    "charges/": ["load"], "penalt": ["load"],
}


_LABELS = {
    "expense ratio", "min. for sip", "fund size (aum)", "rating", "nav", "exit load",
    "min. for 1st investment", "min. for 2nd investment", "stamp duty on investment:",
    "tax implication", "fund benchmark", "benchmark", "investor", "minimum investments",
    "risk", "riskometer", "stamp duty", "tax", "aum", "exit load, stamp duty and tax",
}


def _fact_units(lines: list[str]) -> list[str]:
    """Re-join a stray label with the value that follows it."""
    units: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        looks_like_label = (
            line.endswith(":")
            or line.lower() in _LABELS
            or (not any(ch.isdigit() for ch in line) and len(line.split()) <= 4 and line.isalpha())
        )
        if looks_like_label and nxt and (any(ch.isdigit() for ch in nxt) or nxt.startswith(("Nil", "₹", "Rs", "NIFTY", "Nifty"))):
            units.append(f"{line} {nxt}")
            i += 2
        else:
            units.append(line)
            i += 1
    return units


def _extractive_fallback(question: str, hits: list[Hit]) -> str:
    """No-API-key path: quote the most on-topic facts from the top chunks verbatim.

    Scoped to the single best-matching scheme so a Small Cap question can never
    be answered with a Large Cap line. Intentionally labelled as an extract,
    not an LLM answer.
    """
    if not hits:
        return "Not stated in the sources I have."

    raw_terms = [t for t in re.findall(r"[a-z0-9%.]+", question.lower()) if t not in _STOP and len(t) > 2]
    terms = set(raw_terms)
    for t in raw_terms:
        terms.update(_SYNONYMS.get(t, []))
    terms -= _STOP
    top = hits[0]
    same_scheme = [h for h in hits if h.scheme == top.scheme] or [top]

    scored: list[tuple[float, str]] = []
    for hit in same_scheme:
        for fact in _fact_units([ln.strip() for ln in hit.text.splitlines() if ln.strip()]):
            low = fact.lower()
            score = sum(1.0 for t in terms if t in low)
            if score == 0:
                continue  # no lexical link to the question at all
            if any(ch.isdigit() for ch in fact):
                score += 0.4  # prefer facts that actually carry a number
            if hit.score > 0.5:
                score += 0.2
            if len(fact) > 220:
                score -= 1.0  # skip boilerplate paragraphs
            scored.append((score, fact))

    if not scored:
        return "Not stated in the sources I have."
    scored.sort(key=lambda t: -t[0])
    seen, parts = set(), []
    for _, fact in scored:
        key = fact.lower()[:60]
        if key in seen:
            continue
        seen.add(key)
        parts.append(_trim_sentences(fact, MAX_SENTENCES))
        if len(parts) == 2:
            break
    return "(Extracted from the source - no LLM key configured) " + " ".join(parts)


def best_citation(hits: list[Hit]) -> Hit:
    """Prefer a fact-bearing chunk over a glossary chunk when both were retrieved."""
    citable = [h for h in hits if h.section not in NON_CITABLE_SECTIONS]
    return (citable or hits)[0]


def ask(question: str, k: int | None = None) -> Answer:
    question = question.strip()
    if not question:
        return Answer("Please type a question.", [], "error")

    for pattern in PII_PATTERNS:
        if pattern.search(question):
            return Answer(PII_REFUSAL, [], "pii")

    if ADVICE_RE.search(question):
        return Answer(f"{NO_ADVICE_REPLY} Educational reference: {EDUCATION_LINK}", [], "out_of_scope")

    if PERFORMANCE_RE.search(question):
        return Answer(PERFORMANCE_REFUSAL, [], "performance")

    # A scheme-less question competes against all 5 schemes' chunks, so widen k.
    if k is None:
        k = TOP_K if detect_scheme(question) else TOP_K_GENERIC

    try:
        hits = retrieve(question, k=k)
    except Exception as exc:  # noqa: BLE001
        return Answer(f"Retrieval failed: {exc}. Re-run `python app/ingest.py`.", [], "error")

    if not hits or hits[0].score < MIN_SIM:
        return Answer(
            "I don't have that in my 5 indexed HDFC scheme pages. "
            f"You can check the official source: {EDUCATION_LINK}",
            hits[:1],
            "unknown",
        )

    if not os.getenv(GROQ_API_KEY_ENV):
        text = _extractive_fallback(question, hits)
        kind = "unknown" if text.startswith("Not stated") else "answer"
        return Answer(text, hits, kind)

    try:
        text = _llm_answer(question, hits)
    except Exception as exc:  # noqa: BLE001
        return Answer(f"The LLM call failed: {exc}", hits, "error")

    if not text or "not stated in the sources" in text.lower():
        return Answer(
            "Not stated in the sources I have. The closest published facts I could find "
            f"are in: {best_citation(hits).citation}.",
            hits,
            "unknown",
        )

    if ADVICE_RE.search(text) or PERFORMANCE_RE.search(text):
        return Answer(
            "I can't phrase that without turning it into advice. Here's the published "
            f"fact instead - see {best_citation(hits).citation}.",
            hits,
            "out_of_scope",
        )

    return Answer(text, hits, "answer")
