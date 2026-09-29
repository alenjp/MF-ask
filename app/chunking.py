"""
Stage 2 of ingestion: CHUNK.

Chunking strategy (chosen after inspecting all 5 raw pages):
  Data shape ....... These are not prose pages. Each page is a stack of one-fact-
                     per-line rows ("Min. for SIP" / "Rs 100" / "Fund size (AUM)"
                     / "Rs 39,933.36 Cr") plus a 50-row holdings table, a return
                     calculator, a peer table and a fund-manager link list. There
                     are no blank-line paragraphs, so the indivisible unit is the
                     LINE, not the paragraph: a label and its value must never be
                     separated.
  Why not fixed-size
  token splitting .. A fixed 500/50 split cuts a label away from its value
                     ("Exit load" ends up in one chunk, "Nil" in the next) and
                     wastes overlap budget re-emitting holdings rows.
  Why section-aware . Detected headings flush the buffer and whole lines are
                     packed together, so one chunk == one readable fact group
                     ("expense ratio + minimums + AUM").
  Chunk size ....... 900 characters (~225 tokens). 700 orphaned each scheme's
                     benchmark into a second, weaker-ranked chunk; 900 holds a
                     whole "About the scheme" block while still keeping a top-5
                     retrieval well under one page.
  Overlap .......... 140 characters, applied by re-emitting the *last whole
                     lines* of the previous chunk. Line-granular overlap (not
                     character slicing) so we never cut mid-label.
  Dropped .......... Holdings, return calculator, returns/rankings, peer tables
                     and fund-manager link lists - roughly 75% of the page
                     characters and none of it useful to a scheme-mechanics FAQ.
                     Keeping them let 87 of 132 chunks be portfolio noise that
                     crowded out the fee and exit-load facts.
  Embedded text .... "<section> - <scheme>: <body>". Every raw chunk repeats
                     the scheme name, so "HDFC ... Fund" dominated the cosine
                     score and buried the section term the user asked about.
  Metadata ......... source_url, scheme, category, plan, section, chunk_id,
                     n_chars, retrieved_on - every chunk is independently
                     citable back to one URL.

Result: 36 chunks across 5 documents. All chunks are dumped to data/chunks.txt
so they can be eyeballed before any embedding happens.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

from config import CHUNK_CHARS, CHUNK_OVERLAP_CHARS, CHUNKS_TXT, RAW_DIR

# Headings that mark the start of a new logical section on these pages.
SECTION_HEADINGS = {
    "return calculator": "Return calculator",
    "monthly sip": "Return calculator",
    "historic returns": "Historic returns",
    "returns and rankings": "Returns and rankings",
    "minimum investments": "Minimum investments",
    "understand terms": "Glossary (Understand terms)",
    "exit load": "Exit load & charges",
    "exit load, stamp duty and tax": "Exit load, stamp duty & tax",
    "holdings": "Holdings",
    "fund management": "Fund management",
    "about": "About the scheme",
    "fund house": "Fund house",
    "compare similar funds": "Compare similar funds",
    "riskometer": "Riskometer",
    "mandatory additional": "Mandatory additional disclosure",
    "download": "Downloads & documents",
    "scheme information document": "Downloads & documents",
}

# Blocks that are pure navigation / broker promo: dropped before chunking.
DROP_BLOCK_PATTERNS = (
    re.compile(r"^(Also manages these schemes|Show More|Watchlist|Compare|Contact Us|"
               r"Download the App|GROWW|Products|Privacy Policy|Regulatory)", re.I),
)

PERFORMANCE_SECTIONS = {"Historic returns", "Returns and rankings", "Return calculator"}

# Section headings whose entire body is out of scope for a scheme-mechanics FAQ:
# holdings tables, return calculators, peer tables and fund-manager link lists.
# Dropping them keeps the corpus tight so top-5 retrieval never fills the prompt
# with portfolio rows instead of the fee/exit-load/benchmark facts.
DROP_SECTIONS = {
    "Holdings",
    "Return calculator",
    "Historic returns",
    "Returns and rankings",
    "Compare similar funds",
    "Fund management",
}


@dataclass
class Chunk:
    chunk_id: str
    source_url: str
    scheme: str
    category: str
    plan: str
    section: str
    text: str
    n_chars: int
    retrieved_on: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["metadata"] = {
            "source_url": self.source_url,
            "scheme": self.scheme,
            "category": self.category,
            "plan": self.plan,
            "section": self.section,
            "retrieved_on": self.retrieved_on,
        }
        return d


def _parse_header(lines: list[str]) -> dict:
    header = {"source_url": "", "scheme": "", "category": "", "plan": "", "retrieved_on": ""}
    for line in lines[:8]:
        if ":" in line and not line.startswith("-"):
            key, _, value = line.partition(":")
            key = key.strip().lower()
            if key in header:
                header[key] = value.strip()
    return header


def _is_heading(block: str) -> str | None:
    """Return the raw heading key if this line starts a new section, else None."""
    first = block.strip().splitlines()[0].strip()
    first = re.sub(r"\s*\(\d+\)$", "", first)  # "Holdings (50)" -> "Holdings"
    if len(first) > 45:
        return None
    if any(ch.isdigit() for ch in first) and not first.endswith("%"):
        return None
    key = first.lower()
    return key if key in SECTION_HEADINGS else None


def _blocks(text: str) -> list[str]:
    """Atomic units for packing.

    These pages have no blank-line paragraph structure - every fact is its own
    line ("Min. for SIP" / "Rs 100" / "Fund size (AUM)" / ...). So the line,
    not the paragraph, is the indivisible unit: we must never cut between a
    label and its value.
    """
    units: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("-" * 10):
            continue
        if len(line) < 2 and not line.isdigit():
            continue
        if any(p.match(line) for p in DROP_BLOCK_PATTERNS):
            continue
        if units and line == units[-1] and len(line) > 3:
            continue  # nav duplicate row immediately repeated
        units.append(line)
    return units


def chunk_document(path: Path, retrieved_on: str) -> list[Chunk]:
    text = path.read_text(encoding="utf-8")
    head, _, body = text.partition("-" * 72)
    header = _parse_header(head.splitlines())
    header["retrieved_on"] = retrieved_on

    blocks = _blocks(body)
    chunks: list[Chunk] = []
    section = "Overview"
    section_key = ""
    buf: list[str] = []
    buf_len = 0
    counter = 0

    def flush() -> None:
        nonlocal buf, buf_len, counter
        if not buf or SECTION_HEADINGS.get(section_key, section) in DROP_SECTIONS:
            buf = []
            buf_len = 0
            return
        counter += 1
        blob = "\n\n".join(buf).strip()
        if len(blob) >= 25:
            chunks.append(
                Chunk(
                    chunk_id=f"{header['scheme']}#{counter}",
                    section=section,
                    text=blob,
                    n_chars=len(blob),
                    **header,
                )
            )
        buf = []
        buf_len = 0

    for block in blocks:
        key = _is_heading(block)
        if key:
            flush()
            # Pages repeat a bare parent heading *inside* a sub-section: "Exit
            # load", "Exit load, stamp duty and tax", "Exit load" again. Letting
            # that second parent reset the section would relabel the stamp-duty
            # block back to the generic parent and cost it its section label in
            # the embedded header. Compare on the raw keys, not the labels.
            if section_key and key != section_key and key in section_key:
                continue
            section_key = key
            section = SECTION_HEADINGS[key]
            continue
        if buf_len + len(block) + 1 > CHUNK_CHARS and buf:
            tail: list[str] = []
            size = 0
            for line in reversed(buf):  # block-granular overlap: whole lines only
                if size + len(line) > CHUNK_OVERLAP_CHARS:
                    break
                tail.insert(0, line)
                size += len(line) + 1
            flush()
            buf = list(tail)
            buf_len = size
        buf.append(block)
        buf_len += len(block) + 1

    flush()
    for chunk in chunks:
        if chunk.section == "Overview" and chunk.section not in PERFORMANCE_SECTIONS:
            chunk.section = _guess_section(chunk.text)
        chunk.metadata["is_performance_data"] = chunk.section in PERFORMANCE_SECTIONS
    return chunks


_SECTION_HINTS = (
    ("expense ratio", "Fees: expense ratio"),
    ("benchmark", "Benchmark & investment objective"),
    ("investment objective", "Benchmark & investment objective"),
    ("exit load", "Exit load & charges"),
    ("stamp duty", "Exit load, stamp duty & tax"),
    ("tax implication", "Exit load, stamp duty & tax"),
    ("min. for sip", "Key facts: NAV, AUM, minimums"),
    ("minimum sip", "Key facts: NAV, AUM, minimums"),
    ("fund size (aum)", "Key facts: NAV, AUM, minimums"),
    ("lock", "Lock-in"),
    ("risk", "Riskometer"),
    ("custodian", "Fund house & service providers"),
    ("rating", "Key facts: NAV, AUM, minimums"),
)


def _guess_section(text: str) -> str:
    low = text.lower()
    for hint, label in _SECTION_HINTS:
        if hint in low:
            return label
    return "Key facts & scheme details"


def build_all_chunks(retrieved_on: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(RAW_DIR.glob("*.txt")):
        chunks.extend(chunk_document(path, retrieved_on))
    return chunks


def dump_chunks(chunks: list[Chunk], target: Path = CHUNKS_TXT) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "=" * 78,
        "RAG CHUNK DUMP - HDFC Mutual Fund FAQ corpus",
        f"chunks: {len(chunks)}   model: sentence-transformers/all-MiniLM-L6-v2",
        f"chunk chars: {CHUNK_CHARS}   overlap: {CHUNK_OVERLAP_CHARS} (block granular)",
        "=" * 78,
        "",
    ]
    for chunk in chunks:
        lines += [
            "-" * 78,
            f"chunk_id : {chunk.chunk_id}",
            f"scheme    : {chunk.scheme}",
            f"category  : {chunk.category}  |  plan: {chunk.plan}",
            f"section   : {chunk.section}",
            f"url       : {chunk.source_url}",
            f"n_chars   : {chunk.n_chars}",
            "-" * 78,
            chunk.text,
            "",
        ]
    target.write_text("\n".join(lines), encoding="utf-8")
