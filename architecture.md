# Architecture — Mutual Fund FAQs RAG Chatbot

Companion to [`PRD.md`](PRD.md). Describes how the shipped prototype is put
together and why each decision was made.

**Stack (fixed by the brief):** `sentence-transformers/all-MiniLM-L6-v2` ·
ChromaDB (persisted) · Groq · Streamlit. No orchestration framework — the
pipeline is plain Python so the RAG loop is explainable on a whiteboard.

---

## 1. System shape

Two paths that never meet: a slow **ingestion** path that runs once, and a fast
**query** path that runs per question.

```
                        INGESTION  (run once, offline)
 ┌──────────┐   ┌───────────────┐   ┌──────────┐   ┌──────────┐   ┌──────────────┐
 │ 5 public │──▶│ 1. LOAD       │──▶│ 2. CHUNK │──▶│ 3. EMBED │──▶│ 4. STORE     │
 │ URLs     │   │ requests +    │   │ section- │   │ MiniLM   │   │ ChromaDB     │
 │ (sources │   │ BeautifulSoup │   │ aware    │   │ 384-dim  │   │ Persistent-  │
 │  .csv)   │   │ nav/footer    │   │ line-wise│   │ local    │   │ Client       │
 └──────────┘   │ stripped      │   │ packing  │   │          │   │ data/chroma/ │
                └───────┬───────┘   └────┬─────┘   └────┬─────┘   └──────┬───────┘
                        │                │              │                │
                  data/raw/*.txt   data/chunks.txt    36 x 384 floats    │
                     (5 files)     (human-readable)                    │
                                                                          ▼
                                                                   ┌────────────┐
                                                                   │  ChromaDB  │
                                                                   │  on disk   │
                                                                   └─────┬──────┘
                                                                         │ cosine top-k
                        QUERY  (per question)                           │
 ┌──────────┐   ┌───────────────┐   ┌──────────┐   ┌───────────┐         │
 │ Question │──▶│ 0. GUARD      │──▶│ 1. EMBED │──▶│ 2. RETRIEVE│◀────────┘
 │          │   │ PII           │   │ same     │   │ top-k     │
 │          │   │ advice        │   │ model    │   │ + scheme  │
 │          │   │ performance   │   │          │   │ scoping   │
 └──────────┘   └───────┬───────┘   └──────────┘   └─────┬─────┘
                        │ refuse                         │
                        │ (no LLM call)                   ▼
                        │                          ┌───────────┐
                        │                          │ 3. GENERATE│
                        │                          │ Groq, t=0  │
                        │                          │ ≤3 sent.   │
                        │                          └─────┬─────┘
                        │                                │
                        │      answer + 1 citation +       │
                        │      "Last updated from sources: │
                        ▼                                ▼
                  ┌──────────────────────────────────────────┐
                  │ 4. PRESENT — Streamlit chat UI           │
                  │ answer · source link · top-k expander    │
                  │ sidebar corpus status · disclaimer       │
                  └──────────────────────────────────────────┘
```

**Why split this way:** ingestion is the only slow, network- and model-heavy
step. Persisting to disk means the demo never pays it — `app/ingest.py` is
idempotent and skips when the collection is already populated.

## 2. Module map

| Module | RAG stage | Responsibility | Key exports |
|--------|-----------|----------------|-------------|
| `app/config.py` | — | Paths, model names, thresholds, chunk sizes, scheme aliases, user-facing copy | `EMBED_MODEL`, `GROQ_MODEL`, `TOP_K`, `TOP_K_GENERIC`, `MIN_SIM`, `CHUNK_CHARS`, `MAX_SENTENCES`, `SCHEME_ALIASES`, `NON_CITABLE_SECTIONS`, `DISCLAIMER`, `NO_ADVICE_REPLY` |
| `app/fetch_sources.py` | **LOAD** | Fetch the 5 URLs from `sources.csv`, strip nav/footer/boilerplate | `extract`, `slug` |
| `app/chunking.py` | **CHUNK** | Line-granular, section-aware packing; metadata; `.txt` dump | `chunk_document`, `build_all_chunks`, `dump_chunks` |
| `app/ingest.py` | **EMBED + STORE** | Encode with MiniLM, upsert into persistent Chroma, idempotency check | `get_model`, `get_client`, `ingest` |
| `app/rag.py` | **GUARD / RETRIEVE / GENERATE** | All three guards, retrieval, prompt assembly, LLM call, citation, answer shaping | `ask`, `retrieve`, `detect_scheme`, `best_citation`, `SYSTEM_PROMPT` |
| `app/app.py` | **PRESENT** | Streamlit chat UI, sidebar status, expander of retrieved context | — |
| `app/evaluate.py` | — | 12-query smoke test, regenerates `SAMPLE_QA.md` | `main` |
| `sources.csv` | — | Source-of-record for the 5 URLs and the retrieval date | — |

`SYSTEM_PROMPT` lives in `rag.py` rather than `config.py` on purpose: it is
composed from `config.DISCLAIMER` and interleaved with the guard behaviour it
has to agree with, so the two cannot drift apart.

Dependencies flow one way: `config` ← `chunking` ← `ingest` ← `rag` ← `app`.
`rag` never writes to disk.

**Runtime note:** this project was built on **Python 3.12**. `chromadb` and
`sentence-transformers` depend on compiled wheels that are not published for
every interpreter, so a mismatched `python` on `PATH` fails at import time with
a confusing error rather than at install time. See `README.md` §3.

## 3. Ingestion design

### 3.1 LOAD

- URLs and `retrieved_on` come from `sources.csv` — the corpus is declared in one
  reviewable place, not hardcoded in Python.
- `requests` + `BeautifulSoup`; `header`/`nav`/`footer`/`script`/`style`/`form`
  are decomposed, then the body is sliced at the first footer marker and
  mega-menu link rows are dropped.
- Output is cached per URL in `data/raw/`, so a re-run is cheap and the cleaned
  text is reviewable as a diff.
- Failure is **loud**: a fetch error exits non-zero rather than ingesting a
  partial corpus.

### 3.2 CHUNK — the decision, and the evidence for it

I fetched the pages and read the raw text before writing the chunker. What the
data actually looks like:

- **No paragraphs.** Every fact is its own line: `Min. for SIP` / `₹100` /
  `Fund size (AUM)` / `₹39,933.36 Cr` / …
- **~75% of each page is noise** for this product: a 50-row holdings table, a
  return calculator, a returns/rankings table, a peer-comparison table, and a
  fund-manager "also manages these schemes" link list.

| Decision | Value | Reasoning |
|---|---|---|
| Split unit | **single line** | A label and its value must never be separated. A fixed 500/50 token split puts `Exit load` in one chunk and `Nil` in the next. |
| Packing | section-aware | Detected headings flush the buffer, so one chunk = one readable fact group. |
| Chunk size | **900 chars** (~225 tokens) | Validated against real queries, not guessed — see §3.3. |
| Overlap | **140 chars, line-granular** | Re-emits the *last whole lines* of the previous chunk, so overlap never cuts mid-label. |
| Dropped sections | Holdings, Return calculator, Returns/rankings, Compare-similar, Fund management | Keeping them produced 87 of 132 chunks of portfolio noise that crowded out the fee and exit-load facts. |
| Metadata | `chunk_id, source_url, scheme, category, plan, section, n_chars, retrieved_on` | Every chunk is independently citable back to exactly one URL — the citation is metadata-driven, so the model can never invent one. |
| Auditability | `data/chunks.txt` | Every chunk with its metadata, human-readable, written **before** embedding. |

### 3.3 Three chunking decisions that were made by testing, not theory

All three were caught by running real queries against the index and reading the
output, not by staring at the chunker.

1. **700 → 900 characters.** At 700, each scheme's *About the scheme* section split
   in two, orphaning the benchmark line into a chunk that never ranked in the top
   5 for "What is the benchmark of …?". At 900 the whole block (objective + risk +
   minimums + benchmark) stays together and the query retrieves it.

2. **Sub-headings must not demote the parent section.** These pages repeat a bare
   parent heading *inside* a sub-section — `Exit load`, then
   `Exit load, stamp duty and tax`, then `Exit load` **again**. The last one
   reset the section, so all five schemes' stamp-duty blocks were labelled
   `Exit load & charges` and lost "stamp duty" from the embedded header, which is
   exactly the term the user types. Fixed by tracking the raw heading *key* and
   ignoring a heading that is a prefix of the current one. All five now carry an
   `Exit load, stamp duty & tax` block.

3. **Store what you embed, not just the vector.** Every raw chunk repeats the
   scheme name, so cosine similarity was dominated by *"HDFC … Fund"* matching and
   the section term the user asked about was buried. The fix is to prefix
   `"<section> — <scheme>: "`.

   The subtlety: prefixing **only the vector** fixed retrieval but left the model
   reading a bare label/value list — `Nil` / `Stamp duty on investment:` /
   `0.005% (from July 1st, 2020)` / `Tax implication: …` — with nothing telling it
   what it was looking at. Asked "What is the stamp duty and tax implication on
   redemption?" the model replied *"Not stated in the sources I have"* while the
   fact sat in context block 1. Isolating it in a controlled call showed the same
   chunk answered correctly with the header prefixed and refused without it.

   So the prefixed string is stored as the Chroma `document` as well as embedded.
   **What the LLM reads must match what we embedded** — otherwise retrieval and
   generation disagree about what a chunk means.

Result: **36 chunks** across 5 documents, and the stamp-duty question went from
an empty retrieval to a correct answer citing the right scheme.

### 3.4 EMBED + STORE

- `SentenceTransformer(EMBED_MODEL)`, `normalize_embeddings=True` so cosine ==
  dot product and Chroma is configured with `hnsw:space: cosine`.
- The **same model instance** encodes chunks and questions (cached with
  `lru_cache`), as the brief requires.
- `PersistentClient` at `data/chroma/`, collection `hdfc_mf_faq`.
- **Idempotency:** if the collection already holds ≥ the chunk count, `ingest()`
  skips and says so. `--rebuild` forces a re-embed. Chroma is git-ignored.

## 4. Query design

### 4.1 Guard order (deliberate)

Guards run **before** embedding and before any LLM call, so a PII-bearing
question is never encoded, never sent to Groq, and never logged.

| Order | Guard | Trigger | Response |
|---|---|---|---|
| 0a | **PII** | PAN, Aadhaar, bank account, OTP, card, email, phone regexes | Refuse; ask the user to remove it. Nothing stored. |
| 0b | **Advice** | "should I buy", "which fund is best", "recommend", "book profit", "exit my investment", "help me choose", … | Polite facts-only message + an educational link. |
| 0c | **Performance** | "5 year returns vs category", "expected return", "CAGR", "outperform", "how much will I make" | Refuse; point at the official factsheet / fund page. |
| 1 | **Grounding** | top-1 cosine < `MIN_SIM` (0.22) | "I don't have that in my 5 indexed HDFC scheme pages." |
| 2 | **Output re-check** | the *generated* answer trips an advice/performance regex | Replaced with a facts-only message. |
| 3 | **Length** | any answer | Hard-capped at 3 sentences by `_trim_sentences`. |

The output re-check matters because a model can drift into advice even when the
question and context are clean.

### 4.2 Retrieval

- Same MiniLM model encodes the question; cosine top-k from Chroma.
- **Adaptive k.** `k = 5` when the question names a scheme (we scope to it), `k = 8`
  when it does not — a scheme-less question competes against all 5 schemes' chunks,
  and 5 was not enough to surface the stamp-duty/tax block.
- **Scheme scoping.** `detect_scheme()` matches aliases from `config.SCHEME_ALIASES`
  (`large cap`, `flexi cap`, `elss` / `tax saver`, `small cap`, `balanced
  advantage`) and filters the hits to that scheme. If scoping would empty the
  result set, the unscoped hits are returned anyway.

### 4.3 Cross-scheme contamination — the bug that shaped the design

All five pages are HDFC and use near-identical wording ("…is rated Very High risk.
Minimum SIP Investment is set to ₹100."). With plain top-k retrieval:

- *"What is the exit load on HDFC **Small Cap**…?"* retrieved **Large Cap**'s
  exit-load chunk at 0.841 and cited it. The number happened to be the same, so
  the answer looked right — which is exactly what makes this class of bug
  dangerous.
- *"What is the benchmark of HDFC **Flexi Cap**…?"* was answered from a chunk
  belonging to a different scheme, cited to a glossary block.

Fixes, both in place:
1. **Retrieval-side:** `detect_scheme` + filtering, so the context handed to the
   model is single-scheme whenever the question names a scheme.
2. **Prompt-side:** an explicit rule — *"If the question names a scheme, use ONLY
   context blocks for that scheme. Never blend facts across schemes."*

Verified after the fix: the Small Cap exit-load question now cites the Small Cap
chunk, and the Flexi Cap benchmark cites Flexi Cap's *About the scheme* block.

### 4.4 Citation selection

`best_citation()` returns the top-ranked hit, unless the only candidate is a
`Glossary (Understand terms)` block. The glossary *defines* terms ("Expense
ratio: a fee payable to a mutual fund house…") and is a legitimate retrieval hit,
but citing it for "what is the expense ratio of X" is unhelpful when a
fact-bearing chunk was also retrieved. The section label is retained in metadata
for exactly this decision.

### 4.5 Prompt and answer assembly

System prompt (in `config.SYSTEM_PROMPT`, 10 numbered rules) forbids: outside
knowledge, guessing numbers, advice, performance claims, blending schemes,
adding a URL, and exceeding three sentences. Context blocks are labelled
`[n] SOURCE: <scheme> | SECTION: <section>` so the model can attribute correctly.

One rule earns its own line because it was added in response to a failure:

> **8.** If the question does **not** name a scheme but the context states the
> fact for one or more schemes, answer anyway: name the scheme(s) you used, and if
> the value is identical across them say it applies to all of them. Do NOT reply
> "not stated" merely because no scheme was named. Reserve "not stated" for facts
> that are genuinely absent from the context.

Without it, a scheme-less question (*"What is the stamp duty and tax implication
on redemption?"*) was treated as unanswerable and refused, even with the correct
block at rank 1 — the model was hedging because five near-identical schemes were
in context and it could not tell which one the question was about. Refusing is
the right default, but an over-eager refusal on a fact we demonstrably have is
just as bad as a hallucination.

The answer text returned to the UI **contains no URL**. The link is rendered by
the app from Chroma metadata. This is a structural guarantee that a
hallucinated citation is impossible, not a prompt instruction.

**No-key fallback.** With no `GROQ_API_KEY`, `ask()` returns an *extractive*
answer: it scopes to the top scheme, re-joins label/value pairs, ranks facts by
lexical overlap with the question's content terms (plus a small synonym map:
`riskometer → risk`, `charges → expense/load`, …), and quotes up to two verbatim.
It is explicitly prefixed *"Extracted from the source — no LLM key configured"*.
A fact with **zero** lexical link to the question is never quoted, which is how
"lock-in period" correctly returns a miss rather than an unrelated risk line.
Retrieval, chunking, embedding and all three guards work without a key; only
generation is degraded. This is a demo affordance, not a product feature.

## 5. Data model

### 5.1 `sources.csv` — the corpus declaration

| Column | Purpose |
|---|---|
| `id` | stable ordinal |
| `scheme` | display name, used as the citation label and for scheme scoping |
| `category` | Large Cap / Flexi Cap / ELSS / Small Cap / Hybrid |
| `plan` | Direct Growth / Direct Plan Growth |
| `url` | the public page |
| `publisher` | Groww (public scheme page) |
| `retrieved_on` | drives the "Last updated from sources" stamp |

### 5.2 Chroma document and metadata

| Field | Where | Used by |
|---|---|---|
| `id` (document) | `chunk_id` = `<scheme>#<n>` | debugging, dedupe on re-ingest |
| `document` | raw chunk body | shown in the UI expander |
| `metadata.source_url` | from `sources.csv` | **the citation link** |
| `metadata.scheme` | from `sources.csv` | citation label, scheme scoping |
| `metadata.category`, `metadata.plan` | from `sources.csv` | context for the model |
| `metadata.section` | derived at chunk time | citation label, glossary suppression |
| `metadata.retrieved_on` | from `sources.csv` | freshness stamp |

### 5.3 Data that never exists

No user table. No chat-history file. No PII store. Streamlit session state holds
the conversation in memory for the lifetime of the process only.

## 6. Error and edge paths

| Condition | Behaviour | Where |
|---|---|---|
| Fetch fails / blocked | Non-zero exit with the URL named; no partial corpus | `fetch_sources.py` |
| Vector store empty | Sidebar error: "Run `python app/ingest.py`"; UI stops | `app.py` |
| `GROQ_API_KEY` missing | Sidebar warning; extractive fallback; answers labelled as extracts | `rag.ask` |
| Groq 404 / rate limit / network | Caught, human-readable message, retrieved chunks still shown | `rag._llm_answer` |
| Retrieval returns nothing above `MIN_SIM` | Honest miss + official link | `rag.ask` |
| Model says "not stated in the sources" | Converted to an `unknown` answer naming the closest section | `rag.ask` |
| Model drifts into advice | Replaced with a facts-only message | `rag.ask` |
| Question names two schemes | Scoping disabled, all hits returned, model told not to blend | `detect_scheme` |
| Any PII pattern in the question | Refused before embedding | `rag.ask` |

## 7. Extension points

| Want to add | Change |
|---|---|
| A sixth scheme | Append a row to `sources.csv`, add aliases to `SCHEME_ALIASES`, re-run fetch + ingest. |
| HDFC factsheets / KIM / SIDs (PDF) | Add `pypdf` to the loader; point rows at the PDFs. Fixes the ELSS lock-in miss. |
| "How do I download a statement?" | Add the account-help page to `sources.csv` — it is a Groww/HDFC flow, not scheme data. |
| Better retrieval at scale | Swap top-k for a cross-encoder reranker. Not needed at 36 chunks. |
| Fresh numbers on a schedule | Re-run `fetch_sources.py` → `ingest.py` on a timer. Needs hosting. |
| Multi-turn | Thread prior turns into the query embedding, or summarise the session. |

## 8. Known limits

Carried from `README.md` §9, restated as architecture facts:

1. **ELSS lock-in period is not in the corpus** → honest miss by design.
2. **Statement-download flow is not in the corpus** → honest miss by design.
3. **No multi-turn memory** — each question is independent.
4. **Figures are a 2026-09-29 snapshot**; no scheduler.
5. **Groww is a distributor, not the AMC.** The brief said AMC/SEBI/AMFI; a
   production build should ingest `hdfcfund.com` documents and AMFI NAV data.
6. **No reranking** — plain cosine, fine at 36 chunks, not at thousands.
7. **The no-key extractive fallback is weaker than the LLM path.** It exists so the
   demo is not dead without a key.
8. **No auth, no rate limiting** — local single-user demo.
9. **Loader is coupled to Groww's DOM**; a layout change needs a re-run. Caching
   to `data/raw/` limits the blast radius.
