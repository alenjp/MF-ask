# Implementation Plan — Mutual Fund FAQs RAG Chatbot

**Purpose:** A phase-by-phase build guide, derived from
[`architecture.md`](architecture.md), that a coding agent (Cursor, etc.) can
follow to implement the prototype in the same order and with the same decisions
that produced the shipped, verified v1.

**How to use with Cursor:** Give it this file and `architecture.md` in the
context (`@implementation.md @architecture.md @PRD.md`), then let it execute one
phase at a time. **Stop at the end of every phase and run that phase's
acceptance check before starting the next.** The phases are ordered so each ends
with something runnable and inspectable.

**Read first:** `architecture.md` §1 (system shape) and §2 (module map). Every
phase's "Design reference" column points at the relevant sections.

---

## Golden rules (apply in every phase)

1. **Plain Python, no framework.** No LangChain, no orchestration. The whole
   pipeline is ~200 lines so a presenter can whiteboard it in under two minutes.
2. **Store what you embed.** The Chroma `document` must be the same
   `"<section> — <scheme>: <body>"` string that was embedded. If they diverge,
   retrieval and generation disagree about what a chunk means (see Phase 3).
3. **Sensible defaults live in `app/config.py`, never hardcoded in logic.**
4. **Failure is loud.** A fetch error exits non-zero rather than ingesting a
   partial corpus.
5. **`python` = Python 3.12.** `chromadb` and `sentence-transformers` need
   compiled wheels that 3.13/3.14 may not have. Use `py -3.12` or an explicit
   interpreter path.
6. **Never commit `.env`.** Git-ignore it from Phase 0.

---

## Phase 0 — Environment, repo skeleton, config

**Goal:** a Python 3.12 venv that imports the heavy deps, and a `config.py` that
holds every shared constant.

| Item | Detail |
|---|---|
| Files | `.gitignore`, `.env.example`, `requirements.txt`, `.env`, `app/__init__.py`, `app/config.py`, `sources.csv` (empty or stub) |
| Design reference | architecture.md §2 (module map, config row) |

**Steps**

1. `.gitignore`: exactly
   ```
   .env
   data/chroma/
   __pycache__/
   *.pyc
   .streamlit/secrets.toml
   ```
2. `requirements.txt` (the shipped set — `lxml` is the `BeautifulSoup` parser):
   ```
   beautifulsoup4>=4.12
   chromadb>=1.0
   groq>=0.13
   lxml>=5.0
   python-dotenv>=1.0
   requests>=2.31
   sentence-transformers>=3.0
   streamlit>=1.40
   ```
3. `.env.example` (commit this, never the real `.env`):
   ```
   GROQ_API_KEY=gsk_your_key_here
   GROQ_MODEL=qwen/qwen3.8-27b
   ```
4. `app/config.py` — every constant below, with the rationale as a comment:
   | Constant | Value | Why |
   |---|---|---|
   | `ROOT` | `Path(__file__).resolve().parent.parent` | repo root regardless of CWD |
   | `EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | fixed by the brief, 384-dim, local |
   | `COLLECTION` | `hdfc_mf_faq` | |
   | `GROQ_MODEL` | `os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")` | the only model this key serves |
   | `TOP_K` / `TOP_K_GENERIC` | `5` / `8` | adaptive k — scoped vs scheme-less |
   | `MIN_SIM` | `0.22` | cosine floor below which we say "not in the sources" |
   | `CHUNK_CHARS` / `CHUNK_OVERLAP_CHARS` | `900` / `140` | see Phase 2 |
   | `MAX_SENTENCES` | `3` | requirement FR-6 |
   | `NON_CITABLE_SECTIONS` | `{"Glossary (Understand terms)"}` | see Phase 5 |
   | `DISCLAIMER`, `NO_ADVICE_REPLY`, `EDUCATION_LINK`, `EDUCATION_LABEL` | copy in config.py | user-facing copy in one place |
   | `SCHEME_ALIASES` | 5 schemes × aliases (`large cap`, `flexi cap`, `elss`/`tax saver`, `small cap`, `balanced advantage`) | keys are the display names from `sources.csv` |
5. `app/config.py` calls `load_dotenv()` at import.

**Acceptance check**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -c "import chromadb, sentence_transformers, groq, streamlit, dotenv; print('ok')"
python -c "from app import config; print(config.GROQ_MODEL, config.TOP_K, config.TOP_K_GENERIC)"
```

---

## Phase 1 — Corpus and LOAD stage

**Goal:** a declarative `sources.csv` and a loader that turns 5 public Groww
pages into clean `data/raw/*.txt`.

| Item | Detail |
|---|---|
| Files | `sources.csv`, `data/raw/`, `app/fetch_sources.py` |
| Design reference | architecture.md §3.1, §5.1 |

**Steps**

1. `sources.csv` — one AMC (HDFC), five schemes, all Direct–Growth, one public
   page each. Columns: `id, scheme, category, plan, url, publisher,
   retrieved_on`. Use exactly these display names (they become citation labels
   and the keys of `SCHEME_ALIASES`):
   - HDFC Large Cap Fund (Direct Growth) — Large Cap
   - HDFC Flexi Cap Fund (Direct Growth) — Flexi Cap
   - HDFC ELSS Tax Saver Fund (Direct Plan Growth) — ELSS
   - HDFC Small Cap Fund (Direct Growth) — Small Cap
   - HDFC Balanced Advantage Fund (Direct Growth) — Hybrid (Balanced Advantage)

   Put the 5 URLs from PRD.md §5 in it. `retrieved_on` = today's date; it drives
   the freshness stamp everywhere.

2. `app/fetch_sources.py`:
   - `requests` + `BeautifulSoup`. Drop `header/nav/footer/script/style/form`.
   - Slice the body at `FOOTER_MARKS = ("About Us", "Version:", "© 2016")` and
     drop mega-menu link rows.
   - Write one file per scheme to `data/raw/<slug>.txt` with a header block
     `SOURCE_URL / SCHEME / CATEGORY / PLAN`.
   - Fail non-zero (loud) if any fetch fails. Never write a partial corpus.
   - Cache to `data/raw/` so re-runs are cheap and text is diff-reviewable.

**Acceptance check**

```powershell
python -m pip install -r requirements.txt   # once in phase 0 is enough
python app/fetch_sources.py
Get-ChildItem data\raw   # exactly 5 .txt files
```

Read one file — a presenter must be able to see the source facts as plain lines:
`Min. for SIP` / `₹100` / `Exit load` / `Nil` / `Riskometer` etc. Grep the actual
values you will later answer (expense ratio 1.03%, ELSS min SIP ₹500, benchmark
indexes) so you know the corpus really contains them.

---

## Phase 2 — CHUNK stage

**Goal:** section-aware, line-granular chunking that produces **exactly 36
chunks**, and a human-readable dump.

| Item | Detail |
|---|---|
| Files | `app/chunking.py`, `data/chunks.txt`, `app/utils.py` (optional shared `retrieved_on`) |
| Design reference | architecture.md §3.2, §3.3 |

**Steps** — implement the shipped decision table exactly:

| Decision | Value |
|---|---|
| Split unit | single line (a label and its value must never be separated) |
| Packing | section-aware — detected headings flush the buffer; one chunk = one readable fact group |
| Chunk size | 900 chars (~225 tokens) |
| Overlap | 140 chars, **line-granular** — re-emit the last whole lines, never cut mid-label |
| Dropped sections | Holdings, Return calculator, Returns/rankings, Compare-similar, Fund management (identical `DROP_BLOCK_PATTERNS` present in the raw pages) |
| Metadata | `chunk_id (<scheme>#<n>), source_url, scheme, category, plan, section, n_chars, retrieved_on` |

**The three decisions that were made by testing — implement them deliberately:**

1. **900 not 700.** At 700, a scheme's *About the scheme* block splits and the
   benchmark line is orphaned into a chunk that never ranks in the top-10. 900 keeps
   objective + risk + minimums + benchmark together.
2. **Sub-headings must not demote the parent section.** The pages repeat a bare
   parent heading *inside* a subsection: `Exit load`, then
   `Exit load, stamp duty and tax`, then `Exit load` *again*. The last one
   resets the section. Fix: track the raw heading **key** and ignore any heading
   that is a *prefix* of the current one. Check your dump — all five schemes must
   carry an `Exit load, stamp duty & tax` block, not a generic `Exit load &
   charges`.
3. **`data/chunks.txt` is written before embedding** — it is the audit artefact
   and the corpus map. Chunks are labelled with translated section names
   (`Fees: expense ratio`, `Exit load, stamp duty & tax`, `Minimum investments`,
   `About the scheme`, `Glossary (Understand terms)`, …).

**Acceptance check**

```powershell
python -c "import sys; sys.path.insert(0,'app'); from config import ROOT; from ingest import retrieved_on; from chunking import build_all_chunks; print(len(build_all_chunks(retrieved_on())))"
```

Expected: **36** (5 docs → 36 chunks; ~87% of each raw page is dropped noise —
that is by design). Inspect `data/chunks.txt` and confirm the stamp-duty block
exists under all five schemes and the benchmark sits inside *About the scheme*
for each.

---

## Phase 3 — EMBED + STORE

**Goal:** 36 vectors + documents in a persistent ChromaDB collection, ingested
once and skippable on re-run.

| Item | Detail |
|---|---|
| Files | `app/ingest.py` |
| Design reference | architecture.md §3.4, §5.2 |

**Steps**

1. `SentenceTransformer(EMBED_MODEL)` with `normalize_embeddings=True`; Chroma
   configured `hnsw:space: cosine`.
2. The **same model instance** (cache with `lru_cache`) is reused later for
   questions — the brief requires chunk and query to share the model.
3. `PersistentClient` at `data/chroma/`, collection `hdfc_mf_faq`.
4. **Store what you embed.** The embedded *and* stored document is
   `f"{section} — {scheme}: {body}"`. This is what fixes the "not stated" refusal
   when a bare label/value list sits in the context: with the header the model
   knows what section it is reading. (architecture.md §3.3.3.)
5. Metadata per chunk: `source_url, scheme, category, plan, section,
   retrieved_on` (citation and scoping are metadata-driven, never model-written).
6. Idempotency: if collection count >= chunk count, skip with a message.
   `--rebuild` deletes and re-embeds. `data/chroma/` is git-ignored.

**Acceptance check**

```powershell
python app/ingest.py            # first run: embeds 36
python app/ingest.py            # second run: prints skip message, count unchanged
python -c "import chromadb; c=chromadb.PersistentClient(path='data/chroma'); print(c.get_collection('hdfc_mf_faq').count())"   # 36
```

---

## Phase 4 — GUARD + RETRIEVE

**Goal:** every guard in the shipped order, adaptive-k retrieval, scheme scoping.

| Item | Detail |
|---|---|
| Files | `app/rag.py` (guards + retrieval only; generation comes in Phase 5) |
| Design reference | architecture.md §4.1, §4.2, §4.3 |

**Steps**

1. **Guard order (deliberate — guards run *before* embedding and any LLM call):**
   | Order | Guard | Trigger |
   |---|---|---|
   | 0a | PII | regex: PAN `\b[ABCDEFGHJKLMNPQRSTUVW]{5}\d{4}[A-Z]\b`, Aadhaar `\b\d{4}\s?\d{4}\s?\d{4}\b`, bank account `[6-9]\d{11}`, OTP `\b\d{6}\b`, email, phone `(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{9}(?!\d)\b`, card `\b\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\b` |
   | 0b | Advice | "should I buy", "which fund is best", "recommend", "book profit", "exit my investment", "help me choose", … → `NO_ADVICE_REPLY` + `EDUCATION_LINK`; retrieval never runs |
   | 0c | Performance | "5 year returns vs category", "expected return", "CAGR", "outperform", "how much will I make", … → refuse, point at the official factsheet/fund page |
2. **Retrieval:** question embedded with the cached Phase-3 model; cosine top-k.
   Context window of 10 chunks for every question (`TOP_K` = `TOP_K_GENERIC` = 10);
   a scheme-less question competes against all five schemes.
3. **Scheme scoping** — the bug that shaped the design (architecture.md §4.3).
   `detect_scheme()` matches `SCHEME_ALIASES`; if a scheme is found, filter hits
   to that scheme. If scoping would empty the result set, fall back to unscoped.
   Why: all five pages read almost identically; plain top-k answered "exit load
   on HDFC **Small Cap**" from **Large Cap**'s chunk at 0.841 and cited it.
4. Grounding floor: top-1 cosine < `MIN_SIM` → honest miss.

**Acceptance check** — hit the Chroma console before writing any prompt code:

```powershell
python -c "import sys; sys.path.insert(0,'app'); from rag import detect_scheme, retrieve; q='What is the exit load on HDFC Small Cap Fund Direct Growth?'; print(detect_scheme(q)); [print(h.scheme) for h in retrieve(q, k=10)]"
```

All 5 returned schemes must be the Small Cap one. Repeat with the Flexi Cap
benchmark question and confirm scoping selects Flexi Cap.

---

## Phase 5 — GENERATE + answer shaping

**Goal:** a 10-rule system prompt, one metadata-driven citation, ≤3 sentences,
and a no-key fallback that never fabricates.

| Item | Detail |
|---|---|
| Files | `app/rag.py` (`SYSTEM_PROMPT`, `_llm_answer`, `best_citation`, `_extractive_fallback`, `ask`) |
| Design reference | architecture.md §4.4, §4.5 |

**Steps**

1. `SYSTEM_PROMPT` (10 numbered rules, in `rag.py`, not `config.py`):
   1. Answer ONLY from the CONTEXT blocks (verbatim extracts), labelled
      `[n] SOURCE: <scheme> | SECTION: <section>`.
   2. Never use outside knowledge. Never guess a number.
   3. NO investment advice — no should/ought/best/buy/sell/hold/allocate.
   4. NO performance claims — never compute/compare/rank/forecast returns.
   5. At most 3 sentences.
   6. If the question names a scheme, use ONLY that scheme's context blocks.
   7. Name the scheme explicitly in the answer.
   8. **Scheme-less questions:** if the context states the fact, answer anyway and
      name the scheme(s) used; if identical across schemes, say it applies to
      all. Do NOT say "not stated" merely because no scheme was named. (Added
      because the model refused *"What is the stamp duty…?"* with the fact in
      rank-1 context — an over-eager refusal is as bad as a hallucination.)
   9. Do not add a source URL — the app renders the citation.
   10. Keep the tone factual and neutral.
2. Groq call: `Groq(api_key=_).chat.completions.create(model=GROQ_MODEL,
   temperature=0.0, messages=[system, user])`. Wrap in try/except — on 404
   (wrong model), rate limit, or network, return a human-readable message and
   still show the retrieved chunks.
3. **Citation:** `best_citation()` returns the top hit, unless the only candidate
   is a `NON_CITABLE_SECTIONS` block (glossary). The model never writes URLs; the
   app renders them from `metadata.source_url`.
4. **Output re-check:** if the *generated* answer still trips the advice or
   performance regex, replace it with the facts-only message.
5. **Length:** hard 3-sentence cap via `_trim_sentences` (split on sentence
   boundaries, join first `MAX_SENTENCES`).
6. **No-key fallback:** with no `GROQ_API_KEY`, return an *extractive* answer —
   scope to the top scheme, re-join label/value pairs, rank facts by lexical
   overlap with the question's content terms (small synonym map:
   `riskometer → risk`, `charges → expense/load`), quote ≤2 verbatim, prefix
   *"Extracted from the source — no LLM key configured"*. A fact with zero
   lexical link is never quoted (this is how "lock-in period" returns a miss).
7. `ask()` returns a typed `Answer`: `(text, kind, citation, top_k)` where kind ∈
   `answer | unknown | out_of_scope | performance | pii` — the tests depend on it.

**Acceptance check**

```powershell
python -c "import sys; sys.path.insert(0,'app'); from rag import ask
for q in ['What is the expense ratio of HDFC Large Cap Fund Direct Growth?',
          'What is the stamp duty and tax implication on redemption?',
          'What is the lock-in period for HDFC ELSS Tax Saver Fund?',
          'My PAN is ABCDE1234F, what is the exit load?']:
    a=ask(q); print(a.kind, '|', a.text[:80])"
```

Expect `answer / answer / unknown / pii`. The stamp-duty scheme-less question
must now come back with an answer naming the schemes, not "Not stated".

---

## Phase 6 — Streamlit UI

**Goal:** the FR-12/13/14/15 tiny UI — welcome line, 3 examples, disclaimer,
freshness stamp, corpus status, inspectable context.

| Item | Detail |
|---|---|
| Files | `app/app.py` |
| Design reference | architecture.md §1 (PRESENT box) |

**Steps**

1. `st.set_page_config`; `index_ready()` from `rag` to get chunk/page counts.
2. Sidebar: corpus status (`Indexed **{n_chunks}** chunks from **5** public HDFC
   scheme pages`), model name `GROQ_MODEL` from config, key warning if `GROQ_API_KEY`
   is missing, the `DISCLAIMER` caption.
3. Chat: welcome line, five example buttons (expense ratio, min SIP, lock-in,
   advice refusal, PII refusal), each user message appended with role + content.
4. Every answer footer: `Last updated from sources: <retrieved_on>`.
5. Grounded answers render the **one source link** from `hit.metadata.source_url`
   (never from the model text) and a `st.expander` of the retrieved context
   blocks so grounding is inspectable (FR-15).
6. Types map to distinct UI text: `unknown` → honest miss naming the closest
   section; refusals → the guard copy.

**Acceptance check**

```powershell
streamlit run app/app.py
```

Open the browser: sidebar shows 36/5, the three example buttons work, answers
cap at 3 sentences, every answer has exactly one link + freshness stamp. On a
machine with no key you must still see a working extractive answer plus the key
warning.

---

## Phase 7 — Evaluation and artefacts

**Goal:** a repeatable smoke test, a sample-Q&A artefact regenerated from real
answers, and the doc set.

| Item | Detail |
|---|---|
| Files | `app/evaluate.py`, `SAMPLE_QA.md` (generated), `README.md`, `DISCLAIMER.md` |
| Design reference | PRD.md §9 (deliverables), §10 (demo script) |

**Steps**

1. `app/evaluate.py` — a table of 12 `(question, expected_kind)` pairs covering
   every behaviour:
   | Question | expected kind |
   |---|---|
   | What is the expense ratio of HDFC Large Cap Fund Direct Growth? | answer |
   | What is the minimum SIP amount for HDFC ELSS Tax Saver Fund Direct Plan Growth? | answer |
   | What is the exit load on HDFC Small Cap Fund Direct Growth? | answer |
   | What is the benchmark of HDFC Flexi Cap Fund Direct Growth? | answer |
   | What is the risk category / riskometer level of HDFC Balanced Advantage Fund? | answer |
   | Who is the fund manager of HDFC Large Cap Fund Direct Growth? | answer |
   | What is the stamp duty and tax implication on redemption? | answer |
   | What is the lock-in period for HDFC ELSS Tax Saver Fund? | unknown |
   | Should I buy HDFC Small Cap Fund for my portfolio? | out_of_scope |
   | What are the 5 year returns of HDFC Large Cap Fund versus the category? | performance |
   | My PAN is ABCDE1234F, what is the exit load? | pii |
   | What is the weather in Mumbai? | unknown |
2. Report `guard routing: N/12 as expected`, print each failing question, exit
   non-zero on failure.
3. Regenerate `SAMPLE_QA.md` from the same run: queries, answer kinds, answers,
   source links, freshness date.
4. Write `README.md` (setup incl. the Python 3.12 note, run, scope, chunking
   rationale, the guard table, known limits) and `DISCLAIMER.md`.
5. `PRD.md` and `architecture.md` already exist as the source docs — do not
   regenerate; cross-check the numbers in them (36 chunks, 5 pages, model, dates)
   still match this build.

**Acceptance check**

```powershell
python app/evaluate.py
```

Expected: `[eval] guard routing: 12/12 as expected` and a refreshed
`SAMPLE_QA.md`. This is the repeatable proof for the evaluator and the last
checklist item before the demo.

---

## Final acceptance checklist (maps to PRD FRs)

| Check | Expected | PRD ref |
|---|---|---|
| `python app/fetch_sources.py` then `python app/ingest.py` | 5 files → 36 chunks → Chroma count 36, skip on re-run | FR-1..5 |
| Expense ratio question | 1.03%, one Groww link, ≤3 sentences | FR-6, FR-7, FR-16? |
| ELSS min SIP question | ₹500 | FR-6 |
| Small Cap exit load question | cited to Small Cap exactly | FR-2, cross-scheme |
| Scheme-less stamp-duty question | answered with scheme names, not "not stated" | FR-6, prompt rule 8 |
| Lock-in question | honest miss | FR-11 |
| "Should I buy…" | refusal + educational link | FR-8 |
| "5 year returns…" | performance refusal → factsheet | FR-10 |
| "My PAN is ABCDE1234F…" | PII refusal, nothing stored | FR-9 |
| `data/chunks.txt` | 36 human-readable chunks | FR-5 |
| Every answer footer | "Last updated from sources: <date>" | FR-13 |
| Sidebar | "Indexed 36 chunks from 5 public HDFC scheme pages" | FR-14 |
| `python app/evaluate.py` | 12/12 | FR-17 |
| No `.env` in git status | absent | NFR-4 |