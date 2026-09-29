# HDFC Mutual Fund FAQ Assistant — a RAG chatbot

A small facts-only question-answering bot over **five public HDFC Mutual Fund
scheme pages**. It answers scheme-mechanics questions (expense ratio, exit load,
minimum SIP, benchmark, risk level, tax/stamp duty), cites **one source link per
answer**, and refuses advice-seeking and PII-bearing questions.

Built as a classroom RAG demo: **Ingestion = Load → Chunk → Embed → Store**,
**Query = Question → Embed → Retrieve top-k → LLM → Answer + citation**.

---

## 1. Scope

**AMC:** HDFC Mutual Fund (one AMC, five schemes, all Direct – Growth plans)

| # | Scheme | Category | Source |
|---|--------|----------|--------|
| 1 | HDFC Large Cap Fund – Direct Growth | Large Cap | <https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth> |
| 2 | HDFC Flexi Cap Fund – Direct Growth | Flexi Cap | <https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth> |
| 3 | HDFC ELSS Tax Saver Fund – Direct Plan Growth | ELSS | <https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth> |
| 4 | HDFC Small Cap Fund – Direct Growth | Small Cap | <https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth> |
| 5 | HDFC Balanced Advantage Fund – Direct Growth | Hybrid (Balanced Advantage) | <https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth> |

The machine-readable list is [`sources.csv`](sources.csv).
Sample Q&A is [`SAMPLE_QA.md`](SAMPLE_QA.md). Disclaimer: [`DISCLAIMER.md`](DISCLAIMER.md).

---

## 2. Stack (as required)

| Stage | Choice |
|-------|--------|
| Embedding | `sentence-transformers/all-MiniLM-L6-v2` — local, 384-dim, no API key. Same model encodes chunks and questions. |
| Vector DB | ChromaDB, `PersistentClient` persisted to `data/chroma/` — ingest runs once, not on every restart. |
| LLM | Groq (`llama-3.3-70b-versatile`, temperature 0.0), key in `.env`, `.env` is git-ignored. |
| UI | Streamlit |

No orchestration framework — the pipeline is ~200 lines of plain Python in
`app/`, so the RAG loop is explainable on a whiteboard.

---

## 3. Setup (Windows / PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

Copy-Item .env.example .env     # then paste your Groq key into .env
```

## 4. Run

```powershell
python app\fetch_sources.py     # 1. LOAD  - fetch 5 public pages -> data\raw\*.txt
python app\ingest.py            # 2. CHUNK + 3. EMBED + 4. STORE -> data\chunks.txt + data\chroma\
streamlit run app\app.py       # 5. chat UI on http://localhost:8501
```

`app\ingest.py --rebuild` forces a re-embed. Re-run `fetch_sources.py` to
refresh the numbers (NAV, AUM, expense ratio) and then re-ingest.

Optional: `python app\evaluate.py` runs 12 smoke-test queries and regenerates
`SAMPLE_QA.md`.

### The 60-second demo script

1. Open the app. Sidebar shows *"Indexed 36 chunks from 5 public HDFC scheme
   pages"*. Point at the pipeline diagram in §5.
2. Click the example button **"What is the expense ratio of HDFC Large Cap Fund
   Direct Growth?"** → grounded answer + one Groww link + top-5 chunks in the
   expander.
3. Type **"What is the lock-in period for HDFC ELSS Tax Saver Fund?"** → honest
   miss ("Not stated in the sources I have"). This is the beat that proves it
   isn't hallucinating.
4. Type **"Should I buy HDFC Small Cap Fund for my portfolio?"** → refusal +
   educational link.
5. Type a message containing a PAN → PII refusal.

---

## 5. Architecture

```
                 INGESTION (run once)
  5 public URLs
      │  app/fetch_sources.py      LOAD    requests + BeautifulSoup, strip nav/footer
      ▼
  data/raw/*.txt                   5 cleaned text files
      │  app/chunking.py           CHUNK   section-aware packing, 700 chars / 140 overlap
      ▼
  data/chunks.txt                  human-readable dump of every chunk + metadata
      │  app/ingest.py             EMBED   all-MiniLM-L6-v2 -> 384-dim vectors
      ▼
  data/chroma/                     STORE   ChromaDB persistent collection

                 QUERY (per question)
  question
      │  guards: PII -> advice -> performance        (before any LLM call)
      ▼
  app/rag.py  retrieve()           EMBED question, cosine top-5 from Chroma
      │
      ▼  context blocks + system prompt
  Groq (temperature 0.0)            GENERATE <= 3 sentences, no URL in text
      │
      ▼
  answer + 1 citation link  +  "Last updated from sources: 2026-09-29"
```

### Files

| File | Role |
|------|------|
| `app/config.py` | paths, model names, thresholds, system prompt, disclaimer |
| `app/fetch_sources.py` | **LOAD** — fetch + clean the 5 public pages |
| `app/chunking.py` | **CHUNK** — section-aware packing + `data/chunks.txt` dump |
| `app/ingest.py` | **EMBED + STORE** — MiniLM vectors into persistent Chroma |
| `app/rag.py` | guards + **RETRIEVE** + **GENERATE** + citation assembly |
| `app/app.py` | Streamlit UI (welcome line, 3 examples, disclaimer) |
| `app/evaluate.py` | 12-query smoke test, writes `SAMPLE_QA.md` |
| `sources.csv` | the 5 source URLs |
| `data/chunks.txt` | every chunk, inspectable before embedding |

---

## 6. Chunking strategy (decided after inspecting the data)

I fetched the pages and read the raw text before writing the chunker. Findings:

- These are **not prose pages**. Each is a stack of one-fact-per-line rows:
  `Min. for SIP` / `₹100` / `Fund size (AUM)` / `₹39,933.36 Cr` / …
- There are **no blank-line paragraphs** to split on, so the indivisible unit is
  the *line*, not the paragraph. A label and its value must never be separated.
- The bulk of each page is a 50-row holdings table, a return calculator, a peer
  comparison table and a fund-manager link list — roughly 75% of the characters,
  and none of it useful for a scheme-mechanics FAQ.

Decisions:

| Parameter | Value | Why |
|-----------|-------|-----|
| Split unit | single line | keeps `label` and `value` adjacent |
| Chunk size | **900 characters** (~225 tokens) | 700 orphaned each scheme's benchmark into a second, weaker-ranked chunk, so the query never retrieved it. 900 holds a whole *About the scheme* block (objective + risk + minimums + benchmark for one scheme) while still keeping top-5 retrieval well under one page |
| Overlap | **140 characters**, block-granular | re-emits the *last whole lines* of the previous chunk, so overlap never cuts mid-label |
| Boundary | detected headings flush the buffer | each chunk is one readable section |
| Dropped | Holdings, Return calculator, Returns/rankings, Compare-similar, Fund-management link lists | out of scope; keeping them let 87/132 chunks be portfolio noise that crowded out the fee facts |
| Embedded text | `"<section> — <scheme>: <body>"` | every raw chunk repeats the scheme name, so *"HDFC ... Fund"* dominated the cosine score and buried the section term the user actually asked about. The header puts the question-relevant words first. Body + citation unchanged |
| Metadata per chunk | `chunk_id, source_url, scheme, category, plan, section, n_chars, retrieved_on` | every chunk is independently citable back to exactly one URL |

Result: **36 chunks** across 5 documents, sections labelled *Fees: expense
ratio*, *Exit load & charges*, *Minimum investments*, *Benchmark & investment
objective*, *Key facts*, *Fund house*. Inspect them in `data/chunks.txt`.

---

## 7. Guardrails (all enforced in `app/rag.py`)

| Guard | Trigger | Behaviour |
|-------|---------|-----------|
| **PII** | PAN, Aadhaar, bank account, OTP, card, email, phone regex | Refuses, tells the user to remove it. Nothing is logged or stored. |
| **No advice** | "should I buy", "which fund is best", "recommend", "book profit", "exit my investment", … | Polite facts-only message + an educational link. Retrieval is never even run. |
| **No performance claims** | "5 year returns vs category", "expected return", "CAGR", "outperform", "how much will I make" | Refuses, points at the official factsheet/fund page. The app never computes or compares returns. |
| **Grounding** | top-1 cosine < `MIN_SIM` (0.22) | "I don't have that in my 5 indexed HDFC scheme pages." |
| **Output check** | the generated answer itself trips an advice/performance regex | Replaced with a facts-only message. |
| **Length** | any answer | Hard-capped at **3 sentences** by `_trim_sentences`. |
| **Citation** | every factual answer | Exactly one primary source link rendered under the answer. |
| **Freshness** | every answer | `Last updated from sources: 2026-09-29`. |

The system prompt also forbids outside knowledge and forbids the model from
writing its own URL, so a hallucinated link is not possible — the URL always
comes from the Chroma metadata.

---

## 8. Running without a Groq key

Retrieval, chunking, embedding and all three guards work with no key. `app/rag.py`
falls back to an **extractive** mode that quotes the most on-topic lines from the
top chunks verbatim and labels itself *"Extracted from the source — no LLM key
configured"*. Add a key to `.env` and restart to get the real Groq answers.

---

## 9. Known limits

1. **The lock-in period is not on these pages.** Asking about ELSS lock-in
   returns an honest miss. The answer lives in the Scheme Information Document
   on `hdfcfund.com`; adding that SID/factsheet URL to `sources.csv` is the
   obvious next step (it is a PDF, so `pypdf` would need adding to the loader).
2. **"How do I download a capital-gains statement" is not answerable** from
   scheme pages — that is a Groww/HDFC account-help flow, not scheme data. The
   bot honestly says so rather than guessing. Linking the relevant help page
   would fix it.
3. **No multi-turn memory.** Each question is answered independently.
4. **Numbers go stale.** NAV, AUM, expense ratio and the Groww star rating are a
   snapshot from **2026-09-29**. Re-run `fetch_sources.py` + `ingest.py` to
   refresh. There is no scheduler.
5. **Chunk dropping is opinionated.** Holdings and fund-manager blocks are
   deliberately excluded, so portfolio questions will miss. That is intended,
   but it is a scope decision, not a technical limit.
6. **Groww is a distributor, not the AMC.** The brief said "AMC/SEBI/AMFI"; the
   five URLs supplied are Groww pages. For a production build, swap the loader
   to `hdfcfund.com` factsheets / KIM / SID PDFs and AMFI NAV data.
7. **Reranking is not used** — plain cosine top-5. On a 41-chunk corpus that is
   fine; it would need a cross-encoder at a few thousand chunks.
8. **Extractive fallback quality is poor** compared to the LLM path. It exists
   so the demo is not dead without a key, not as a product feature.
9. **No rate limiting or auth.** Local single-user demo only.

## 10. Data handling

- Public pages only. No login, no app back-end, no screenshots, no third-party
  blogs.
- `.env` is git-ignored; `data/chroma/` is git-ignored.
- No PII is accepted, processed, logged or persisted. The PII guard runs before
  the question is embedded or sent anywhere.
- Chat history lives in the Streamlit session only and is not written to disk.
