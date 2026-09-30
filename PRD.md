# Product Requirements Document
## Mutual Fund FAQs (Facts-Only Q&A) — a RAG Chatbot for Class Demo

**Status:** Implemented (v1)  
**Date:** 29 September 2026  
**Source:** `Problemstatement.txt` (the brief was supplied in the build session; the
file itself was not present in the project folder at the time of writing — this PRD
is derived from the statement as given and should be re-checked if the file is added.)  
**Context:** Build-hour demo. Deliverable is a working RAG prototype plus the
written artefacts listed in §9.

---

## 1. Problem statement

> **Mutual Fund FAQs (Facts-Only Q&A)** — Build a small FAQ assistant that answers
> facts about mutual fund schemes (expense ratio, exit load, minimum SIP, lock-in,
> riskometer, benchmark, how to download statements) using only official public
> pages. Every answer must include one source link. No advice.
>
> **Who this helps:** Retail users comparing schemes; support/content teams
> answering repetitive MF questions.
>
> **Pipeline:** Ingestion `Load → Chunk → Embed → Store in Vector DB`.
> Query `Question → Embed → Retrieve top chunks → LLM → Answer`.
>
> **Tech constraints:** Embedding model `sentence-transformers/all-MiniLM-L6-v2`
> (local, 384-dim, no API key; same model for chunks and questions). Vector DB
> ChromaDB, persisted to disk so ingestion runs once. LLM Groq, key in `.env`,
> never committed to Git. Chunking strategy is the agent's decision, must be
> justified from the data, and all chunks must be dumped to a readable `.txt`.

## 2. Why the existing options fail

| Failure mode | Consequence for this product |
|---|---|
| A generic LLM answers from pretraining | Expense ratios, exit loads and minimums are stated confidently and are often **wrong or out of date**. For a fee question, a wrong number is worse than no answer. |
| No citation | The user cannot verify. Support teams cannot send the link onward. |
| Advice creep | A bot that says "this fund is good for you" is making a regulated-adjacent statement it is not qualified to make. |
| PII handling | Users paste PAN/account numbers into chat boxes. Those must never be stored or forwarded. |
| Performance claims | Comparing or forecasting returns is a performance claim. Out of scope, full stop. |

## 3. Goals and non-goals

### In scope (must ship)

1. A **scoped corpus**: one AMC and 3–5 schemes under it, from 5 public pages.
2. **Facts-only answers** on scheme mechanics, **≤3 sentences**, with **one source link**.
3. **Refusal** on advice/portfolio questions (politely, with an educational link).
4. **PII refusal** on PAN, Aadhaar, account numbers, OTPs, emails, phone numbers.
5. **No performance claims** — never compute, compare or forecast returns; link the
   official factsheet instead.
6. **Honest misses** — say "not stated in the sources" rather than invent.
7. **Tiny UI**: welcome line, 3 example questions, "Facts-only. No investment advice."
8. **Freshness stamp**: `Last updated from sources: <date>`.
9. One-command local run for a presenter.

### Non-goals (explicitly not built)

- Any form of investment advice, recommendation, ranking or portfolio construction.
- Return calculation, comparison, backtesting or forecasting.
- Multi-user auth, roles, cloud accounts, hosted deployment.
- Full-corpus crawling of an AMC's whole site.
- Fine-tuning, custom model training, or an evaluation harness at research scale.
- Real-time NAV feeds (AMFI/AMC APIs) — figures are a dated snapshot.

## 4. Users

| User | Need in the demo |
|------|-------------------|
| **Retail user** comparing schemes | A factual, sourced, verifiable answer on a fee, minimum or lock-in. Nothing else. |
| **Support / content team** | The same repetitive questions answered identically every time, with a link they can send on. |
| **Presenter / builder** | Start in one command, show ingest → retrieve → answer, explain RAG in 2 minutes. |
| **Evaluator** | See grounded answers with visible citations, plus at least one honest miss and one refusal. |

## 5. Scope of the corpus

**AMC:** HDFC Mutual Fund — one AMC, five schemes, all Direct – Growth plans.

| # | Scheme | Category | URL |
|---|--------|----------|-----|
| 1 | HDFC Large Cap Fund – Direct Growth | Large Cap | <https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth> |
| 2 | HDFC Flexi Cap Fund – Direct Growth | Flexi Cap | <https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth> |
| 3 | HDFC ELSS Tax Saver Fund – Direct Plan Growth | ELSS | <https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth> |
| 4 | HDFC Small Cap Fund – Direct Growth | Small Cap | <https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth> |
| 5 | HDFC Balanced Advantage Fund – Direct Growth | Hybrid (Balanced Advantage) | <https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth> |

Machine-readable list: `sources.csv`. Retrieved **2026-09-29**.

## 6. Functional requirements

| ID | Requirement | Priority | Status |
|----|-------------|----------|--------|
| FR-1 | Ingest 5 public pages from one AMC | P0 | Done |
| FR-2 | Chunk with overlap, store metadata (source_url, scheme, section, chunk_id) | P0 | Done |
| FR-3 | Embed with `all-MiniLM-L6-v2` (384-dim, local, no key) | P0 | Done |
| FR-4 | Store vectors in ChromaDB persisted to disk; ingest runs **once** | P0 | Done — re-run skips |
| FR-5 | Chunking strategy decided from the data, justified, and dumped to a readable `.txt` | P0 | Done — `data/chunks.txt` |
| FR-6 | Answer the question from retrieved chunks only, ≤3 sentences | P0 | Done |
| FR-7 | Exactly **one** source link on every factual answer | P0 | Done |
| FR-8 | Refuse "should I buy/sell" and portfolio questions politely, with an educational link | P0 | Done |
| FR-9 | Refuse PII (PAN, Aadhaar, account, OTP, email, phone); store nothing | P0 | Done |
| FR-10 | Make no performance claims; point at the official factsheet | P0 | Done |
| FR-11 | Say "not stated in the sources" rather than hallucinate | P0 | Done |
| FR-12 | UI: welcome line, 3 example questions, "Facts-only. No investment advice." | P1 | Done |
| FR-13 | Show `Last updated from sources:` on every answer | P1 | Done |
| FR-14 | Show corpus status ("indexed N chunks from M pages") | P1 | Done |
| FR-15 | Show retrieved context so grounding is inspectable | P1 | Done |
| FR-16 | One-command local run | P1 | Done |
| FR-17 | Sample Q&A artefact (5–10+ queries with answers and links) | P1 | Done — 12 queries |
| FR-18 | Source list artefact (CSV/MD of the 5 URLs) | P1 | Done — `sources.csv` |

## 7. Non-functional requirements

| ID | Requirement | Status |
|----|-------------|--------|
| NFR-1 | Runs on a student laptop (Windows target), no cloud services | Done |
| NFR-2 | Answer in seconds; first embed is the only slow step and is pre-cached | Done |
| NFR-3 | Demo-scale corpus (5 pages, ~36 chunks) | Done |
| NFR-4 | Secrets via env var only, never committed | Done — `.env` git-ignored |
| NFR-5 | Public sources only; no app back-end, no third-party blogs | Done |
| NFR-6 | No PII accepted, processed, logged or persisted | Done |
| NFR-7 | README with setup, scope, chunking rationale and known limits | Done |
| NFR-8 | Pipeline explainable on a whiteboard in under two minutes | Done |

## 8. Constraints (from the brief, treated as non-negotiable)

| Constraint | How it is met |
|---|---|
| **Public sources only.** No app-back-end screenshots, no third-party blogs. | Loader fetches five public HTML pages and nothing else. |
| **No PII.** Do not accept or store PAN, Aadhaar, account numbers, OTPs, emails, phones. | Regex guard runs **before** the question is embedded or sent to the LLM. Nothing is written to disk. |
| **No performance claims.** Don't compute/compare returns; link the official factsheet. | Performance-claim guard plus a system-prompt rule; the app never performs arithmetic on returns. |
| **Clarity & transparency.** Answers ≤3 sentences; add "Last updated from sources:". | Hard sentence cap in code; freshness stamp rendered under every answer. |
| **One citation per answer.** | The URL is taken from Chroma metadata, never written by the model, so a hallucinated link is structurally impossible. |

## 9. Deliverables to submit

| # | Deliverable | File / command | Status |
|---|-------------|----------------|--------|
| 1 | Working prototype | `streamlit run app/app.py` | Done |
| 2 | Source list (CSV/MD) of the 5 URLs | `sources.csv` | Done |
| 3 | README: setup, scope (AMC + schemes), known limits | `README.md` | Done |
| 4 | Sample Q&A (queries + answers + links) | `SAMPLE_QA.md` — 12 queries | Done |
| 5 | Disclaimer snippet used in the UI | `DISCLAIMER.md`, and `DISCLAIMER` in `app/config.py` | Done |
| 6 | Chunking strategy + inspectable chunk dump | `app/chunking.py` docstring, `data/chunks.txt` | Done |
| 7 | Architecture document | `architecture.md` | Done |
| 8 | This PRD | `PRD.md` | Done |

A hosted link or ≤3-minute demo video is a fallback if hosting is not possible;
this prototype runs locally, which satisfies the brief for a classroom demo.

## 10. Demo success path (60 seconds)

1. App opens; sidebar reads *"Indexed 36 chunks from 5 public HDFC scheme pages"*.
2. Click the example **"What is the expense ratio of HDFC Large Cap Fund Direct Growth?"** →
   grounded answer + one Groww link + top-k context in an expander. *(grounded hit)*
3. Type **"What is the lock-in period for HDFC ELSS Tax Saver Fund?"** →
   *"Not stated in the sources I have."* *(honest miss — proves it is not hallucinating)*
4. Type **"Should I buy HDFC Small Cap Fund for my portfolio?"** →
   refusal + educational link. *(advice guard)*
5. Type **"What are the 5 year returns of HDFC Large Cap Fund versus the category?"** →
   performance refusal, pointed at the factsheet. *(no performance claims)*
6. Type a message containing a PAN → PII refusal. *(no PII)*

## 11. Risks and mitigations

| Risk | Mitigation | Status |
|---|---|---|
| Numbers go stale (NAV, AUM, expense ratio) | Freshness stamp on every answer; documented re-run (`fetch_sources.py` → `ingest.py`) | Mitigated, not solved — no scheduler |
| Hallucinated fee number | Grounding, 3-sentence cap, one citation, "not stated" path, visible retrieved context | Mitigated |
| Cross-scheme contamination (5 near-identical HDFC pages) | Scheme-scoped retrieval + a system-prompt rule to never blend schemes | Fixed — see `architecture.md` §4.3 |
| Retrieval misses the right chunk (e.g. benchmark orphaned in a small chunk) | Chunk-size tuning validated against real queries; adaptive k | Fixed |
| Advice creep | Input guard **and** output guard | Mitigated |
| No API key at demo time | `.env.example`, visible key warning in sidebar, extractive no-key fallback | Mitigated |
| Slow first embed on a cold laptop | Ingest run once before the demo; vector store persisted | Mitigated |
| Third-party site layout changes break the loader | Loader caches to `data/raw/`; failure is loud, not silent | Open — a selector change needs a re-run |

## 12. Open questions

1. **Source of record.** The brief says "AMC/SEBI/AMFI" but the five supplied URLs
   are Groww (a distributor) pages. A production version should ingest
   `hdfcfund.com` factsheets, KIM and SIDs, plus AMFI NAV data. **Open.**
2. **PDF ingestion.** SID/KIM/factsheets are PDFs; `pypdf` is not yet a dependency. **Open.**
3. **The ELSS lock-in period and the capital-gains statement flow are not on these
   five pages.** Both currently return an honest miss. Fixing them means adding the
   scheme's SID and the relevant account-help page to `sources.csv`. **Open.**
4. **How stale may the numbers be before the demo?** A weekly refresh job would
   need hosting. **Open — out of scope for the build hour.**

## 13. Companion document

`architecture.md` — component diagram, data flow, module map, retrieval and guard
design, data model, error paths, and extension points.
