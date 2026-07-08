# PubMed RAG starter demo

DISCLAIMER: This was mostly vibe coded by claude code, little is my own work. 
Mainly created as a demo/proof of concept to show off.

A minimal, end-to-end Retrieval-Augmented Generation (RAG) pipeline matching the
stack from the meeting notes: **PubMed articles → OpenAI embeddings → Postgres +
pgvector → grounded Q&A with PMID citations and verbatim quotes**. Answer it two
ways — a one-shot RAG call or an **agentic** loop where the model drives
retrieval through tools — from the CLI or a **Streamlit GUI**. Built as a
hands-on learning exercise — every module is short enough to read in one sitting.

```
 PubMed E-utilities                  OpenAI /v1/embeddings
   (esearch+efetch)                  (text-embedding-3-small)
        │                                     ▲
        ▼                                     │ title+abstract
    fetch.py ────────────────────────────► embed.py
                                              │ vector(1536)
                                              ▼
                              Postgres 17 + pgvector  (one table)
                              ├── HNSW index   (cosine ANN)
                              └── GIN index    (tsvector keyword)
                                              │
              query ──► embed.py ──► search.py: vector | keyword | hybrid (RRF)
                                              │ top-k articles
                                              ▼
            full text (PMC open-access) where available, else abstract
                                              │
            ask.py  (one-shot)      ·      agent.py  (model-driven, tool-calling)
                                              ▼
                       OpenAI chat ──► cited + quoted draft answer
                                              ▲
                          CLI  (python -m pubmed_rag)  ·  app.py  (Streamlit GUI)
```

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

docker compose up -d          # Postgres 17 + pgvector on localhost:5433
cp .env.example .env          # put your OPENAI_API_KEY here when you have one
```

No compose provider (e.g., Fedora with rootless Podman)? Same container, one command:

```bash
podman run -d --name pubmed-rag-db \
  -e POSTGRES_USER=rag -e POSTGRES_PASSWORD=rag -e POSTGRES_DB=pubmed_rag \
  -p 5433:5432 -v pubmed_rag_pgdata:/var/lib/postgresql/data \
  docker.io/pgvector/pgvector:pg17
```

Then:

```bash

python -m pubmed_rag init-db
python -m pubmed_rag load --query "metformin chronic kidney disease" --max 50
python -m pubmed_rag search "is metformin safe in renal impairment" --mode hybrid
python -m pubmed_rag ask "is metformin safe in moderate CKD?"
python -m pubmed_rag agent "compare metformin and SGLT2 inhibitors in patients with both T2DM and CKD"
python -m pubmed_rag stats
```

Or use the GUI (chat UI with a sources panel and a live agent trace):

```bash
pip install -r requirements-gui.txt   # adds streamlit
streamlit run app.py
```

No `OPENAI_API_KEY`? Everything still runs: the demo falls back to a
deterministic offline embedder (`hashbow`, a hashed bag-of-words) so you can
exercise the full pipeline for free. Offline similarity is lexical, not
semantic — fine for learning the plumbing, not for judging retrieval quality.
`ask` without a key prints the fully assembled grounded prompt instead of
calling the chat model, which is itself instructive.

## What each module teaches

| Module | The lesson |
|---|---|
| `fetch.py` | NCBI E-utilities: `esearch` (query → PMIDs), `efetch` (PMIDs → XML), rate-limit etiquette, messy real-world metadata (labelled abstract sections, irregular dates) |
| `embed.py` | Batched calls to `/v1/embeddings`, the `dimensions` parameter, input-length caps, and why doc & query vectors must come from the same model |
| `db.py` | `vector(1536)` column, HNSW vs GIN indexes, the 2000-dim HNSW ceiling, upserts, and recording the embedding model with every row |
| `search.py` | Cosine ANN (`<=>`), Postgres full-text as the sparse arm, and Reciprocal Rank Fusion for hybrid search |
| `ask.py` | One-shot RAG: the grounding/citation/**quote**/abstention/human-review prompt pattern, with open-access full text pulled into context at answer time |
| `agent.py` | Agentic RAG: OpenAI **function calling** — the model drives retrieval through `search_literature`/`get_full_text` tools in a loop, instead of one fixed retrieval |
| `app.py` | A ~170-line **Streamlit** GUI over both modes: chat, a sources panel, and a live tool-call trace |

## Answer modes & the GUI

There are two ways to answer a question, sharing the same retrieval and safety rules:

- **One-shot (`ask`)** — runs a single hybrid search, pulls the top-k articles
  (open-access full text where available, else abstract) into one prompt, and
  makes one chat call. Simple, cheap, predictable.
- **Agentic (`agent`)** — gives the model two tools (`search_literature`,
  `get_full_text`) and lets it decide what to look up. It can search, read a
  result, refine the query, search again, and only then answer — an OpenAI
  function-calling (ReAct-style) loop. Better for multi-part questions; costs
  more (several round trips). The CLI prints each tool call live. The first turn
  is forced to search so the model can't answer from its own memory, and if it
  runs out of tool-call rounds the final answer is re-grounded (it must cite from
  what it already retrieved or admit it lacks evidence — never fall back on
  outside knowledge).

Both enforce the same contract: answer **only** from retrieved sources, every
claim carries a `[PMID]` **and a verbatim quote**, abstain when unsupported, and
ignore any instructions embedded inside the sources.

The **GUI** (`streamlit run app.py`) wraps both: a chat box, an engine toggle
(agentic vs one-shot), `k`/mode/abstract-only controls, an expandable **Sources
used** panel (with full-text-vs-abstract badges), and — in agentic mode — a live
**"what the agent did"** trace of every search and full-text fetch. That trace is
the most compelling thing to show: it makes model-driven retrieval visible.

### Patient-education modes (compliance assistant)

The compliance assistant adds two grounded patient-education engines, selectable in
the GUI (or via `python -m pubmed_rag agent --engine general|article`):

- **General assistant** — grounded Q&A for basic and *tangential* patient questions.
- **Article writer** — drafts a structured, plain-language patient-information
  article (fixed template: *What it is · Why · Benefits · Risks · What to expect ·
  Recovery · When to get help · Sources*), shown in a tabbed card (Article / Sources
  / How it was built) with a Markdown download.

Both stay **strictly grounded** — no model parametric knowledge — and ground across
the compliance corpus plus an **optional general-knowledge corpus**. That second
corpus is additive and off by default: tick **Include general-knowledge corpus**
(or pass `--include-general`), and it only takes effect once ingested
(`python -m pubmed_rag ingest --corpus general`; see `data/general_docs/README.md`).
Sources are labelled by corpus so official policy is never confused with general
info, and free-text input is PHI-screened before anything is sent.

**Conversation memory:** both modes accept a `history` of prior turns, so
follow-ups work ("…and in *severe* CKD?" remembers the subject). Only the
dialogue text is carried — the retrieved **sources are re-fetched fresh every
turn, never re-sent** — which is what keeps the context small (the short answers
are cheap; the full-text sources are what would blow the window). History is
capped to the last `ask.HISTORY_TURNS` (6) turns.

## Technical details

Concrete specs for anyone evaluating this as a reference implementation. Every
number below is measured from a live run, not estimated.

### Stack

| Layer | Choice | Version (as run) |
|---|---|---|
| Language | Python — no orchestration framework (no LangChain/LlamaIndex) | 3.14 |
| Vector store | PostgreSQL + [pgvector](https://github.com/pgvector/pgvector) | 17.10 / 0.8.2 |
| DB driver | psycopg (v3, binary) — no ORM, raw parameterized SQL | 3.3 |
| Embeddings | OpenAI `text-embedding-3-small` (1536-dim) + offline `hashbow` fallback | openai 2.41 |
| Chat | OpenAI `gpt-4o-mini` — one-shot **and** agentic tool calling (configurable) | — |
| Fetch | NCBI E-utilities over `requests` (abstracts + open-access full text via PMC) | 2.34 |
| GUI (optional) | Streamlit | 1.58 |

The package is **~920 lines across 10 modules** (`agent` 218, `fetch` 162,
`cli` 139, `db`/`search` 109 each, `ask` 74, `embed` 67, `config` 40), plus a
~170-line Streamlit GUI (`app.py`). It is meant to be read end-to-end, not
depended on.

### Corpus (as loaded in `example_output.txt`)

- **47 articles** retrieved from 50 PMIDs for *"metformin chronic kidney
  disease"* — 3 were dropped for having no abstract (book records / ahead-of-print).
- Spanning **2008–2026** across **39 distinct journals**.
- Abstracts average **~1,690 characters (~420 tokens)**, max 5,333 — comfortably
  under both the 8,192-token per-input cap and the per-request token ceiling.
- On disk: **~2.1 MB** total (table 1.1 MB, GIN index 552 kB, HNSW index 416 kB).

### Retrieval internals

- **Distance:** cosine (`<=>`); score reported as `1 - distance`. OpenAI vectors
  are unit-normalized, so cosine and inner product rank identically.
- **Dense index:** HNSW with pgvector defaults (`m=16`, `ef_construction=64`).
  `hnsw.ef_search` is raised **per query, transaction-locally** to the requested
  depth — pgvector otherwise caps an index scan at 40 rows and silently truncates
  deeper `k`.
- **Sparse arm:** a `STORED GENERATED` `tsvector` column (`english` config) with a
  GIN index, queried via `websearch_to_tsquery` (so quoted phrases and `-`
  exclusions work like a search box).
- **Hybrid = Reciprocal Rank Fusion:** each arm contributes `1/(60 + rank)` over
  its top **50 candidates**; the fused sums are re-sorted. RRF needs no score
  calibration between the two arms, which is its whole appeal.
- **Latency:** ~15–18 ms/query at this corpus size — but that's connection +
  Python overhead, not the index. At 47 rows Postgres seq-scans anyway; HNSW only
  earns its keep in the thousands-to-millions range. Worth saying out loud in a
  demo so nobody over-reads a small-N number.

### Two correctness guards worth demoing

Both are the classic RAG foot-guns, enforced in code rather than left to discipline:

1. **Embedding-space guard.** Every row stores a signature
   (`openai:text-embedding-3-small:1536` or `hashbow:1536`); search *refuses* to
   run if the live config doesn't match what's stored. Comparing vectors from two
   models silently returns garbage — this turns that into a hard error.
2. **Dimension guard.** The `vector(N)` width is baked into the schema at table
   creation, so a later `EMBED_DIM` change can't be fixed by re-loading. `load`
   checks this **before** its destructive `--fresh` truncate, so a misconfigured
   re-embed fails fast instead of wiping the corpus and burning API spend first.

## Key considerations (carried over from the design discussion)

- **Vectors are married to their model.** Every row stores an embedding
  signature (`openai:text-embedding-3-small:1536`), and search refuses to run
  if your current config doesn't match what's stored. Switching models means
  re-embedding the corpus (`load --fresh`) — and if `EMBED_DIM` changes too,
  the table itself must be rebuilt first (`DROP TABLE articles;` then
  `init-db`), because the dimension is baked into the `vector(N)` column at
  creation. `load` checks this up front and refuses before touching your data.
- **Hybrid search is the default**, because biomedical queries lean on exact
  tokens (drug names, gene symbols, codes) that dense retrieval alone fumbles.
- **The HNSW index caps at 2000 dimensions** on the plain `vector` type — fine
  for `text-embedding-3-small` (1536), but `-3-large` at 3072 needs `halfvec`
  or the API's `dimensions` truncation.
- **Embedding calls are batched** (100 per request here; the API allows up to
  2048 inputs) — for a large one-time backfill, use the Batch API tier.
- **Raw text is stored next to the vector**, so re-embedding with a different
  model later is a pure re-run, not a re-fetch.
- **Cost intuition:** a PubMed abstract is ~300 tokens; at $0.02/1M tokens,
  embedding 100k abstracts costs on the order of a dollar.

## Compliance note

This demo touches **public literature only** — PubMed abstracts contain no
patient data, so calling the OpenAI API with them is fine. The moment any
patient-derived text enters a pipeline like this, every embedding and chat call
becomes a PHI disclosure: that requires a BAA-covered deployment (e.g., Azure
OpenAI / Bedrock / Vertex inside your VPC) and the rest of the controls from
the design discussion. Keep real records out of this repo.

## Where to take it next

Already added since the first cut: **agentic tool-calling** (`agent.py`),
**open-access full text** pulled into the answer context (`fetch.fetch_full_texts`),
and a **Streamlit GUI** (`app.py`). Still open:

1. **Reranking** — add a cross-encoder pass over the top-50 hybrid candidates.
2. **Chunking** — index full-text PMC articles split by section with overlap
   (one row per chunk, plus a `chunk_id`) so *retrieval* — not just the answer —
   sees the body, instead of one row per abstract.
3. **Recency filtering** — `WHERE pub_year >= ...` is already possible; note
   that with HNSW the filter applies *after* the index scan, so selective
   filters need iterative index scans (pgvector ≥ 0.8).
4. **Evaluation** — a gold set of questions with known-relevant PMIDs; measure
   recall@k before/after every change. This is the piece that separates a demo
   from something deployable.
5. **Streaming + productionizing the GUI** — stream tokens; when it needs real
   users/auth, move to a FastAPI backend (the agent loop is already a clean
   library function) with a React or HTMX front end.
6. **Scale** — pgvector is comfortably sufficient here; revisit dedicated
   engines (e.g., turbopuffer) only at hundreds of millions of chunks.

## References

- text-embedding-3-small: https://developers.openai.com/api/docs/models/text-embedding-3-small
- pgvector: https://github.com/pgvector/pgvector
- turbopuffer (the "if we outgrow pgvector" option): https://turbopuffer.com/
- NCBI E-utilities: https://www.ncbi.nlm.nih.gov/books/NBK25501/
- RAG survey (Gao et al. 2024): https://arxiv.org/abs/2312.10997
