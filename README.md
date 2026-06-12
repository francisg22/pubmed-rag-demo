# PubMed RAG starter demo

DISCLAIMER: this was completely vibe coded by fable, none of it is my own work

A minimal, end-to-end Retrieval-Augmented Generation (RAG) pipeline matching the
stack from the meeting notes: **PubMed articles → OpenAI embeddings → Postgres +
pgvector → grounded Q&A with PMID citations**. Built as a hands-on learning
exercise — every module is short enough to read in one sitting.

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
                                              │ top-k abstracts
                                              ▼
                              ask.py ──► OpenAI chat ──► cited draft answer
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
python -m pubmed_rag stats
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
| `ask.py` | The grounding/citation/abstention/human-review prompt pattern for clinical RAG |

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
| Chat | OpenAI `gpt-4o-mini` (configurable) | — |
| Fetch | NCBI E-utilities over `requests` | 2.34 |

The whole pipeline is **~600 lines across 9 modules** (`fetch` 112, `cli` 111,
`db`/`search` 109 each, `embed` 67, `ask` 51, `config` 40). It is meant to be
read end-to-end, not depended on.

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

1. **Reranking** — add a cross-encoder pass over the top-50 hybrid candidates.
2. **Chunking** — pull full-text PMC articles and split by section with overlap
   (one row per chunk, plus a `chunk_id`), instead of one row per abstract.
3. **Recency filtering** — `WHERE pub_year >= ...` is already possible; note
   that with HNSW the filter applies *after* the index scan, so selective
   filters need iterative index scans (pgvector ≥ 0.8).
4. **Evaluation** — a gold set of questions with known-relevant PMIDs; measure
   recall@k before/after every change. This is the piece that separates a demo
   from something deployable.
5. **Scale** — pgvector is comfortably sufficient here; revisit dedicated
   engines (e.g., turbopuffer) only at hundreds of millions of chunks.

## References

- text-embedding-3-small: https://developers.openai.com/api/docs/models/text-embedding-3-small
- pgvector: https://github.com/pgvector/pgvector
- turbopuffer (the "if we outgrow pgvector" option): https://turbopuffer.com/
- NCBI E-utilities: https://www.ncbi.nlm.nih.gov/books/NBK25501/
- RAG survey (Gao et al. 2024): https://arxiv.org/abs/2312.10997
