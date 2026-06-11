# 5-minute demo script

A walkthrough for showing the PubMed RAG pipeline live. Total time ~5 minutes,
no API key required (the offline embedder makes every step free).

## 0. Before the demo (one minute, off-screen)

```bash
podman start pubmed-rag-db        # container already exists; data persists in its volume
source .venv/bin/activate
python -m pubmed_rag stats        # sanity check: expect "29 articles ... hashbow:1536"
```

If the corpus is empty (fresh machine), load it first — takes ~30 seconds:

```bash
python -m pubmed_rag init-db
python -m pubmed_rag load --query "metformin chronic kidney disease" --max 50
```

## 1. Show the ingestion story (30s)

```bash
python -m pubmed_rag load --query "metformin chronic kidney disease" --max 50
```

Narrate the pipeline as it prints: **esearch** turns the query into PMIDs →
**efetch** pulls titles/abstracts (rate-limited politely) → each title+abstract
is embedded → upserted into one Postgres table with its vector. Point out the
printed embedding signature: every row records which model made its vector.

## 2. Retrieval — and why hybrid is the default (90s)

```bash
python -m pubmed_rag search "is metformin safe in renal impairment" --mode hybrid
```

Then show the two arms separately with a query that breaks one of them:

```bash
python -m pubmed_rag search "eGFR threshold metformin" --mode keyword   # 0 hits!
python -m pubmed_rag search "eGFR threshold metformin" --mode hybrid    # works
```

Talking point: keyword search ANDs the terms and "threshold" appears in no
abstract — exact-match search is brittle. But pure vector search has the
opposite weakness: biomedical queries lean on exact tokens (drug names, gene
symbols, codes) that semantic similarity fumbles. Hybrid fuses both ranked
lists with Reciprocal Rank Fusion and is robust to either failure.

## 3. Grounded Q&A — the clinical safety pattern (90s)

```bash
python -m pubmed_rag ask "is metformin safe in moderate CKD?"
```

Without an API key this prints the **fully assembled grounded prompt** instead
of calling a chat model — which is the most instructive part anyway. Walk
through the system prompt's four rules:

1. answer ONLY from the provided abstracts (grounding)
2. every claim cites a [PMID] (provenance)
3. "say so" if the abstracts don't answer it (abstention)
4. output is a draft for clinician review (human in the loop)

With `OPENAI_API_KEY` in `.env`, the same command returns the model's cited
answer.

## 4. The guardrails (60s, optional but impressive)

The two classic RAG foot-guns, caught live:

```bash
# Mixed embedding spaces: stored vectors are hashbow, config says OpenAI -> refuses
EMBEDDINGS=openai python -m pubmed_rag search "metformin" --mode vector

# Dimension change: vector(1536) is baked into the schema -> refuses BEFORE
# truncating anything (previously this would have wiped the corpus first)
EMBED_DIM=512 python -m pubmed_rag load --query "metformin" --fresh
```

Talking point: vectors are married to the model that produced them. The demo
records the signature on every row and checks it at query time, because
comparing vectors from two different embedding spaces silently returns garbage.

## 5. Wrap-up talking points (30s)

- **Cost intuition:** ~300 tokens/abstract at $0.02/1M tokens → embedding 100k
  abstracts costs about a dollar. Retrieval is effectively free; the chat call
  is the only real spend.
- **Compliance:** this touches public literature only. The moment patient text
  enters a pipeline like this, every embedding/chat call is a PHI disclosure —
  that's the BAA / in-VPC deployment conversation, not this demo.
- **Next steps** (README "Where to take it next"): reranking, full-text
  chunking, recency filters, and a gold-standard eval set — the piece that
  separates a demo from something deployable.

## Reset between demos

```bash
python -m pubmed_rag load --query "metformin chronic kidney disease" --max 50 --fresh
```

`--fresh` truncates and reloads, so reruns are deterministic. To nuke the
schema too: `podman exec pubmed-rag-db psql -U rag -d pubmed_rag -c "DROP TABLE articles;"`
then `init-db`.
