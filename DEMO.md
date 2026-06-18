# Demo script

A live walkthrough of the PubMed RAG demo. The beats are modular — run all of
them (~12 min) or cherry-pick. The **agentic mode (§5)** and the **GUI (§7)** are
the showpieces; lead with those for a non-technical audience.

Current state assumed below: the corpus holds **~427 articles** across ~7+
clinical topics (metformin/CKD, statins, atrial fibrillation/anticoagulation,
HFpEF, SGLT2 inhibitors, hypertension, COPD), embedded with
`openai:text-embedding-3-small`, and a real `OPENAI_API_KEY` is in `.env` so
`ask`/`agent` call the model for real.

> Without a key: `ask` prints the assembled grounded prompt instead of an answer
> (still instructive), and `agent` can't run (it needs the model to drive). The
> ingestion/retrieval/guardrail beats still work via the offline `hashbow` embedder.

## 0. Before the demo (off-screen)

```bash
podman start pubmed-rag-db        # data persists in its volume
source .venv/bin/activate
python -m pubmed_rag stats        # expect ~427 articles, openai:text-embedding-3-small:1536
```

If the corpus is empty (fresh machine), load a few topics first (~1 min):

```bash
python -m pubmed_rag init-db
python -m pubmed_rag load --query "metformin chronic kidney disease" --max 50
python -m pubmed_rag load --query "SGLT2 inhibitors type 2 diabetes cardiovascular outcomes" --max 60
```

## 1. Ingestion story (30s)

```bash
python -m pubmed_rag load --query "atrial fibrillation anticoagulation" --max 40
```

Narrate as it prints: **esearch** turns the query into PMIDs → **efetch** pulls
titles/abstracts (rate-limited politely) → each title+abstract is embedded →
upserted into one Postgres table. Point out the embedding signature — every row
records which model made its vector.

## 2. Retrieval — why hybrid is the default (90s)

```bash
python -m pubmed_rag search "blood thinners for an irregular heartbeat" --mode keyword   # 0 hits
python -m pubmed_rag search "blood thinners for an irregular heartbeat" --mode hybrid    # spot-on
```

Talking point: none of "blood thinners", "irregular", or "heartbeat" appear
verbatim in the literature, so keyword/BM25 returns nothing — but the embedding
arm matches on *meaning* and surfaces the atrial-fibrillation anticoagulation
papers. Pure vector search has the opposite weakness (exact tokens — drug names,
gene symbols, codes), so hybrid fuses both with Reciprocal Rank Fusion.
(Pre-captured in `demo_outputs/09_semantic_vs_keyword.txt`.)

## 3. Grounded Q&A with cited quotes + full text (2 min)

```bash
python -m pubmed_rag ask "How do SGLT2 inhibitors affect heart failure outcomes?"
```

Walk the safety contract the answer obeys:

1. answers **only** from retrieved sources (grounding)
2. every claim carries a `[PMID]` **and a verbatim quote** (provenance)
3. says so if the sources don't support it (abstention)
4. ends "Draft for clinician review" (human in the loop)

Then show that it quotes the **article body**, not just the abstract:

```bash
python -m pubmed_rag ask "How do SGLT2 inhibitors affect heart failure outcomes?" --abstract-only
```

The default pulls open-access **full text** from PubMed Central for the hits that
have it (the rest fall back to the abstract), so the answer can quote
body-only sentences the abstract-only version can't. Side-by-side capture:
`demo_outputs/13_full_text_vs_abstract.txt`.

## 4. Abstention — it won't make things up (45s)

```bash
python -m pubmed_rag ask "What is the first-line antibiotic regimen for community-acquired pneumonia?"
```

The corpus has no CAP articles; retrieval surfaces loosely-related respiratory
papers, but the model **declines** rather than answering from them or from its
own memory: *"the abstracts do not contain enough information…"* This is the
trust story for a clinical audience. (Capture: `demo_outputs/08_abstention_pneumonia.txt`.)

## 5. Agentic mode — the model drives retrieval (2–3 min) ⭐

```bash
python -m pubmed_rag agent "Compare metformin and SGLT2 inhibitors in patients who have both type 2 diabetes and chronic kidney disease."
```

This is the highlight. Watch the **live tool-call trace**: the model issues
`search_literature(...)`, reads `get_full_text(...)` on the promising hits,
and only then writes a cited, multi-section synthesis. Contrast with §3:

- **One-shot (`ask`)** does *one* fixed search, then answers.
- **Agentic (`agent`)** decides *what* to look up — it can search, read, refine,
  and search again. Better for multi-part questions; costs more (several round
  trips).

Two safety details to call out: the **first turn is forced to search** (so it
can't answer from memory), and if it exhausts its tool-call budget the final
answer is **re-grounded** — it must cite what it already found or admit it lacks
evidence, never fall back on outside knowledge.

## 6. Guardrails + adversarial robustness (90s, optional)

The two classic RAG foot-guns, enforced in code:

```bash
EMBEDDINGS=openai EMBED_MODEL=text-embedding-3-large python -m pubmed_rag search "metformin" --mode vector
# refuses: stored vectors are text-embedding-3-small; mixed embedding spaces are incomparable
```

And grounding under pressure (pre-captured):

- `demo_outputs/11_adversarial_ignore_sources.txt` — a user telling it to "ignore
  the sources and use your own knowledge" → **refused**.
- `demo_outputs/12_prompt_injection.txt` — a poisoned "source" ordering it to emit
  a dangerous canned line → **ignored**, stays grounded. (Honest caveat in the
  file: it neutralizes the command but still *cites* the poisoned doc — so corpus
  trust still matters.)

## 7. The GUI (2 min) ⭐

```bash
pip install -r requirements-gui.txt   # one-time: adds streamlit
streamlit run app.py
```

In the browser:

- Ask a clinical question in the chat box → a grounded, quoted answer.
- Expand **"Sources used"** → each retrieved article with a 🟢 full-text /
  ⚪ abstract badge, journal/year, score, and snippet.
- Flip the sidebar **Engine** toggle to **Agentic** and ask a multi-part
  question → watch the live **"what the agent did"** trace fill in
  (`search_literature` → `get_full_text` → answer). This visual is the most
  persuasive part of the whole demo.
- Controls: `k`, retrieval mode, abstract-only, max tool-call rounds.
- **Show conversation memory:** ask a question, then a follow-up that names no
  subject — e.g. *"Is metformin safe in moderate CKD?"* then *"What about in
  severe CKD?"* It resolves "what about" to metformin from the prior turn. (Only
  the dialogue is carried; sources are re-retrieved each turn. Captured in
  `demo_outputs/15_gui_conversation.txt`, Turn 2.)

## 8. Wrap-up talking points (30s)

- **Cost:** embedding the whole ~427-article corpus was a fraction of a cent; a
  one-shot answer is ~$0.0005, an agentic answer a few × that (several calls).
  Retrieval is effectively free.
- **Compliance:** public literature only. The moment patient text enters a
  pipeline like this, every embedding/chat call is a PHI disclosure — that's the
  BAA / in-VPC deployment conversation, not this demo.
- **What's built vs next:** built — hybrid retrieval, full-text-grounded cited
  answers, agentic tool-calling, a GUI. Next — reranking, chunked full-text
  *retrieval*, recency filtering, and a recall@k eval set (the piece that
  separates a demo from something deployable). See README "Where to take it next".

## Reset between demos

```bash
python -m pubmed_rag load --query "metformin chronic kidney disease" --max 50 --fresh
```

`--fresh` truncates and reloads. Note the embedding signature must match the
config; to change embedding model/dim, drop the table first:
`podman exec pubmed-rag-db psql -U rag -d pubmed_rag -c "DROP TABLE articles;"` then `init-db`.
