"""Command-line entry point: python -m pubmed_rag <command>"""
import argparse
import json

from . import agent as agent_mod
from . import ask as ask_mod
from . import config, db, embed, fetch
from . import search as search_mod


def cmd_init_db(args) -> None:
    db.init_db()
    print(
        f"Schema ready: articles table with vector({config.EMBED_DIM}), "
        f"HNSW + GIN indexes, at {config.DATABASE_URL}"
    )


def cmd_load(args) -> None:
    # Fail fast on a dimension mismatch -- BEFORE the destructive --fresh
    # truncate and before spending fetch/embedding effort on rows that the
    # vector(N) column would reject at upsert time.
    db.assert_dim_matches()
    sig = config.embedding_signature()
    print(f"Embedding provider: {sig}")
    if config.EMBEDDINGS != "openai":
        print("  note: offline stand-in embeddings (set OPENAI_API_KEY for the real thing)")
    if args.fresh:
        with db.connect() as conn:
            conn.execute("TRUNCATE articles")
        print("Cleared existing rows (--fresh).")
    others = db.stored_signatures() - {sig}
    if others:
        print(
            f"  WARNING: database already holds vectors from {sorted(others)}; "
            "search refuses mixed corpora. Consider --fresh."
        )
    pmids = fetch.search_pmids(args.query, args.max)
    print(f"PubMed: {len(pmids)} PMIDs for {args.query!r}")
    articles = fetch.fetch_articles(pmids)
    print(f"{len(articles)} of them have abstracts; embedding ...")
    vectors = embed.embed_texts([f"{a.title}\n\n{a.abstract}" for a in articles])
    rows = [
        (a.pmid, a.title, a.abstract, a.journal, a.pub_year, sig, db.vec_literal(v))
        for a, v in zip(articles, vectors)
    ]
    db.upsert_articles(rows)
    print(f"Upserted {len(rows)} articles.")


def cmd_search(args) -> None:
    hits = search_mod.search(args.query, k=args.k, mode=args.mode)
    if not hits:
        print("No results -- is the corpus loaded? (python -m pubmed_rag load --query ...)")
        return
    for i, h in enumerate(hits, 1):
        print(f"{i:2}. [{h.score:.4f}] PMID {h.pmid} ({h.pub_year or '????'}) {h.title}")
        if args.verbose:
            print(f"      {h.journal or ''}")
            print(f"      {h.abstract[:300]}...")


def cmd_ask(args) -> None:
    print(ask_mod.ask(args.question, k=args.k, full_text=not args.abstract_only))


def cmd_agent(args) -> None:
    def on_event(ev) -> None:
        if ev["type"] == "tool_call":
            shown = ", ".join(f"{k}={v!r}" for k, v in ev["args"].items())
            print(f"  -> {ev['name']}({shown})")
        elif ev["type"] == "tool_result":
            print(f"     {ev['result']}")

    print("Agent working (tool calls shown live):\n")
    result = agent_mod.run_agent(
        args.question, k=args.k, max_steps=args.max_steps, on_event=on_event
    )
    print(f"\n--- answer ({result['steps']} step(s)) ---\n")
    print(result["answer"])


def cmd_ingest(args) -> None:
    from . import chunk as chunk_mod
    from . import phi
    from .sources import localfiles

    root = args.path or config.LOCAL_DOCS_DIR
    enabled = config.ALLOW_EMBEDDING and not args.dry_run
    print(f"Source: {root}")
    if enabled:
        print("Embedding gate: ON  -- will embed via OpenAI and store in pgvector.")
        db.init_local_docs()
        sig = config.embedding_signature()
    else:
        why = "--dry-run" if args.dry_run else "ALLOW_EMBEDDING not set"
        print(f"Embedding gate: OFF ({why}) -- analysis only. NO OpenAI calls, nothing stored.\n")

    n_docs = n_chunks = n_tokens = n_skip = 0
    phi_totals: dict[str, int] = {}

    def on_skip(rel, reason):
        nonlocal n_skip
        n_skip += 1
        if n_skip <= 15:
            print(f"  skip: {rel} -- {reason}")

    for doc in localfiles.iter_documents(root, on_skip=on_skip):
        chunks = chunk_mod.chunk_text(doc.text)
        if not chunks:
            on_skip(doc.doc_id, "no chunks after splitting")
            continue
        n_docs += 1
        n_chunks += len(chunks)
        n_tokens += sum(c.n_tokens for c in chunks)
        for name, cnt in phi.screen(doc.text).items():
            phi_totals[name] = phi_totals.get(name, 0) + cnt
        if enabled:
            vectors = embed.embed_texts([c.text for c in chunks])
            rows = [
                (
                    f"{doc.doc_id}#{c.ordinal}", doc.doc_id, c.ordinal, doc.title,
                    doc.category, doc.source_path, c.text, json.dumps(doc.metadata),
                    sig, db.vec_literal(v),
                )
                for c, v in zip(chunks, vectors)
            ]
            db.upsert_local_chunks(rows)

    print(f"\n{n_docs} documents -> {n_chunks} chunks, ~{n_tokens:,} tokens "
          f"({n_skip} files skipped)")
    print(f"Estimated one-time embedding cost (text-embedding-3-small @ $0.02/1M): "
          f"${n_tokens / 1_000_000 * 0.02:.4f}")
    if phi_totals:
        flags = ", ".join(f"{k}={v}" for k, v in sorted(phi_totals.items()))
        print(f"PHI screen: POSSIBLE identifiers found -> {flags}. Review before enabling.")
    else:
        print("PHI screen: no obvious structured identifiers found "
              "(not a guarantee -- does not catch free-text names).")
    if enabled:
        print(f"\nStored {n_chunks} chunks in '{config.LOCAL_DOCS_TABLE}'.")
    else:
        print("\nGate OFF: nothing was embedded, sent to OpenAI, or stored. "
              "After clearing PHI, re-run with ALLOW_EMBEDDING=1 to embed + store.")


def cmd_stats(args) -> None:
    with db.connect() as conn:
        total = conn.execute("SELECT count(*) FROM articles").fetchone()[0]
        by_model = conn.execute(
            "SELECT embed_model, count(*), min(pub_year), max(pub_year) "
            "FROM articles GROUP BY embed_model"
        ).fetchall()
    print(f"{total} articles")
    for model, n, y0, y1 in by_model:
        print(f"  {n:6} embedded with {model} (years {y0}-{y1})")


def main() -> None:
    p = argparse.ArgumentParser(
        prog="pubmed_rag",
        description="PubMed -> OpenAI embeddings -> pgvector RAG starter demo",
    )
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("init-db", help="create extension, table, indexes").set_defaults(
        fn=cmd_init_db
    )

    lp = sub.add_parser("load", help="fetch PubMed abstracts, embed, upsert")
    lp.add_argument(
        "--query", required=True, help="PubMed search, e.g. 'metformin chronic kidney disease'"
    )
    lp.add_argument("--max", type=int, default=50, help="max articles (default 50)")
    lp.add_argument("--fresh", action="store_true", help="truncate the table first")
    lp.set_defaults(fn=cmd_load)

    sp = sub.add_parser("search", help="retrieve from the local corpus")
    sp.add_argument("query")
    sp.add_argument("--k", type=int, default=5)
    sp.add_argument("--mode", choices=["vector", "keyword", "hybrid"], default="hybrid")
    sp.add_argument("--verbose", "-v", action="store_true")
    sp.set_defaults(fn=cmd_search)

    ap = sub.add_parser("ask", help="grounded Q&A with cited quotes (full text where available)")
    ap.add_argument("question")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument(
        "--abstract-only",
        action="store_true",
        help="skip open-access full-text fetch; ground answers on abstracts only",
    )
    ap.set_defaults(fn=cmd_ask)

    gp = sub.add_parser("agent", help="agentic Q&A -- the model drives retrieval via tools")
    gp.add_argument("question")
    gp.add_argument("--k", type=int, default=6, help="results per search (default 6)")
    gp.add_argument("--max-steps", type=int, default=6, help="max tool-calling rounds (default 6)")
    gp.set_defaults(fn=cmd_agent)

    ip = sub.add_parser(
        "ingest",
        help="local files -> extract, chunk, PHI-screen (embeds ONLY if ALLOW_EMBEDDING=1)",
    )
    ip.add_argument("--path", help=f"corpus directory (default {config.LOCAL_DOCS_DIR})")
    ip.add_argument(
        "--dry-run",
        action="store_true",
        help="force analysis only, even if ALLOW_EMBEDDING is set",
    )
    ip.set_defaults(fn=cmd_ingest)

    sub.add_parser("stats", help="corpus summary").set_defaults(fn=cmd_stats)

    args = p.parse_args()
    args.fn(args)
