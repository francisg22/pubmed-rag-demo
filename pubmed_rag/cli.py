"""Command-line entry point: python -m pubmed_rag <command>"""
import argparse
import json

from . import agent as agent_mod
from . import ask as ask_mod
from . import compliance as compliance_mod
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
    if (args.corpus or config.CORPUS) == "compliance":
        hits = compliance_mod.search(
            args.query, k=args.k, mode=args.mode, jurisdiction=args.jurisdiction
        )
        if not hits:
            print("No results -- is the compliance corpus ingested?")
            return
        for i, h in enumerate(hits, 1):
            print(f"{i:2}. [{h.score:.4f}] {h.jurisdiction or '?'} | {h.doc_id} -- {h.title}")
            if args.verbose:
                print(f"      {h.text[:300]}...")
        return
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
    if (args.corpus or config.CORPUS) == "compliance":
        print(compliance_mod.ask(args.question, k=args.k, jurisdiction=args.jurisdiction))
        return
    print(ask_mod.ask(args.question, k=args.k, full_text=not args.abstract_only))


def cmd_agent(args) -> None:
    from . import phi

    engine = getattr(args, "engine", "agentic")

    def on_event(ev) -> None:
        if ev["type"] == "tool_call":
            shown = ", ".join(f"{k}={v!r}" for k, v in ev["args"].items())
            print(f"  -> {ev['name']}({shown})")
        elif ev["type"] == "tool_result":
            print(f"     {ev['result']}")

    # Patient-education modes take open-ended input -> screen it for PHI/PII first.
    if engine in ("general", "article"):
        refusal = phi.input_guard(args.question)
        if refusal:
            print(refusal)
            return

    include_general = getattr(args, "include_general", False)
    jurisdiction = getattr(args, "jurisdiction", None)

    if engine == "article":
        src = "compliance + general corpora" if include_general else "compliance corpus"
        print(f"Drafting a patient-information article (grounded in the {src}):\n")
        result = agent_mod.write_article(
            args.question, jurisdiction=jurisdiction, include_general=include_general,
            max_steps=args.max_steps, on_event=on_event,
        )
    elif engine == "general":
        src = "compliance + general corpora" if include_general else "compliance corpus"
        print(f"General assistant (grounded in the {src}):\n")
        question = args.question
        if jurisdiction:
            question += f"\n\n(For any policy/compliance details, use the {jurisdiction} jurisdiction.)"
        result = agent_mod.run_general(
            question, k=args.k, max_steps=args.max_steps, on_event=on_event,
            include_general=include_general,
        )
    else:
        corpus = args.corpus or config.CORPUS
        print(f"Agent working on the '{corpus}' corpus (tool calls shown live):\n")
        result = agent_mod.run_agent(
            args.question, k=args.k, max_steps=args.max_steps, on_event=on_event, corpus=corpus
        )
    print(f"\n--- answer ({result['steps']} step(s)) ---\n")
    print(result["answer"])


def cmd_ingest(args) -> None:
    import hashlib
    import sys
    from collections import Counter

    from . import chunk as chunk_mod
    from . import phi
    from .sources import localfiles

    corpus = getattr(args, "corpus", None) or "compliance"
    prof = config.corpus_profile(corpus)
    table = prof["table"]
    root = args.path or prof.get("dir") or config.LOCAL_DOCS_DIR
    enabled = config.ALLOW_EMBEDDING and not args.dry_run
    print(f"Corpus: {corpus}  (table '{table}')")
    print(f"Source: {root}")
    if enabled:
        print("Embedding gate: ON  -- will embed via OpenAI and store in pgvector.")
        db.init_local_docs(table)
        sig = config.embedding_signature()
        existing = db.local_doc_hashes(table)  # incremental: skip files whose content is unchanged
    else:
        why = "--dry-run" if args.dry_run else "ALLOW_EMBEDDING not set"
        print(f"Embedding gate: OFF ({why}) -- analysis only. NO OpenAI calls, nothing stored.")
        existing = {}

    n_docs = n_chunks = n_tokens = n_unchanged = 0
    phi_totals: Counter = Counter()
    skips: Counter = Counter()            # by reason, never filenames (compliance-safe)
    by_juris: dict[str, list] = {}        # jurisdiction -> [docs, chunks, tokens]

    def on_skip(_rel, reason):
        skips["extract error" if reason.startswith("extract error") else reason] += 1

    files = list(localfiles.iter_files(root, on_skip))  # discovery only; counts unsupported skips
    total = len(files)

    def progress(i):
        tail = f", {n_unchanged} unchanged" if n_unchanged else ""
        sys.stdout.write(f"\r  [{i}/{total}] {n_chunks} chunks{tail}        ")
        sys.stdout.flush()

    for i, (rel, path, category) in enumerate(files, 1):
        sha = None
        if enabled:
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            if existing.get(rel) == sha:        # unchanged since last ingest -> skip (no re-embed)
                n_unchanged += 1
                progress(i)
                continue
        doc = localfiles.extract_document(rel, path, category, on_skip)
        if doc is None:
            progress(i)
            continue
        chunks = chunk_mod.chunk_text(doc.text)
        if not chunks:
            skips["no chunks after splitting"] += 1
            progress(i)
            continue
        toks = sum(c.n_tokens for c in chunks)
        n_docs += 1
        n_chunks += len(chunks)
        n_tokens += toks
        agg = by_juris.setdefault(category or "(root)", [0, 0, 0])
        agg[0] += 1
        agg[1] += len(chunks)
        agg[2] += toks
        phi_totals.update(phi.screen(doc.text))
        if enabled:
            doc.metadata["sha256"] = sha
            if rel in existing:               # changed file -> drop old chunks before re-embedding
                db.delete_local_doc(rel, table)
            vectors = embed.embed_texts([c.text for c in chunks])
            rows = [
                (
                    f"{doc.doc_id}#{c.ordinal}", doc.doc_id, c.ordinal, doc.title,
                    doc.category, doc.source_path, c.text, json.dumps(doc.metadata),
                    sig, db.vec_literal(v),
                )
                for c, v in zip(chunks, vectors)
            ]
            db.upsert_local_chunks(rows, table)
        progress(i)
    if total:
        sys.stdout.write("\n")

    facet = "jurisdiction" if corpus == "compliance" else "category (top-level folder)"
    print(f"\nBy {facet} (docs / chunks / tokens):")
    for juris in sorted(by_juris):
        d, c, t = by_juris[juris]
        print(f"  {juris:12} {d:5} docs  {c:7} chunks  {t:>12,} tokens")
    print(f"\nTotal: {n_docs} documents -> {n_chunks} chunks, ~{n_tokens:,} tokens")
    if n_unchanged:
        print(f"Unchanged (already embedded, skipped): {n_unchanged}")
    if skips:
        print("Skipped (by reason, no filenames): "
              + ", ".join(f"{r} x{n}" for r, n in skips.most_common()))
    print(f"Estimated one-time embedding cost (text-embedding-3-small @ $0.02/1M): "
          f"${n_tokens / 1_000_000 * 0.02:.4f}")
    if phi_totals:
        flags = ", ".join(f"{k}={v}" for k, v in sorted(phi_totals.items()))
        print(f"PHI screen: POSSIBLE identifiers found -> {flags}. Review before enabling.")
    else:
        print("PHI screen: no obvious structured identifiers found "
              "(not a guarantee -- does not catch free-text names).")
    if enabled:
        print(f"\nStored {n_chunks} chunks in '{table}'.")
    else:
        print("\nGate OFF: nothing was embedded, sent to OpenAI, or stored. "
              "Set ALLOW_EMBEDDING=1 (after review) to embed + store.")


def cmd_inventory(args) -> None:
    """Extensions + counts only -- no filenames or content, no API. Safe to run
    on a sensitive corpus to see what's there before any extraction."""
    from collections import Counter
    from pathlib import Path

    from .sources import localfiles

    root = Path(args.path or config.LOCAL_DOCS_DIR)
    if not root.exists():
        raise SystemExit(f"Not found: {root}")
    exts: Counter = Counter()
    by_juris: Counter = Counter()
    parseable = 0
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        ext = p.suffix.lower() or "(no-ext)"
        exts[ext] += 1
        rel = p.relative_to(root)
        by_juris[rel.parts[0] if len(rel.parts) > 1 else "(root)"] += 1
        if p.suffix.lower() in localfiles.SUPPORTED:
            parseable += 1
    print(f"Inventory of {root}  (extensions + counts only -- no filenames/content)\n")
    print("Per-jurisdiction file counts:")
    for juris, n in sorted(by_juris.items()):
        print(f"  {juris}: {n}")
    print("\nExtension histogram  (* = parsed as text):")
    for ext, n in exts.most_common():
        print(f"  {'*' if ext in localfiles.SUPPORTED else ' '} {ext:10} {n}")
    total = sum(exts.values())
    print(f"\n{total} files total; {parseable} parseable as text "
          f"({total - parseable} skipped as media/other).")


def cmd_stats(args) -> None:
    with db.connect() as conn:
        total = conn.execute("SELECT count(*) FROM articles").fetchone()[0]
        by_model = conn.execute(
            "SELECT embed_model, count(*), min(pub_year), max(pub_year) "
            "FROM articles GROUP BY embed_model"
        ).fetchall()
        print(f"pubmed: {total} articles")
        for model, n, y0, y1 in by_model:
            print(f"  {n:6} embedded with {model} (years {y0}-{y1})")

        # Local-files corpora (compliance, general): one chunk table each.
        for corpus in ("compliance", "general"):
            table = config.corpus_profile(corpus)["table"]
            if conn.execute("SELECT to_regclass(%s)", (table,)).fetchone()[0] is None:
                print(f"{corpus}: not ingested (table '{table}' absent)")
                continue
            n_chunks = conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            n_docs = conn.execute(f"SELECT count(DISTINCT doc_id) FROM {table}").fetchone()[0]
            models = [r[0] for r in conn.execute(f"SELECT DISTINCT embed_model FROM {table}")]
            print(f"{corpus}: {n_docs} docs -> {n_chunks} chunks "
                  f"[{', '.join(models) or 'no rows'}]")


def cmd_interactions(args) -> None:
    """Read-only analytics over the logged Q&A interactions (no content is sent
    anywhere). PHI-flagged questions were stored with their text withheld."""
    table = config.INTERACTIONS_TABLE
    with db.connect() as conn:
        if conn.execute("SELECT to_regclass(%s)", (table,)).fetchone()[0] is None:
            print(f"No interactions yet (table '{table}' absent).")
            return
        total = conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        flagged = conn.execute(f"SELECT count(*) FROM {table} WHERE question_flagged").fetchone()[0]
        up = conn.execute(f"SELECT count(*) FROM {table} WHERE feedback = 1").fetchone()[0]
        down = conn.execute(f"SELECT count(*) FROM {table} WHERE feedback = -1").fetchone()[0]
        clicks = conn.execute(
            f"SELECT count(*) FROM {table} WHERE clicked_suggestion IS NOT NULL"
        ).fetchone()[0]
        print(f"interactions: {total} total ({flagged} PHI-flagged, text withheld)")
        print(f"  feedback: up {up} / down {down}    follow-up clicks: {clicks}")
        print("  by corpus / mode:")
        for corpus, mode, n in conn.execute(
            f"SELECT corpus, mode, count(*) FROM {table} "
            f"GROUP BY corpus, mode ORDER BY count(*) DESC"
        ).fetchall():
            print(f"    {corpus or '?':12} {mode or '?':20} {n}")
        print(f"  recent questions (last {args.limit}):")
        for q, fb, nsrc in conn.execute(
            f"SELECT question, feedback, n_sources FROM {table} "
            f"WHERE question IS NOT NULL ORDER BY created_at DESC LIMIT %s",
            (args.limit,),
        ).fetchall():
            mark = {1: "up  ", -1: "down"}.get(fb, "    ")
            print(f"    {mark} [{nsrc} src] {(q or '')[:80]}")


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

    sp = sub.add_parser("search", help="retrieve from a corpus (pubmed or compliance)")
    sp.add_argument("query")
    sp.add_argument("--k", type=int, default=5)
    sp.add_argument("--mode", choices=["vector", "keyword", "hybrid"], default="hybrid")
    sp.add_argument("--corpus", choices=["pubmed", "compliance"], help="default: $CORPUS or pubmed")
    sp.add_argument("--jurisdiction", choices=["US", "UK", "Australia"], help="compliance only")
    sp.add_argument("--verbose", "-v", action="store_true")
    sp.set_defaults(fn=cmd_search)

    ap = sub.add_parser("ask", help="grounded Q&A (pubmed: cited quotes; compliance: jurisdiction-scoped)")
    ap.add_argument("question")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--corpus", choices=["pubmed", "compliance"], help="default: $CORPUS or pubmed")
    ap.add_argument("--jurisdiction", choices=["US", "UK", "Australia"], help="compliance only")
    ap.add_argument(
        "--abstract-only",
        action="store_true",
        help="pubmed only: skip open-access full-text fetch",
    )
    ap.set_defaults(fn=cmd_ask)

    gp = sub.add_parser("agent", help="agentic Q&A -- the model drives retrieval via tools")
    gp.add_argument("question", help="a question, or (with --engine article) an article topic")
    gp.add_argument("--k", type=int, default=6, help="results per search (default 6)")
    gp.add_argument("--max-steps", type=int, default=6, help="max tool-calling rounds (default 6)")
    gp.add_argument("--corpus", choices=["pubmed", "compliance"], help="default: $CORPUS or pubmed")
    gp.add_argument(
        "--engine",
        choices=["agentic", "general", "article"],
        default="agentic",
        help="agentic: single-corpus tool loop (default). general: grounded patient-education "
             "Q&A over compliance (+general). article: draft a patient-information article.",
    )
    gp.add_argument(
        "--include-general",
        action="store_true",
        help="general/article engines: also ground in the optional general-knowledge corpus "
             "(only takes effect once it's ingested)",
    )
    gp.add_argument("--jurisdiction", choices=["US", "UK", "Australia"],
                    help="general/article engines: prefer this jurisdiction for policy details")
    gp.set_defaults(fn=cmd_agent)

    ip = sub.add_parser(
        "ingest",
        help="local files -> extract, chunk, PHI-screen (embeds ONLY if ALLOW_EMBEDDING=1)",
    )
    ip.add_argument(
        "--corpus",
        choices=["compliance", "general"],
        default="compliance",
        help="which local-files corpus to ingest into (default compliance)",
    )
    ip.add_argument("--path", help="corpus directory (default: the corpus's configured dir)")
    ip.add_argument(
        "--dry-run",
        action="store_true",
        help="force analysis only, even if ALLOW_EMBEDDING is set",
    )
    ip.set_defaults(fn=cmd_ingest)

    vp = sub.add_parser(
        "inventory",
        help="list file extensions + counts in the corpus dir (no content, no API)",
    )
    vp.add_argument("--path", help=f"corpus directory (default {config.LOCAL_DOCS_DIR})")
    vp.set_defaults(fn=cmd_inventory)

    sub.add_parser("stats", help="corpus summary").set_defaults(fn=cmd_stats)

    xp = sub.add_parser("interactions", help="analytics on logged Q&A interactions")
    xp.add_argument("--limit", type=int, default=15, help="recent questions to show")
    xp.set_defaults(fn=cmd_interactions)

    args = p.parse_args()
    args.fn(args)
