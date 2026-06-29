"""Retrieval + grounded Q&A over the local compliance-documents corpus.

Mirrors the PubMed path (search.py / ask.py) but targets the chunk table
(config.LOCAL_DOCS_TABLE), supports a jurisdiction filter (the top-level folder:
US / UK / Australia, stored in the `category` column), and uses the restricted
compliance system prompt. Kept as a separate module so the PubMed flow is
completely untouched.

No network calls happen until a query actually needs a query embedding (vector /
hybrid modes) or the chat answer — i.e. only once OPENAI_API_KEY is set.
"""
from dataclasses import dataclass

from . import config, db
from .ask import history_messages
from .embed import embed_texts
from .search import CANDIDATES, RRF_K, _set_ef_search

TABLE = config.LOCAL_DOCS_TABLE


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    jurisdiction: str | None
    source_path: str | None
    text: str
    score: float


def _rows(rows):
    return [Chunk(*r) for r in rows]


def _assert_ready() -> None:
    """No-op if the corpus isn't ingested yet; otherwise guard against querying a
    corpus embedded with a different model than the current config (the hashbow vs
    openai foot-gun)."""
    current = config.embedding_signature()
    with db.connect() as conn:
        if conn.execute("SELECT to_regclass(%s)", (TABLE,)).fetchone()[0] is None:
            raise SystemExit(
                f"Compliance corpus '{TABLE}' not ingested yet. Run "
                "`ALLOW_EMBEDDING=1 python -m pubmed_rag ingest` (with your key set)."
            )
        stale = {r[0] for r in conn.execute(f"SELECT DISTINCT embed_model FROM {TABLE}")} - {current}
    if stale:
        raise SystemExit(
            f"Compliance corpus holds vectors from {sorted(stale)} but the current "
            f"config produces '{current}'. Re-ingest with matching EMBED settings."
        )


def search(query: str, k: int = 6, mode: str = "hybrid", jurisdiction: str | None = None) -> list[Chunk]:
    _assert_ready()
    if mode == "keyword":
        return _keyword(query, k, jurisdiction)
    if mode == "vector":
        return _vector(query, k, jurisdiction)
    return _hybrid(query, k, jurisdiction)


def _vector(query, k, jurisdiction):
    qvec = db.vec_literal(embed_texts([query])[0])
    where = "WHERE category = %s" if jurisdiction else ""
    sql = f"""
        SELECT chunk_id, doc_id, title, category, source_path, text,
               (1 - (embedding <=> %s::vector))::float8 AS score
        FROM {TABLE}
        {where}
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """
    params = [qvec] + ([jurisdiction] if jurisdiction else []) + [qvec, k]
    with db.connect() as conn:
        _set_ef_search(conn, max(k, CANDIDATES) if jurisdiction else k)
        rows = conn.execute(sql, tuple(params)).fetchall()
    return _rows(rows)


def _keyword(query, k, jurisdiction):
    extra = "AND category = %s" if jurisdiction else ""
    sql = f"""
        SELECT chunk_id, doc_id, title, category, source_path, text,
               ts_rank_cd(fts, q)::float8 AS score
        FROM {TABLE}, websearch_to_tsquery('english', %s) q
        WHERE fts @@ q {extra}
        ORDER BY score DESC
        LIMIT %s
    """
    params = [query] + ([jurisdiction] if jurisdiction else []) + [k]
    with db.connect() as conn:
        rows = conn.execute(sql, tuple(params)).fetchall()
    return _rows(rows)


def _hybrid(query, k, jurisdiction):
    qvec = db.vec_literal(embed_texts([query])[0])
    vec_where = "WHERE category = %s" if jurisdiction else ""
    kw_extra = "AND category = %s" if jurisdiction else ""
    sql = f"""
        WITH vec AS (
            SELECT chunk_id, row_number() OVER (ORDER BY embedding <=> %s::vector) AS r
            FROM {TABLE}
            {vec_where}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        ),
        kw AS (
            SELECT chunk_id, row_number() OVER (ORDER BY ts_rank_cd(fts, q) DESC) AS r
            FROM {TABLE}, websearch_to_tsquery('english', %s) q
            WHERE fts @@ q {kw_extra}
            ORDER BY r
            LIMIT %s
        ),
        fused AS (
            SELECT chunk_id,
                   coalesce(1.0 / (%s + vec.r), 0) + coalesce(1.0 / (%s + kw.r), 0) AS score
            FROM vec FULL OUTER JOIN kw USING (chunk_id)
        )
        SELECT a.chunk_id, a.doc_id, a.title, a.category, a.source_path, a.text, fused.score::float8
        FROM fused JOIN {TABLE} a USING (chunk_id)
        ORDER BY fused.score DESC
        LIMIT %s
    """
    p = [qvec] + ([jurisdiction] if jurisdiction else []) + [qvec, CANDIDATES]
    p += [query] + ([jurisdiction] if jurisdiction else []) + [CANDIDATES]
    p += [RRF_K, RRF_K, k]
    with db.connect() as conn:
        _set_ef_search(conn, max(CANDIDATES, k))
        rows = conn.execute(sql, tuple(p)).fetchall()
    return _rows(rows)


def build_context(chunks: list[Chunk]) -> str:
    blocks = []
    for c in chunks:
        src = f"[{c.title or c.doc_id}] (jurisdiction: {c.jurisdiction or 'unknown'}; source file: {c.doc_id})"
        blocks.append(f"{src}\n{c.text}")
    return "\n\n---\n\n".join(blocks)


def _prompt() -> str:
    return config.corpus_profile("compliance").get("system_prompt") or config.COMPLIANCE_SYSTEM_PROMPT


def ask(question: str, k: int = 6, mode: str = "hybrid", jurisdiction: str | None = None, history=None) -> str:
    chunks = search(question, k=k, mode=mode, jurisdiction=jurisdiction)
    if not chunks:
        scope = f" in jurisdiction '{jurisdiction}'" if jurisdiction else ""
        return f"No matching passages in the compliance corpus{scope}."
    user_msg = f"Compliance documents:\n\n{build_context(chunks)}\n\nQuestion: {question}"
    prompt = _prompt()

    if not config.OPENAI_API_KEY:
        return (
            "[no OPENAI_API_KEY set -- showing the grounded prompt that would be sent]\n\n"
            f"SYSTEM:\n{prompt}\nUSER:\n{user_msg}"
        )

    from openai import OpenAI

    messages = [
        {"role": "system", "content": prompt},
        *history_messages(history),
        {"role": "user", "content": user_msg},
    ]
    resp = OpenAI().chat.completions.create(model=config.CHAT_MODEL, messages=messages)
    return resp.choices[0].message.content


# --- Agentic tool surface (used by agent.py when corpus="compliance") ---
SNIPPET_CHARS = 320


def tool_search(query, k=6, jurisdiction=None):
    try:
        chunks = search(query, k=int(k or 6), mode="hybrid", jurisdiction=jurisdiction)
    except SystemExit as e:  # corpus not ingested / signature mismatch -> tell the model
        return {"error": str(e)}
    return [
        {
            "doc_id": c.doc_id,
            "title": c.title,
            "jurisdiction": c.jurisdiction,
            "score": round(c.score, 4),
            "snippet": (c.text or "")[:SNIPPET_CHARS],
        }
        for c in chunks
    ]


def tool_get_document(doc_id):
    text = db.local_doc_text(str(doc_id))
    if not text:
        return {"doc_id": doc_id, "error": "no such document"}
    return {"doc_id": doc_id, "text": text[:16000]}


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": (
                "Search the internal compliance corpus (US / UK / Australia policies and "
                "procedures). Optionally restrict to one jurisdiction. Returns matching "
                "passages with their source doc_id and jurisdiction. Call again with refined "
                "terms for multi-part questions; if results don't address the question, say so."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "k": {"type": "integer", "description": "number of passages (default 6)"},
                    "jurisdiction": {"type": "string", "enum": ["US", "UK", "Australia"]},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_document",
            "description": (
                "Fetch the full text of one compliance document by its doc_id (from a "
                "search result) to read or quote it precisely."
            ),
            "parameters": {
                "type": "object",
                "properties": {"doc_id": {"type": "string"}},
                "required": ["doc_id"],
            },
        },
    },
]

IMPLS = {"search_documents": tool_search, "get_document": tool_get_document}
