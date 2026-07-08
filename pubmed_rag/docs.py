"""Table-parameterized retrieval over a local-files chunk corpus.

This is the shared engine behind every local-files corpus (compliance, general,
...). It mirrors the PubMed path (search.py / ask.py) but targets a chunk table
whose name is passed in, with an optional `category`-column filter (compliance
uses it for jurisdiction; general leaves it unset). The corpus-specific modules
(compliance.py, general.py) are thin bindings over this file -- they fix the
table + presentation and keep their own public API.

No network calls happen until a query actually needs a query embedding (vector /
hybrid modes) -- i.e. only once OPENAI_API_KEY is set.
"""
from dataclasses import dataclass

from . import config, db
from .embed import embed_texts
from .search import CANDIDATES, RRF_K, _set_ef_search


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    category: str | None      # jurisdiction for compliance; topic/folder for general
    source_path: str | None
    text: str
    score: float

    @property
    def jurisdiction(self) -> str | None:
        """Back-compat alias: for the compliance corpus `category` IS the jurisdiction."""
        return self.category


def _rows(rows) -> list[Chunk]:
    return [Chunk(*r) for r in rows]


def assert_ready(table: str, label: str = "corpus") -> None:
    """No-op-free guard: raise SystemExit if the corpus isn't ingested, or if it
    holds vectors from a different embedding model than the current config (the
    hashbow vs openai foot-gun)."""
    current = config.embedding_signature()
    with db.connect() as conn:
        if conn.execute("SELECT to_regclass(%s)", (table,)).fetchone()[0] is None:
            raise SystemExit(
                f"{label} corpus '{table}' not ingested yet. Run "
                "`ALLOW_EMBEDDING=1 python -m pubmed_rag ingest "
                f"--corpus ...` (with your key set)."
            )
        stale = {r[0] for r in conn.execute(f"SELECT DISTINCT embed_model FROM {table}")} - {current}
    if stale:
        raise SystemExit(
            f"{label} corpus holds vectors from {sorted(stale)} but the current "
            f"config produces '{current}'. Re-ingest with matching EMBED settings."
        )


def is_ingested(table: str) -> bool:
    """True if the corpus table exists and holds at least one chunk (no exceptions)."""
    try:
        with db.connect() as conn:
            if conn.execute("SELECT to_regclass(%s)", (table,)).fetchone()[0] is None:
                return False
            return conn.execute(f"SELECT EXISTS (SELECT 1 FROM {table})").fetchone()[0]
    except Exception:
        return False


def search(table: str, query: str, k: int = 6, mode: str = "hybrid",
           category: str | None = None, label: str = "corpus") -> list[Chunk]:
    assert_ready(table, label)
    if mode == "keyword":
        return _keyword(table, query, k, category)
    if mode == "vector":
        return _vector(table, query, k, category)
    return _hybrid(table, query, k, category)


def _vector(table, query, k, category):
    qvec = db.vec_literal(embed_texts([query])[0])
    where = "WHERE category = %s" if category else ""
    sql = f"""
        SELECT chunk_id, doc_id, title, category, source_path, text,
               (1 - (embedding <=> %s::vector))::float8 AS score
        FROM {table}
        {where}
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """
    params = [qvec] + ([category] if category else []) + [qvec, k]
    with db.connect() as conn:
        _set_ef_search(conn, max(k, CANDIDATES) if category else k)
        rows = conn.execute(sql, tuple(params)).fetchall()
    return _rows(rows)


def _keyword(table, query, k, category):
    extra = "AND category = %s" if category else ""
    sql = f"""
        SELECT chunk_id, doc_id, title, category, source_path, text,
               ts_rank_cd(fts, q)::float8 AS score
        FROM {table}, websearch_to_tsquery('english', %s) q
        WHERE fts @@ q {extra}
        ORDER BY score DESC
        LIMIT %s
    """
    params = [query] + ([category] if category else []) + [k]
    with db.connect() as conn:
        rows = conn.execute(sql, tuple(params)).fetchall()
    return _rows(rows)


def _hybrid(table, query, k, category):
    qvec = db.vec_literal(embed_texts([query])[0])
    vec_where = "WHERE category = %s" if category else ""
    kw_extra = "AND category = %s" if category else ""
    sql = f"""
        WITH vec AS (
            SELECT chunk_id, row_number() OVER (ORDER BY embedding <=> %s::vector) AS r
            FROM {table}
            {vec_where}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        ),
        kw AS (
            SELECT chunk_id, row_number() OVER (ORDER BY ts_rank_cd(fts, q) DESC) AS r
            FROM {table}, websearch_to_tsquery('english', %s) q
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
        FROM fused JOIN {table} a USING (chunk_id)
        ORDER BY fused.score DESC
        LIMIT %s
    """
    p = [qvec] + ([category] if category else []) + [qvec, CANDIDATES]
    p += [query] + ([category] if category else []) + [CANDIDATES]
    p += [RRF_K, RRF_K, k]
    with db.connect() as conn:
        _set_ef_search(conn, max(CANDIDATES, k))
        rows = conn.execute(sql, tuple(p)).fetchall()
    return _rows(rows)
