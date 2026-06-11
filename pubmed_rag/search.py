"""Retrieval: vector, keyword, or hybrid (Reciprocal Rank Fusion of both).

Biomedical queries are dense with exact tokens -- drug names, gene symbols,
ICD codes -- that pure semantic search fumbles, which is why hybrid is the
default. RRF fuses the two ranked lists without needing their scores to be
comparable: each arm contributes 1/(60 + rank) and the sums are re-sorted.
"""
from dataclasses import dataclass

from . import config, db
from .embed import embed_texts

CANDIDATES = 50  # how deep each arm of the hybrid search looks
RRF_K = 60       # standard damping constant for Reciprocal Rank Fusion
EF_SEARCH_MAX = 1000  # pgvector's hnsw.ef_search ceiling


def _set_ef_search(conn, depth: int) -> None:
    """An HNSW index scan returns at most hnsw.ef_search rows (default 40),
    silently -- a LIMIT above that is truncated with no error. Raise it to the
    requested depth for this transaction (set_config instead of SET because
    SET can't take bind parameters; is_local=true resets it on commit)."""
    conn.execute(
        "SELECT set_config('hnsw.ef_search', %s, true)",
        (str(min(max(depth, 40), EF_SEARCH_MAX)),),
    )


@dataclass
class Hit:
    pmid: str
    title: str
    abstract: str
    journal: str | None
    pub_year: int | None
    score: float


def search(query: str, k: int = 5, mode: str = "hybrid") -> list[Hit]:
    db.assert_signature_matches()
    if mode == "vector":
        return _vector(query, k)
    if mode == "keyword":
        return _keyword(query, k)
    return _hybrid(query, k)


def _vector(query: str, k: int) -> list[Hit]:
    qvec = db.vec_literal(embed_texts([query])[0])
    # ORDER BY embedding <=> constant LIMIT k is the exact shape the HNSW
    # index accelerates; score = 1 - cosine distance = cosine similarity.
    sql = """
        SELECT pmid, title, abstract, journal, pub_year,
               (1 - (embedding <=> %s::vector))::float8 AS score
        FROM articles
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """
    with db.connect() as conn:
        _set_ef_search(conn, k)
        rows = conn.execute(sql, (qvec, qvec, k)).fetchall()
    return [Hit(*r) for r in rows]


def _keyword(query: str, k: int) -> list[Hit]:
    sql = """
        SELECT pmid, title, abstract, journal, pub_year,
               ts_rank_cd(fts, q)::float8 AS score
        FROM articles, websearch_to_tsquery('english', %s) q
        WHERE fts @@ q
        ORDER BY score DESC
        LIMIT %s
    """
    with db.connect() as conn:
        rows = conn.execute(sql, (query, k)).fetchall()
    return [Hit(*r) for r in rows]


def _hybrid(query: str, k: int) -> list[Hit]:
    qvec = db.vec_literal(embed_texts([query])[0])
    sql = """
        WITH vec AS (
            SELECT pmid, row_number() OVER (ORDER BY embedding <=> %s::vector) AS r
            FROM articles
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        ),
        kw AS (
            SELECT pmid, row_number() OVER (ORDER BY ts_rank_cd(fts, q) DESC) AS r
            FROM articles, websearch_to_tsquery('english', %s) q
            WHERE fts @@ q
            ORDER BY r
            LIMIT %s
        ),
        fused AS (
            SELECT pmid,
                   coalesce(1.0 / (%s + vec.r), 0) + coalesce(1.0 / (%s + kw.r), 0) AS score
            FROM vec FULL OUTER JOIN kw USING (pmid)
        )
        SELECT a.pmid, a.title, a.abstract, a.journal, a.pub_year, fused.score::float8
        FROM fused JOIN articles a USING (pmid)
        ORDER BY fused.score DESC
        LIMIT %s
    """
    params = (qvec, qvec, CANDIDATES, query, CANDIDATES, RRF_K, RRF_K, k)
    with db.connect() as conn:
        _set_ef_search(conn, max(CANDIDATES, k))
        rows = conn.execute(sql, params).fetchall()
    return [Hit(*r) for r in rows]
