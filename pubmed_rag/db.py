"""Postgres + pgvector access. One table, one row per article (title+abstract
fits comfortably in a single chunk, so no chunking layer is needed here).

Every row records which embedding model produced its vector, and search
refuses to run against a corpus whose signature doesn't match the current
configuration -- the single most common RAG implementation mistake.
"""
import json
import uuid

import psycopg

from . import config


def connect():
    return psycopg.connect(config.DATABASE_URL)


def stored_dim() -> int | None:
    """Dimension of the existing embedding column, or None if no table yet.
    pgvector stores the dimension as the column's typmod."""
    with connect() as conn:
        row = conn.execute(
            "SELECT atttypmod FROM pg_attribute "
            "WHERE attrelid = to_regclass('articles') AND attname = 'embedding'"
        ).fetchone()
    return row[0] if row else None


def assert_dim_matches() -> None:
    """CREATE TABLE IF NOT EXISTS locks the vector dimension at first init;
    a later EMBED_DIM change needs the table rebuilt, not just re-loaded."""
    dim = stored_dim()
    if dim is not None and dim != config.EMBED_DIM:
        raise SystemExit(
            f"articles.embedding is vector({dim}) but the current config has "
            f"EMBED_DIM={config.EMBED_DIM}. The dimension is baked into the "
            "schema, so re-loading alone can't fix this: drop the table first "
            '(psql: "DROP TABLE articles;"), then run init-db and load again.'
        )


def init_db() -> None:
    assert_dim_matches()
    statements = [
        "CREATE EXTENSION IF NOT EXISTS vector",
        f"""
        CREATE TABLE IF NOT EXISTS articles (
            pmid        text PRIMARY KEY,
            title       text NOT NULL,
            abstract    text NOT NULL,
            journal     text,
            pub_year    int,
            embed_model text NOT NULL,
            embedding   vector({config.EMBED_DIM}) NOT NULL,
            fts         tsvector GENERATED ALWAYS AS
                          (to_tsvector('english', coalesce(title, '') || ' ' || coalesce(abstract, ''))) STORED
        )
        """,
        # HNSW: better query recall/latency than IVFFlat, slower build. The plain
        # vector type indexes up to 2000 dims -- fine for 1536, not for 3072
        # (text-embedding-3-large would need halfvec or the dimensions param).
        """
        CREATE INDEX IF NOT EXISTS articles_embedding_idx
            ON articles USING hnsw (embedding vector_cosine_ops)
        """,
        "CREATE INDEX IF NOT EXISTS articles_fts_idx ON articles USING gin (fts)",
    ]
    with connect() as conn:
        for stmt in statements:
            conn.execute(stmt)


def vec_literal(vec: list[float]) -> str:
    """pgvector accepts '[0.1,0.2,...]' text literals cast with ::vector. Using
    them keeps this demo adapter-free; in production code reach for the
    pgvector-python package instead."""
    return "[" + ",".join(f"{v:.7g}" for v in vec) + "]"


def upsert_articles(rows: list[tuple]) -> int:
    """rows: (pmid, title, abstract, journal, pub_year, embed_model, vec_literal)"""
    sql = """
        INSERT INTO articles (pmid, title, abstract, journal, pub_year, embed_model, embedding)
        VALUES (%s, %s, %s, %s, %s, %s, %s::vector)
        ON CONFLICT (pmid) DO UPDATE SET
            title = EXCLUDED.title, abstract = EXCLUDED.abstract,
            journal = EXCLUDED.journal, pub_year = EXCLUDED.pub_year,
            embed_model = EXCLUDED.embed_model, embedding = EXCLUDED.embedding
    """
    with connect() as conn, conn.cursor() as cur:
        cur.executemany(sql, rows)
    return len(rows)


def stored_signatures() -> set[str]:
    with connect() as conn:
        return {r[0] for r in conn.execute("SELECT DISTINCT embed_model FROM articles")}


def assert_signature_matches() -> None:
    current = config.embedding_signature()
    stale = stored_signatures() - {current}
    if stale:
        raise SystemExit(
            f"Database holds vectors from {sorted(stale)} but the current config "
            f"produces '{current}'. Vectors from different embedding models are "
            "incompatible -- re-load the corpus (load --fresh; if EMBED_DIM also "
            'changed, first "DROP TABLE articles;" and re-run init-db) or restore '
            "the original EMBEDDINGS/EMBED_MODEL settings before searching."
        )


# --- Local-files corpus (a separate chunk table; the PubMed `articles` flow is
#     untouched). One row per chunk; the table name is passed in so several corpora
#     (compliance, general, ...) can share this code, each isolated in its own table
#     with the signature guard applied per table. Defaults to LOCAL_DOCS_TABLE. ---

def _docs_table(table: str | None = None) -> str:
    return table or config.LOCAL_DOCS_TABLE


def init_local_docs(table: str | None = None) -> None:
    table = _docs_table(table)
    statements = [
        "CREATE EXTENSION IF NOT EXISTS vector",
        f"""
        CREATE TABLE IF NOT EXISTS {table} (
            chunk_id    text PRIMARY KEY,
            doc_id      text NOT NULL,
            ordinal     int  NOT NULL,
            title       text,
            category    text,
            source_path text,
            text        text NOT NULL,
            metadata    jsonb,
            embed_model text NOT NULL,
            embedding   vector({config.EMBED_DIM}) NOT NULL,
            fts         tsvector GENERATED ALWAYS AS
                          (to_tsvector('english', coalesce(title, '') || ' ' || coalesce(text, ''))) STORED
        )
        """,
        f"CREATE INDEX IF NOT EXISTS {table}_embedding_idx ON {table} USING hnsw (embedding vector_cosine_ops)",
        f"CREATE INDEX IF NOT EXISTS {table}_fts_idx ON {table} USING gin (fts)",
        f"CREATE INDEX IF NOT EXISTS {table}_doc_idx ON {table} (doc_id)",
    ]
    with connect() as conn:
        for stmt in statements:
            conn.execute(stmt)


def upsert_local_chunks(rows: list[tuple], table: str | None = None) -> int:
    """rows: (chunk_id, doc_id, ordinal, title, category, source_path, text,
    metadata_json, embed_model, vec_literal)"""
    table = _docs_table(table)
    sql = f"""
        INSERT INTO {table}
            (chunk_id, doc_id, ordinal, title, category, source_path, text, metadata, embed_model, embedding)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s::vector)
        ON CONFLICT (chunk_id) DO UPDATE SET
            doc_id = EXCLUDED.doc_id, ordinal = EXCLUDED.ordinal, title = EXCLUDED.title,
            category = EXCLUDED.category, source_path = EXCLUDED.source_path,
            text = EXCLUDED.text, metadata = EXCLUDED.metadata,
            embed_model = EXCLUDED.embed_model, embedding = EXCLUDED.embedding
    """
    with connect() as conn, conn.cursor() as cur:
        cur.executemany(sql, rows)
    return len(rows)


def local_doc_hashes(table: str | None = None) -> dict[str, str]:
    """{doc_id: sha256} for already-stored docs -- lets `ingest` skip unchanged files."""
    table = _docs_table(table)
    with connect() as conn:
        if conn.execute("SELECT to_regclass(%s)", (table,)).fetchone()[0] is None:
            return {}
        rows = conn.execute(
            f"SELECT DISTINCT doc_id, metadata->>'sha256' FROM {table}"
        ).fetchall()
    return {doc_id: h for doc_id, h in rows if h}


def delete_local_doc(doc_id: str, table: str | None = None) -> None:
    with connect() as conn:
        conn.execute(f"DELETE FROM {_docs_table(table)} WHERE doc_id = %s", (doc_id,))


def local_doc_text(doc_id: str, table: str | None = None) -> str:
    """Full text of one document = its chunks in order (for the agent's get_document)."""
    with connect() as conn:
        rows = conn.execute(
            f"SELECT text FROM {_docs_table(table)} WHERE doc_id = %s ORDER BY ordinal",
            (doc_id,),
        ).fetchall()
    return "\n\n".join(r[0] for r in rows)


# --- Interaction log (usage-driven follow-ups + analytics). One row per answered
#     turn. Partly vector-backed: `q_embedding` powers "people also asked" via
#     similarity over past questions. We store QUESTIONS, never re-serve past
#     answers as fact. Self-provisions (CREATE ... IF NOT EXISTS) on first write, so
#     it needs no migration step on the deployed DB. ---

def _interactions_table(table: str | None = None) -> str:
    return table or config.INTERACTIONS_TABLE


def init_interactions(table: str | None = None) -> None:
    table = _interactions_table(table)
    statements = [
        "CREATE EXTENSION IF NOT EXISTS vector",
        f"""
        CREATE TABLE IF NOT EXISTS {table} (
            id                 text PRIMARY KEY,
            created_at         timestamptz NOT NULL DEFAULT now(),
            session_id         text,
            corpus             text,
            mode               text,
            jurisdiction       text,
            question           text,            -- NULL when PHI-flagged
            question_flagged   boolean NOT NULL DEFAULT false,
            answer             text,            -- for analytics only; never re-served as fact
            answered           boolean NOT NULL DEFAULT false,
            source_ids         jsonb,
            n_sources          int NOT NULL DEFAULT 0,
            steps              int NOT NULL DEFAULT 0,
            feedback           smallint,        -- NULL / 1 (up) / -1 (down)
            feedback_note      text,
            suggestions        jsonb,
            clicked_suggestion text,
            q_embedding        vector({config.EMBED_DIM})
        )
        """,
        f"CREATE INDEX IF NOT EXISTS {table}_created_idx ON {table} (created_at DESC)",
        f"CREATE INDEX IF NOT EXISTS {table}_corpus_idx ON {table} (corpus)",
        f"CREATE INDEX IF NOT EXISTS {table}_qembed_idx ON {table} "
        f"USING hnsw (q_embedding vector_cosine_ops)",
    ]
    with connect() as conn:
        for stmt in statements:
            conn.execute(stmt)


def log_interaction(*, session_id=None, corpus=None, mode=None, jurisdiction=None,
                    question=None, question_flagged=False, answer=None, answered=False,
                    source_ids=None, n_sources=0, steps=0, suggestions=None,
                    q_embedding=None, table: str | None = None) -> str:
    """Insert one interaction row; return its id. Self-provisions the table. Callers
    should still wrap in try/except so logging never breaks the answer path."""
    table = _interactions_table(table)
    init_interactions(table)
    iid = uuid.uuid4().hex
    vec = vec_literal(q_embedding) if q_embedding else None  # None -> NULL::vector
    sql = f"""
        INSERT INTO {table}
            (id, session_id, corpus, mode, jurisdiction, question, question_flagged,
             answer, answered, source_ids, n_sources, steps, suggestions, q_embedding)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s::jsonb, %s::vector)
    """
    params = (
        iid, session_id, corpus, mode, jurisdiction,
        (None if question_flagged else question), question_flagged,
        answer, answered, json.dumps(source_ids or []), n_sources, steps,
        json.dumps(suggestions or []), vec,
    )
    with connect() as conn:
        conn.execute(sql, params)
    return iid


def set_feedback(interaction_id: str, value: int, note: str | None = None,
                 table: str | None = None) -> None:
    table = _interactions_table(table)
    with connect() as conn:
        conn.execute(
            f"UPDATE {table} SET feedback = %s, "
            f"feedback_note = COALESCE(%s, feedback_note) WHERE id = %s",
            (value, note, interaction_id),
        )


def set_clicked_suggestion(interaction_id: str, text: str, table: str | None = None) -> None:
    table = _interactions_table(table)
    with connect() as conn:
        conn.execute(
            f"UPDATE {table} SET clicked_suggestion = %s WHERE id = %s",
            (text, interaction_id),
        )


def related_questions(query_embedding, corpus: str, k: int = 5,
                      table: str | None = None) -> list[str]:
    """Past questions most similar to the current one, for "people also asked".
    Only *answerable*, PHI-clean, non-thumbs-down questions in the same corpus.
    Best-effort: returns [] if the table/embedding is unavailable; never raises."""
    if query_embedding is None:
        return []
    table = _interactions_table(table)
    try:
        with connect() as conn:
            if conn.execute("SELECT to_regclass(%s)", (table,)).fetchone()[0] is None:
                return []
            rows = conn.execute(
                f"""
                SELECT question
                FROM {table}
                WHERE corpus = %s AND answered AND NOT question_flagged
                      AND question IS NOT NULL
                      AND feedback IS DISTINCT FROM -1
                      AND q_embedding IS NOT NULL
                ORDER BY q_embedding <=> %s::vector
                LIMIT %s
                """,
                (corpus, vec_literal(query_embedding), k),
            ).fetchall()
        return [r[0] for r in rows if r[0]]
    except Exception:
        return []
