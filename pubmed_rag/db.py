"""Postgres + pgvector access. One table, one row per article (title+abstract
fits comfortably in a single chunk, so no chunking layer is needed here).

Every row records which embedding model produced its vector, and search
refuses to run against a corpus whose signature doesn't match the current
configuration -- the single most common RAG implementation mistake.
"""
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
#     untouched). One row per chunk; the table name comes from config so a corpus
#     stays isolated and the signature guard applies per table. ---

def init_local_docs() -> None:
    table = config.LOCAL_DOCS_TABLE
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


def upsert_local_chunks(rows: list[tuple]) -> int:
    """rows: (chunk_id, doc_id, ordinal, title, category, source_path, text,
    metadata_json, embed_model, vec_literal)"""
    table = config.LOCAL_DOCS_TABLE
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
