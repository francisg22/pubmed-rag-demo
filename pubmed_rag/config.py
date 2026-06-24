"""Central configuration, read once from the environment (and .env if present).

Every stored vector is married to the model that produced it. EMBED_MODEL and
EMBED_DIM are recorded with each row and checked again at query time, so you
can't accidentally compare vectors from two different embedding spaces.
"""
import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://rag:rag@localhost:5433/pubmed_rag"
)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# "openai" needs OPENAI_API_KEY; "hashbow" is a free offline stand-in that
# lets the whole pipeline run without any external service.
EMBEDDINGS = os.environ.get("EMBEDDINGS", "openai" if OPENAI_API_KEY else "hashbow")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-3-small")
EMBED_DIM = int(os.environ.get("EMBED_DIM", "1536"))

CHAT_MODEL = os.environ.get("CHAT_MODEL", "gpt-4o-mini")

# NCBI etiquette: identify your tool; an API key raises the rate limit 3 -> 10 req/s.
NCBI_API_KEY = os.environ.get("NCBI_API_KEY", "")
NCBI_TOOL = os.environ.get("NCBI_TOOL", "clinic-chatbot-rag-demo")
NCBI_EMAIL = os.environ.get("NCBI_EMAIL", "")


def embedding_signature() -> str:
    """Identifier stored alongside every vector; search refuses mixed signatures."""
    if EMBEDDINGS == "openai":
        return f"openai:{EMBED_MODEL}:{EMBED_DIM}"
    return f"hashbow:{EMBED_DIM}"


# --- Local-files corpus ---
LOCAL_DOCS_DIR = os.environ.get("LOCAL_DOCS_DIR", "data/local_docs")
LOCAL_DOCS_TABLE = os.environ.get("LOCAL_DOCS_TABLE", "local_docs")

# HARD SAFETY GATE. While this is off, `ingest` only extracts, chunks, and
# screens locally -- it makes NO embedding calls and sends NOTHING to OpenAI,
# and writes nothing to the database. Turn it on (ALLOW_EMBEDDING=1) only after
# you have confirmed the local corpus is free of PHI/PII.
ALLOW_EMBEDDING = os.environ.get("ALLOW_EMBEDDING", "").lower() in ("1", "true", "yes")


# --- Corpus profiles ---
# Per-corpus presentation + storage metadata. Drives the GUI framing now, and
# (once wired) the retrieval table and citation style. `retrieval_ready` flags
# whether search/ask/agent are wired for a corpus yet -- only PubMed is so far.
CORPORA = {
    "pubmed": {
        "label": "PubMed RAG",
        "title": "🔬 PubMed RAG — clinician literature assistant",
        "icon": "🔬",
        "banner": "public literature only, no PHI · answers are drafts for clinician review",
        "placeholder": "Ask a clinical question…",
        "table": "articles",
        "unit": "articles",
        "citation": "PMID",
        "retrieval_ready": True,
    },
    "local_docs": {
        "label": "Local Docs RAG",
        "title": "📁 Local Docs RAG — internal document assistant",
        "icon": "📁",
        "banner": "⚠️ local documents (NOT public literature) — may contain sensitive material; confirm PHI handling",
        "placeholder": "Ask about the local documents…",
        "table": LOCAL_DOCS_TABLE,
        "unit": "chunks",
        "citation": "source file",
        "retrieval_ready": False,
    },
}

# Active corpus, selectable at startup via the CORPUS flag (e.g. CORPUS=local_docs).
CORPUS = os.environ.get("CORPUS", "pubmed")


def corpus_profile(name: str | None = None) -> dict:
    """Profile for the named corpus (default: the active CORPUS)."""
    return CORPORA.get(name or CORPUS, CORPORA["pubmed"])
