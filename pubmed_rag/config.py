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
