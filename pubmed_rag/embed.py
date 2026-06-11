"""Turn text into vectors.

Two providers:
  * openai  -- text-embedding-3-small via /v1/embeddings (the real demo)
  * hashbow -- deterministic hashed bag-of-words, so the pipeline runs offline
               at zero cost. Texts sharing words land near each other, which
               makes offline search results sensible-looking, but it captures
               no semantics ("MI" and "heart attack" stay far apart). Use it
               to test plumbing, never to judge retrieval quality.

The same provider+model MUST embed both documents and queries: vectors from
different models live in incompatible spaces and comparing them is meaningless.
"""
import hashlib
import math
import re

from . import config

# The API allows up to 2048 inputs and at most 300,000 total tokens per
# request. 100 abstracts (~300 tokens each) sit far below that, but 100
# inputs at MAX_CHARS (~7.5k tokens each) would not -- shrink the batch
# or account for tokens per batch if you embed longer documents
# (e.g. full-text articles).
OPENAI_BATCH = 100
# text-embedding-3 inputs cap at 8192 tokens (~4 chars/token on average).
MAX_CHARS = 30_000

_client = None


def embed_texts(texts: list[str]) -> list[list[float]]:
    texts = [t[:MAX_CHARS] for t in texts]
    if config.EMBEDDINGS == "openai":
        return _openai(texts)
    return [_hashbow(t) for t in texts]


def _openai(texts: list[str]) -> list[list[float]]:
    global _client
    if _client is None:
        from openai import OpenAI

        _client = OpenAI()  # reads OPENAI_API_KEY from the environment
    vectors: list[list[float]] = []
    for i in range(0, len(texts), OPENAI_BATCH):
        resp = _client.embeddings.create(
            model=config.EMBED_MODEL,
            input=texts[i : i + OPENAI_BATCH],
            dimensions=config.EMBED_DIM,
        )
        vectors.extend(d.embedding for d in sorted(resp.data, key=lambda d: d.index))
    return vectors


_TOKEN = re.compile(r"[a-z0-9]+")


def _hashbow(text: str) -> list[float]:
    """Hashing trick: each token hashes to a dimension; counts, then normalize.
    Deterministic, so re-runs and queries are consistent with the stored corpus."""
    vec = [0.0] * config.EMBED_DIM
    for token in _TOKEN.findall(text.lower()):
        h = int.from_bytes(hashlib.sha1(token.encode()).digest()[:8], "big")
        vec[h % config.EMBED_DIM] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]
