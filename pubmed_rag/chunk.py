"""Split extracted document text into overlapping, token-bounded chunks.

Local files (reports, slide decks) have no abstract to embed, so we embed chunks
of the body. Chunks are token-bounded so each fits the embedding model's input
cap, with overlap so a fact spanning a boundary still lands whole in one chunk.
A sliding token window keeps this robust across messy extracted text; splitting
on section/paragraph boundaries is a later refinement.
"""
from dataclasses import dataclass

CHUNK_TOKENS = 600    # target tokens per chunk
OVERLAP_TOKENS = 90   # ~15% overlap between consecutive chunks


@dataclass
class Chunk:
    ordinal: int
    text: str
    n_tokens: int


def _encoder():
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")  # encoder for text-embedding-3-*
    except Exception:
        return None


def chunk_text(text: str, chunk_tokens: int = CHUNK_TOKENS,
               overlap_tokens: int = OVERLAP_TOKENS) -> list[Chunk]:
    text = (text or "").strip()
    if not text:
        return []
    enc = _encoder()
    if enc is None:  # char-based fallback (~4 chars/token) if tiktoken is unavailable
        return _chunk_chars(text, chunk_tokens * 4, overlap_tokens * 4)

    ids = enc.encode(text)
    step = max(1, chunk_tokens - overlap_tokens)
    chunks: list[Chunk] = []
    ordinal = 0
    for start in range(0, len(ids), step):
        window = ids[start:start + chunk_tokens]
        if not window:
            break
        piece = enc.decode(window).strip()
        if piece:
            chunks.append(Chunk(ordinal, piece, len(window)))
            ordinal += 1
        if start + chunk_tokens >= len(ids):
            break
    return chunks


def _chunk_chars(text: str, size: int, overlap: int) -> list[Chunk]:
    step = max(1, size - overlap)
    chunks: list[Chunk] = []
    ordinal = 0
    for start in range(0, len(text), step):
        piece = text[start:start + size].strip()
        if piece:
            chunks.append(Chunk(ordinal, piece, len(piece) // 4))
            ordinal += 1
        if start + size >= len(text):
            break
    return chunks
