"""Compliance corpus: retrieval + grounded Q&A over the internal policy documents.

A thin binding over the shared engine in `docs.py`: it fixes the compliance table,
maps the `category` column to a jurisdiction (US / UK / Australia), and uses the
restricted compliance system prompt. The PubMed flow is completely untouched.

No network calls happen until a query needs a query embedding (vector / hybrid
modes) or the chat answer -- i.e. only once OPENAI_API_KEY is set.
"""
from . import config, db, docs
from .ask import history_messages

TABLE = config.LOCAL_DOCS_TABLE
LABEL = "Compliance"

# Re-exported so existing callers keep importing Chunk from here.
Chunk = docs.Chunk


def _assert_ready() -> None:
    docs.assert_ready(TABLE, LABEL)


def search(query: str, k: int = 6, mode: str = "hybrid", jurisdiction: str | None = None) -> list[docs.Chunk]:
    return docs.search(TABLE, query, k=k, mode=mode, category=jurisdiction, label=LABEL)


def build_context(chunks: list[docs.Chunk]) -> str:
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


# --- Agentic tool surface (used by agent.py) ---
SNIPPET_CHARS = 320


def tool_search(query, k=6, jurisdiction=None):
    try:
        chunks = search(query, k=int(k or 6), mode="hybrid", jurisdiction=jurisdiction)
    except SystemExit as e:  # corpus not ingested / signature mismatch -> tell the model
        return {"error": str(e)}
    return [
        {
            "doc_id": c.doc_id,
            "corpus": "compliance",
            "title": c.title,
            "jurisdiction": c.jurisdiction,
            "score": round(c.score, 4),
            "snippet": (c.text or "")[:SNIPPET_CHARS],
        }
        for c in chunks
    ]


def tool_get_document(doc_id):
    text = db.local_doc_text(str(doc_id), table=TABLE)
    if not text:
        return {"doc_id": doc_id, "error": "no such document"}
    return {"doc_id": doc_id, "corpus": "compliance", "text": text[:16000]}


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
