"""General-knowledge corpus: OPTIONAL, additive grounding for the patient-education
modes. A thin binding over the shared engine in `docs.py` -- same as compliance.py
but jurisdiction-neutral (no category filter) and with its own table + prompt.

This corpus is optional: `is_ingested()` lets callers offer its search tool only
when it has actually been populated, so the general/article modes degrade
gracefully to compliance-only grounding when it's empty.
"""
from . import config, db, docs
from .ask import history_messages

TABLE = config.GENERAL_DOCS_TABLE
LABEL = "General-knowledge"

Chunk = docs.Chunk


def is_ingested() -> bool:
    return docs.is_ingested(TABLE)


def search(query: str, k: int = 6, mode: str = "hybrid") -> list[docs.Chunk]:
    return docs.search(TABLE, query, k=k, mode=mode, category=None, label=LABEL)


def build_context(chunks: list[docs.Chunk]) -> str:
    blocks = []
    for c in chunks:
        topic = f"; topic: {c.category}" if c.category else ""
        src = f"[{c.title or c.doc_id}] (general information; source file: {c.doc_id}{topic})"
        blocks.append(f"{src}\n{c.text}")
    return "\n\n---\n\n".join(blocks)


def _prompt() -> str:
    return config.corpus_profile("general").get("system_prompt") or config.GENERAL_SYSTEM_PROMPT


def ask(question: str, k: int = 6, mode: str = "hybrid", history=None) -> str:
    chunks = search(question, k=k, mode=mode)
    if not chunks:
        return "No matching passages in the general-knowledge corpus."
    user_msg = f"General information:\n\n{build_context(chunks)}\n\nQuestion: {question}"
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


# --- Agentic tool surface (used by agent.py's combined toolset) ---
SNIPPET_CHARS = 320


def tool_search(query, k=6):
    try:
        chunks = search(query, k=int(k or 6), mode="hybrid")
    except SystemExit as e:  # corpus not ingested / signature mismatch -> tell the model
        return {"error": str(e)}
    return [
        {
            "doc_id": c.doc_id,
            "corpus": "general",
            "title": c.title,
            "topic": c.category,
            "score": round(c.score, 4),
            "snippet": (c.text or "")[:SNIPPET_CHARS],
        }
        for c in chunks
    ]


def tool_get_document(doc_id):
    text = db.local_doc_text(str(doc_id), table=TABLE)
    if not text:
        return {"doc_id": doc_id, "error": "no such document"}
    return {"doc_id": doc_id, "corpus": "general", "text": text[:16000]}


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_general",
            "description": (
                "Search the curated general-knowledge / patient-education corpus (plain-language "
                "medical information, definitions, treatment explanations). Jurisdiction-neutral. "
                "Returns matching passages with their source doc_id. Use for general or tangential "
                "questions and for background when writing patient articles."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "k": {"type": "integer", "description": "number of passages (default 6)"},
                },
                "required": ["query"],
            },
        },
    },
]

IMPLS = {"search_general": tool_search}
