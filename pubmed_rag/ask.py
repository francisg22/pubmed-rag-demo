"""Grounded Q&A: retrieve abstracts, hand them to a chat model, demand citations.

The clinical-RAG safety pattern in miniature:
  * the model may only use the abstracts we hand it       (grounding)
  * every claim must carry a [PMID ...] citation          (provenance)
  * if the abstracts don't answer it, it must say so      (abstention)
  * output is framed as a draft for clinician review      (human in the loop)
"""
from . import config
from .search import Hit, search

SYSTEM_PROMPT = """\
You are a literature assistant for clinicians, running in a proof-of-concept demo.
Answer ONLY from the PubMed abstracts provided in the user message.
Cite the PMID in square brackets, e.g. [PMID 12345678], after every claim.
If the abstracts do not contain enough information to answer, say exactly that;
do not speculate and do not use outside knowledge.
End every answer with: "Draft for clinician review -- verify against primary sources."
"""


def build_context(hits: list[Hit]) -> str:
    blocks = []
    for h in hits:
        src = f"[PMID {h.pmid}] {h.title} ({h.journal or 'journal unknown'}, {h.pub_year or 'year unknown'})"
        blocks.append(f"{src}\n{h.abstract}")
    return "\n\n---\n\n".join(blocks)


def ask(question: str, k: int = 5) -> str:
    hits = search(question, k=k, mode="hybrid")
    if not hits:
        return "No matching articles in the local corpus -- run `load` first."
    user_msg = f"Abstracts:\n\n{build_context(hits)}\n\nQuestion: {question}"

    if not config.OPENAI_API_KEY:
        return (
            "[no OPENAI_API_KEY set -- showing the grounded prompt that would be sent]\n\n"
            f"SYSTEM:\n{SYSTEM_PROMPT}\nUSER:\n{user_msg}"
        )

    from openai import OpenAI

    resp = OpenAI().chat.completions.create(
        model=config.CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )
    return resp.choices[0].message.content
