"""Suggested follow-up questions shown after an answer.

One cheap chat call proposes 2-4 SHORT, in-scope follow-ups that are answerable from
the same passages that grounded the answer, optionally seeded with real past questions
("people also asked") pulled from the interaction log. It degrades to [] when there is
no API key or on any error, so the feature never breaks the answer path.

Only questions are produced here — never re-served answers — so this cannot undermine
the app's grounded-citation guarantee: a suggested question is answered fresh through
the normal retrieval pipeline when clicked.
"""
import json

from . import config

_SYS = """\
You suggest follow-up questions for a grounded medical/compliance retrieval assistant.
Return 2-4 short questions the user might ask next. Rules:
- Each MUST be answerable from the same documents that grounded the answer (stay in
  scope; nothing needing outside knowledge).
- Short (max ~12 words), distinct, and a genuinely useful next step.
- Never suggest questions asking for individualised medical, legal, or personal advice
  (diagnosis, dosing, or decisions for a specific person) — this is general information.
- If candidate questions from past users are provided, prefer/adapt the ones that fit
  the current answer's scope; ignore any that are off-topic, unsafe, or garbled.
Return ONLY compact JSON: {"followups": ["...", "..."]}. No prose, no markdown.
"""


def generate_followups(answer: str, context: str = "", corpus: str = "", mode: str = "",
                       prior_questions=None, max_n: int = 4) -> list[str]:
    if not config.OPENAI_API_KEY or not (answer or "").strip():
        return []
    prior = [p.strip() for p in (prior_questions or []) if p and p.strip()]
    user = (
        f"Assistant answer:\n{answer[:2000]}\n\n"
        f"Grounding context (excerpt):\n{(context or '')[:2500]}\n\n"
        f"Corpus: {corpus or '?'} · mode: {mode or '?'}\n"
    )
    if prior:
        user += (
            "Candidate questions real users asked before (adapt any that are in scope):\n"
            + "\n".join(f"- {p}" for p in prior[:8]) + "\n"
        )
    user += "\nReturn the JSON now."
    try:
        from openai import OpenAI

        resp = OpenAI().chat.completions.create(
            model=config.CHAT_MODEL,
            messages=[
                {"role": "system", "content": _SYS},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
        )
        data = json.loads(resp.choices[0].message.content or "{}")
    except Exception:
        return []

    out: list[str] = []
    seen: set[str] = set()
    for f in data.get("followups", []) if isinstance(data, dict) else []:
        f = (f or "").strip().rstrip("?").strip()
        if not f or f.lower() in seen:
            continue
        seen.add(f.lower())
        out.append(f + "?")
        if len(out) >= max_n:
            break
    return out
