"""Lightweight PHI/PII screen -- a decision aid, NOT a guarantee.

Flags common identifier patterns so you can review a corpus before allowing
anything to be embedded or sent to OpenAI. It catches obvious structured
identifiers (SSN, MRN, phone, email, dates); it does NOT reliably catch free-text
patient names. Treat a clean result as "no obvious identifiers found", not
"confirmed free of PHI".
"""
import re

PATTERNS = {
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "mrn": re.compile(r"\bMRN[:#]?\s*\d{4,}\b", re.I),
    "phone": re.compile(r"\b\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"),
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "dob": re.compile(r"\b(?:DOB|date of birth)\b", re.I),
    "date": re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),
}


def screen(text: str) -> dict[str, int]:
    """Return {pattern_name: match_count} for every pattern that matched."""
    out: dict[str, int] = {}
    for name, pat in PATTERNS.items():
        hits = pat.findall(text or "")
        if hits:
            out[name] = len(hits)
    return out
