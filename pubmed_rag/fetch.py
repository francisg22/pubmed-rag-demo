"""Fetch PubMed records via NCBI E-utilities (public literature -- no patient data).

Two-step dance:
  1. esearch -- turn a query string into a list of PMIDs
  2. efetch  -- pull title/abstract/journal/year for those PMIDs (XML)

Etiquette: without an NCBI API key you get 3 requests/second, so we pause
between calls; batching 200 PMIDs per efetch keeps the request count low.
"""
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import requests

from . import config

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
EFETCH_BATCH = 200

# Full text lives in PubMed Central, and only the open-access subset is
# fetchable (~1/4 to 1/3 of a typical PubMed result set; higher for recent
# papers). PMID -> PMCID via the ID Converter, then efetch from db=pmc.
PMC_IDCONV = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
FULLTEXT_MAX_CHARS = 16_000  # per-article cap so one long paper can't swamp the context


@dataclass
class Article:
    pmid: str
    title: str
    abstract: str
    journal: str | None
    pub_year: int | None


def _params(extra: dict) -> dict:
    p = {"db": "pubmed", "tool": config.NCBI_TOOL}
    if config.NCBI_EMAIL:
        p["email"] = config.NCBI_EMAIL
    if config.NCBI_API_KEY:
        p["api_key"] = config.NCBI_API_KEY
    p.update(extra)
    return p


def _polite_pause() -> None:
    time.sleep(0.11 if config.NCBI_API_KEY else 0.34)


def search_pmids(query: str, max_results: int) -> list[str]:
    resp = requests.get(
        f"{EUTILS}/esearch.fcgi",
        params=_params(
            {"term": query, "retmax": max_results, "retmode": "json", "sort": "relevance"}
        ),
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["esearchresult"]["idlist"]


def fetch_articles(pmids: list[str]) -> list[Article]:
    articles: list[Article] = []
    for i in range(0, len(pmids), EFETCH_BATCH):
        _polite_pause()
        # POST rather than GET so a long PMID list can't overflow the URL
        resp = requests.post(
            f"{EUTILS}/efetch.fcgi",
            data=_params({"id": ",".join(pmids[i : i + EFETCH_BATCH]), "retmode": "xml"}),
            timeout=60,
        )
        resp.raise_for_status()
        articles.extend(_parse(resp.text))
    return articles


def _parse(xml_text: str) -> list[Article]:
    out: list[Article] = []
    for node in ET.fromstring(xml_text).iterfind(".//PubmedArticle"):
        pmid = node.findtext(".//MedlineCitation/PMID")
        art = node.find(".//MedlineCitation/Article")
        if pmid is None or art is None:
            continue
        title = _text(art.find("ArticleTitle"))
        # Abstracts arrive as one or more (optionally labelled) sections.
        sections = []
        for sec in art.iterfind(".//Abstract/AbstractText"):
            body = _text(sec)
            label = sec.get("Label")
            sections.append(f"{label}: {body}" if label else body)
        abstract = "\n".join(s for s in sections if s).strip()
        if not title or not abstract:
            continue  # nothing to embed without an abstract
        journal = art.findtext(".//Journal/Title")
        year = art.findtext(".//Journal/JournalIssue/PubDate/Year")
        if year is None:
            # Irregular issues use MedlineDate, e.g. "2023 Nov-Dec" or "Winter 2024"
            medline = art.findtext(".//Journal/JournalIssue/PubDate/MedlineDate") or ""
            year = next(
                (t for t in medline.replace("-", " ").split() if len(t) == 4 and t.isdigit()),
                None,
            )
        out.append(
            Article(
                pmid=pmid,
                title=title,
                abstract=abstract,
                journal=journal,
                pub_year=int(year) if year else None,
            )
        )
    return out


def _text(el) -> str:
    """Flatten an element that may contain markup children (<i>, <sup>, ...)."""
    return "".join(el.itertext()).strip() if el is not None else ""


def fetch_full_texts(pmids: list[str], max_chars: int = FULLTEXT_MAX_CHARS) -> dict[str, str]:
    """Best-effort open-access full text for the given PMIDs.

    Returns {pmid: body_text} only for articles whose full text is in PMC's
    open-access subset; PMIDs absent from the result have no fetchable full
    text and the caller should fall back to the abstract. Retrieval is
    unaffected by this -- it only enriches what the chat model can quote.
    """
    if not pmids:
        return {}
    idconv = _params({"ids": ",".join(pmids), "format": "json"})
    idconv.pop("db", None)  # ID Converter has no db parameter
    resp = requests.get(PMC_IDCONV, params=idconv, timeout=30)
    resp.raise_for_status()
    pmcids = [r["pmcid"] for r in resp.json().get("records", []) if r.get("pmcid")]
    if not pmcids:
        return {}
    _polite_pause()
    # One efetch for all candidate PMCIDs; map each article back to its PMID.
    resp = requests.post(
        f"{EUTILS}/efetch.fcgi",
        data=_params({"db": "pmc", "id": ",".join(c.replace("PMC", "") for c in pmcids), "retmode": "xml"}),
        timeout=120,
    )
    resp.raise_for_status()
    return _parse_pmc(resp.text, max_chars)


def _parse_pmc(xml_text: str, max_chars: int) -> dict[str, str]:
    out: dict[str, str] = {}
    for art in ET.fromstring(xml_text).iterfind(".//article"):
        pmid = next(
            (a.text for a in art.iterfind(".//article-id") if a.get("pub-id-type") == "pmid"),
            None,
        )
        body = art.find(".//body")
        if not pmid or body is None:
            continue  # no PMID match or not open-access (metadata-only record)
        text = " ".join(t.strip() for t in body.itertext() if t.strip())
        if text:
            out[pmid.strip()] = text[:max_chars]
    return out
