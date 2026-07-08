# General-knowledge corpus drop directory

An **optional, additive** corpus of curated **general / patient-education**
information (plain-language treatment explanations, definitions, background
medical info). It grounds the compliance assistant's two patient-education modes:

- **General assistant** — grounded answers to basic and tangential questions.
- **Article writer** — drafts patient-information articles.

It is **jurisdiction-neutral** and kept **separate** from the compliance corpus so
official policy is never confused with general information. The whole feature works
with this corpus **off / empty** (compliance-only grounding); it only takes effect
once ingested and switched on (GUI toggle / `--include-general`).

Supported types: `.pdf`, `.docx`, `.pptx`, `.xlsx`, `.txt`, `.md`. The top-level
subfolder of each file is captured as its `category` (use it as a topic/specialty
folder, e.g. `cardiology/…`, `glossary/…`).

## What to put here (curation matters — answer quality = corpus quality)

Keep it **tight and on-topic** (the clinic's specialties + questions patients
actually ask), and **license-clean**. Prefer public-domain government content:

- **MedlinePlus Health Topics** (NIH/NLM) — patient-education, public domain.
- **MeSH scope notes** (NLM) — definitions without the bloat.
- **CDC**, **NCI/PDQ patient summaries** — public domain, patient-facing.
- **NHS Health A–Z** (UK, OGL) / **healthdirect** (AU, check terms) / **WHO fact sheets**.
- A **hand-written `glossary.md`** of on-topic terms is often the highest-signal option.
- **Avoid** copyrighted references (Merck Manual, UpToDate, textbooks).

## Rules

- **No PHI/PII.** This is general education only — never drop patient records here.
  Extraction runs a PHI screen; review it before enabling embedding.
- **Nothing here is committed to git** (except this README and any `example-*.md`
  templates) and **nothing is embedded or sent to OpenAI until you enable it.**

## Workflow

1. Drop curated files here (optionally in topic subfolders).
2. Analyze locally — extraction + chunking + PHI screen, **no API calls**:
   ```bash
   python -m pubmed_rag ingest --corpus general            # report only
   ```
3. Review the PHI report. Once the files are confirmed clear, enable embedding:
   ```bash
   ALLOW_EMBEDDING=1 python -m pubmed_rag ingest --corpus general
   ```
4. In the GUI, pick the **General assistant** or **Article writer** engine and tick
   **Include general-knowledge corpus** (or pass `--include-general` on the CLI).
