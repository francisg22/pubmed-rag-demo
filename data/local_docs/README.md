# Local documents drop directory

**Unzip the archive into this folder** (the ingester walks the directory tree;
it does not read inside `.zip` files). Supported types: `.pdf`, `.docx`,
`.pptx`, `.txt`, `.md`. The top-level subfolder of each file is captured as its
`category` (e.g. patient-info vs medical).

**Nothing in this directory is committed to git** (it may contain sensitive
files) and **nothing is embedded or sent to OpenAI until you explicitly enable
it.**

## Workflow

1. Drop files/zips here.
2. Analyze locally — extraction + chunking + a PHI screen, **no API calls**:
   ```bash
   python -m pubmed_rag ingest            # report only: file/chunk counts, est. cost, PHI flags
   ```
3. Review the PHI report. Only once you've confirmed the files are clear of
   patient data, enable embedding and re-run:
   ```bash
   ALLOW_EMBEDDING=1 python -m pubmed_rag ingest   # embeds via OpenAI + stores in pgvector
   ```

The `ALLOW_EMBEDDING` gate defaults to **off** — `ingest` will extract, chunk,
and screen but stop before any embedding or network call.
