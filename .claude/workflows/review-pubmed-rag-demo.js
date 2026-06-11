export const meta = {
  name: 'review-pubmed-rag-demo',
  description: 'Review the new PubMed RAG demo across four dimensions, adversarially verify findings',
  phases: [
    { title: 'Review', detail: 'four parallel domain reviewers' },
    { title: 'Verify', detail: 'adversarial check of each finding' },
  ],
}

const ROOT = '/home/grantfrancis/claude_projects/clinic_chatbot'

const FINDINGS_SCHEMA = {
  type: 'object',
  properties: {
    findings: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          file: { type: 'string' },
          title: { type: 'string' },
          detail: { type: 'string' },
          severity: { type: 'string', enum: ['high', 'medium', 'low'] },
          suggestedFix: { type: 'string' },
        },
        required: ['file', 'title', 'detail', 'severity'],
      },
    },
  },
  required: ['findings'],
}

const VERDICT_SCHEMA = {
  type: 'object',
  properties: {
    isReal: { type: 'boolean' },
    reasoning: { type: 'string' },
  },
  required: ['isReal', 'reasoning'],
}

const COMMON = `You are reviewing a small, newly written teaching demo: a PubMed -> embeddings -> Postgres/pgvector RAG pipeline at ${ROOT}. Python package is in ${ROOT}/pubmed_rag/ (config.py, fetch.py, embed.py, db.py, search.py, ask.py, cli.py). It was just executed successfully end-to-end (init-db, load of 29 real PubMed abstracts with the offline 'hashbow' embedder, vector/keyword/hybrid search, ask). Postgres 17 + pgvector runs in a container on localhost:5433. Report only REAL problems that would bite a user (bugs, incorrect API usage, wrong claims), not style preferences. This is a deliberately minimal teaching demo -- 'could add feature X' is not a finding.`

const DIMENSIONS = [
  {
    key: 'ncbi',
    prompt: `${COMMON}
Read ${ROOT}/pubmed_rag/fetch.py closely. Check the NCBI E-utilities usage for correctness: esearch/efetch parameter names and values, JSON/XML response shapes actually returned by these endpoints, the XML paths used for PMID/title/abstract/journal/year, handling of labelled AbstractText sections and MedlineDate, rate-limit handling (3/s keyless, 10/s with key), POST usage for efetch, and any edge case in the parsing that would crash or silently drop data. Use web search/fetch of NCBI E-utilities docs if you need to confirm specifics.`,
  },
  {
    key: 'openai',
    prompt: `${COMMON}
Read ${ROOT}/pubmed_rag/embed.py, ${ROOT}/pubmed_rag/ask.py, and ${ROOT}/pubmed_rag/config.py. Check OpenAI API usage: embeddings.create parameters (model, input list, dimensions param validity for text-embedding-3-small), batch-size and token-limit reasoning, response shape (data[].embedding, data[].index), chat.completions.create usage, and whether the default CHAT_MODEL 'gpt-4o-mini' is a reasonable still-served default in mid-2026 (check the web if unsure). Also check the hashbow fallback for determinism bugs.`,
  },
  {
    key: 'sql',
    prompt: `${COMMON}
Read ${ROOT}/pubmed_rag/db.py and ${ROOT}/pubmed_rag/search.py. Check all SQL for correctness against Postgres 17 + pgvector: the DDL (generated tsvector column, HNSW index with vector_cosine_ops, GIN index), the vector text-literal formatting (%.7g precision -- any correctness risk?), the upsert, the cosine-distance queries and whether their shape can use the HNSW index, and especially the hybrid RRF query: placeholder order vs the params tuple, window-function rank vs LIMIT ordering semantics, FULL OUTER JOIN USING semantics, score arithmetic. Also psycopg3 usage (multiple statements, executemany, context managers, transaction behavior). You may run psql via: podman exec pubmed-rag-db psql -U rag -d pubmed_rag -c '...' to test queries against the live loaded database (29 rows, hashbow vectors).`,
  },
  {
    key: 'docs',
    prompt: `${COMMON}
Read ${ROOT}/README.md, ${ROOT}/.env.example, ${ROOT}/docker-compose.yml, ${ROOT}/requirements.txt and cross-check every claim and command against the actual code in ${ROOT}/pubmed_rag/. Does the quickstart sequence actually work in order? Are flag names, env var names, defaults (port 5433, model names, dims), table/column names, and factual claims (pricing, token limits, HNSW 2000-dim ceiling, pgvector behavior) consistent with the code and reality? Flag anything a reader following the README verbatim would trip over.`,
  },
]

phase('Review')
const results = await pipeline(
  DIMENSIONS,
  (d) => agent(d.prompt, { label: `review:${d.key}`, phase: 'Review', schema: FINDINGS_SCHEMA }),
  (review, d) =>
    parallel(
      (review?.findings ?? []).map((f) => () =>
        agent(
          `${COMMON}
A reviewer claims the following problem in the demo. Adversarially verify it: read the relevant file(s), reproduce or disprove the claim (you may query the live DB via: podman exec pubmed-rag-db psql -U rag -d pubmed_rag -c '...', or run ${ROOT}/.venv/bin/python for quick checks). Default to isReal=false unless the problem is concrete and would actually affect a user.
Claim [${f.severity}] in ${f.file}: ${f.title}
Detail: ${f.detail}`,
          { label: `verify:${d.key}:${f.title.slice(0, 40)}`, phase: 'Verify', schema: VERDICT_SCHEMA }
        ).then((v) => ({ ...f, dimension: d.key, verdict: v }))
      )
    )
)

const all = results.filter(Boolean).flat().filter(Boolean)
const confirmed = all.filter((f) => f.verdict?.isReal)
log(`${all.length} findings raised, ${confirmed.length} confirmed`)
return {
  confirmed,
  rejected: all.filter((f) => !f.verdict?.isReal).map((f) => ({ title: f.title, why: f.verdict?.reasoning })),
}
