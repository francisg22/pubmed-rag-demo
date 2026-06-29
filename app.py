"""Streamlit GUI for the RAG demo.

A thin presentation layer over the existing package:
  - One-shot RAG  -> pubmed_rag.ask  (fixed hybrid retrieval, then answer)
  - Agentic RAG   -> pubmed_rag.agent (the model drives retrieval via tools)

The active corpus (PubMed vs the local-files corpus) drives the title, banner,
and stats. Retrieval is currently wired for PubMed only; other corpora show an
ingestion-only notice rather than silently answering from the wrong data.

Run with:  streamlit run app.py     (CORPUS=local_docs streamlit run app.py to start on that corpus)
"""
import streamlit as st

from pubmed_rag import agent as agent_mod
from pubmed_rag import ask as ask_mod
from pubmed_rag import compliance as compliance_mod
from pubmed_rag import config, db, fetch
from pubmed_rag.search import search

# Browser-tab title/icon are fixed at launch from the CORPUS flag; the in-page
# title reacts live to the sidebar corpus selector below.
_launch = config.corpus_profile()
st.set_page_config(page_title=f"{_launch['label']} (POC)", page_icon=_launch["icon"], layout="wide")
SNIPPET = 320


@st.cache_data(ttl=60)
def corpus_info(table):
    """(total, models) for a corpus table, or None if it doesn't exist yet."""
    with db.connect() as conn:
        if conn.execute("SELECT to_regclass(%s)", (table,)).fetchone()[0] is None:
            return None
        total = conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        models = [r[0] for r in conn.execute(f"SELECT DISTINCT embed_model FROM {table}")]
    return {"total": total, "models": models}


def render_sources(sources):
    full = sum(1 for s in sources if s["full_text"])
    with st.expander(f"📚 Sources used ({len(sources)} · {full} full text)"):
        for s in sources:
            badge = "🟢 full text" if s["full_text"] else "⚪ abstract"
            st.markdown(
                f"**[PMID {s['pmid']}]** {s['title']}  \n"
                f"*{s['journal'] or 'journal unknown'}, {s['year'] or '?'}* · "
                f"{badge} · score {s['score']:.4f}"
            )
            st.caption(s["snippet"])


def render_compliance_sources(chunks):
    with st.expander(f"📚 Sources used ({len(chunks)})"):
        for c in chunks:
            st.markdown(
                f"**{c.title or c.doc_id}** — *{c.jurisdiction or 'jurisdiction unknown'}*  \n"
                f"`{c.doc_id}` · score {c.score:.4f}"
            )
            st.caption((c.text or "")[:SNIPPET])


def render_trace(trace):
    with st.expander(f"🔧 What the agent did ({len(trace)} events)"):
        for ev in trace:
            if ev["type"] == "tool_call":
                args = ", ".join(f"{key}={val!r}" for key, val in ev["args"].items())
                st.markdown(f"→ `{ev['name']}`({args})")
            else:
                st.caption(f"   ↳ {ev['result']}")


def one_shot_answer(q, k, mode, abstract_only, history=None):
    """Single retrieval + single answer, reusing ask.py's prompt/context so the
    GUI shows the exact sources that fed the answer (no double fetch). Prior turns
    (dialogue only) are prepended for conversation memory; sources are re-retrieved
    fresh every turn and never carried forward."""
    hits = search(q, k=k, mode=mode)
    if not hits:
        return "No matching articles in the local corpus.", []
    ft = {} if abstract_only else fetch.fetch_full_texts([h.pmid for h in hits])
    user_msg = f"Sources:\n\n{ask_mod.build_context(hits, ft)}\n\nQuestion: {q}"
    if config.OPENAI_API_KEY:
        from openai import OpenAI

        messages = [
            {"role": "system", "content": ask_mod.SYSTEM_PROMPT},
            *ask_mod.history_messages(history),
            {"role": "user", "content": user_msg},
        ]
        answer = (
            OpenAI()
            .chat.completions.create(model=config.CHAT_MODEL, messages=messages)
            .choices[0]
            .message.content
        )
    else:
        answer = "[no OPENAI_API_KEY — showing the assembled grounded prompt]\n\n" + user_msg
    sources = [
        {
            "pmid": h.pmid,
            "title": h.title,
            "journal": h.journal,
            "year": h.pub_year,
            "score": h.score,
            "full_text": h.pmid in ft,
            "snippet": (h.abstract or "")[:SNIPPET],
        }
        for h in hits
    ]
    return answer, sources


# ----- sidebar (part 1): corpus selection + status (shown for every corpus) -----
with st.sidebar:
    st.header("Settings")
    keys = list(config.CORPORA)
    corpus_key = st.selectbox(
        "Corpus",
        keys,
        index=keys.index(config.CORPUS) if config.CORPUS in keys else 0,
        format_func=lambda k: config.CORPORA[k]["label"],
    )
    prof = config.corpus_profile(corpus_key)

    st.divider()
    try:
        info = corpus_info(prof["table"])
        if info is None:
            st.metric("Corpus", "not ingested")
            st.caption(f"Table `{prof['table']}` is empty / not created yet.")
        else:
            st.metric("Corpus", f"{info['total']} {prof['unit']}")
            st.caption("Embeddings: " + ", ".join(info["models"]))
    except Exception as e:
        st.error(f"Database not reachable — is the container up?\n\n{e}")
    if not config.OPENAI_API_KEY:
        st.warning("No OPENAI_API_KEY set — answers fall back to showing the assembled prompt.")

# ----- header (reacts to the corpus selector) -----
st.title(prof["title"])
st.caption(f"Proof of concept · {prof['banner']}")

# ----- corpora whose retrieval isn't wired yet: ingestion-only notice -----
if not prof["retrieval_ready"]:
    st.info(
        f"**{prof['label']}** is set up for **ingestion only** — retrieval and chat "
        "aren't wired to this corpus yet.\n\n"
        "Ingest local files with `python -m pubmed_rag ingest` (extraction, chunking, "
        "and a PHI screen run locally; embedding stays gated off until you set "
        "`ALLOW_EMBEDDING=1`). Switch the **Corpus** selector to PubMed to use the assistant."
    )
    st.stop()

# ----- sidebar (part 2): controls (only for a retrieval-ready corpus) -----
is_compliance = corpus_key == "compliance"
jurisdiction = None
c_agentic = False
c_max_steps = 6
with st.sidebar:
    if is_compliance:
        c_agentic = st.radio("Engine", ["One-shot RAG", "Agentic (tool-calling)"]).startswith("Agentic")
        juris = st.selectbox("Jurisdiction", ["All", "US", "UK", "Australia"])
        jurisdiction = None if juris == "All" else juris
        k = st.slider("Sources (k)", 3, 12, 6)
        if c_agentic:
            c_max_steps = st.slider("Max tool-call rounds", 2, 10, 6)
        agentic = False
    else:
        engine = st.radio("Engine", ["Agentic (model drives retrieval)", "One-shot RAG"])
        agentic = engine.startswith("Agentic")
        k = st.slider("Sources (k)", 3, 12, 6)
        if agentic:
            max_steps = st.slider("Max tool-call rounds", 2, 10, 6)
            mode, abstract_only = "hybrid", False
        else:
            mode = st.selectbox("Retrieval mode", ["hybrid", "vector", "keyword"])
            abstract_only = st.checkbox("Abstract-only (skip full-text fetch)", value=False)
            max_steps = None
    st.caption(f"Chat model: `{config.CHAT_MODEL}`")
    st.caption(
        f"🧠 Memory: last {ask_mod.HISTORY_TURNS} turns carried as dialogue; "
        "sources are re-retrieved each turn (never carried)."
    )
    if st.button("Clear conversation"):
        st.session_state.history = []

# ----- conversation history -----
if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant"):
        st.markdown(turn["answer"])
        if turn.get("trace"):
            render_trace(turn["trace"])
        if turn.get("sources"):
            render_sources(turn["sources"])

# ----- input -----
q = st.chat_input(prof["placeholder"])
if q:
    # Prior turns become conversation memory (dialogue text only). session_state
    # holds only earlier turns at this point — the current one is appended after.
    prior = [(t["question"], t["answer"]) for t in st.session_state.history]
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        if is_compliance and c_agentic:
            status = st.status("Working over the compliance corpus…", expanded=True)

            def on_event(ev):
                if ev["type"] == "tool_call":
                    a = ", ".join(f"{key}={val!r}" for key, val in ev["args"].items())
                    status.write(f"→ `{ev['name']}`({a})")
                else:
                    status.write(f"   ↳ {ev['result']}")

            q_agent = q if not jurisdiction else f"{q}\n\n(Restrict to the {jurisdiction} jurisdiction.)"
            result = agent_mod.run_agent(
                q_agent, k=k, max_steps=c_max_steps, on_event=on_event,
                history=prior, corpus="compliance",
            )
            status.update(label=f"Done — {result['steps']} step(s)", state="complete")
            st.markdown(result["answer"])
            render_trace(result["trace"])
            st.session_state.history.append(
                {"question": q, "answer": result["answer"], "trace": result["trace"]}
            )
        elif is_compliance:
            chunks = []
            try:
                with st.spinner("Searching compliance documents…"):
                    chunks = compliance_mod.search(q, k=k, mode="hybrid", jurisdiction=jurisdiction)
                    if not chunks:
                        answer = "No matching passages in the compliance corpus" + (
                            f" for {jurisdiction}." if jurisdiction else "."
                        )
                    else:
                        user_msg = (
                            f"Compliance documents:\n\n{compliance_mod.build_context(chunks)}"
                            f"\n\nQuestion: {q}"
                        )
                        prompt = prof.get("system_prompt") or config.COMPLIANCE_SYSTEM_PROMPT
                        if config.OPENAI_API_KEY:
                            from openai import OpenAI

                            msgs = [
                                {"role": "system", "content": prompt},
                                *ask_mod.history_messages(prior),
                                {"role": "user", "content": user_msg},
                            ]
                            answer = (
                                OpenAI()
                                .chat.completions.create(model=config.CHAT_MODEL, messages=msgs)
                                .choices[0]
                                .message.content
                            )
                        else:
                            answer = "[no OPENAI_API_KEY — showing the assembled grounded prompt]\n\n" + user_msg
            except SystemExit as e:  # corpus not ingested / signature mismatch
                answer, chunks = str(e), []
            st.markdown(answer)
            if chunks:
                render_compliance_sources(chunks)
            st.session_state.history.append({"question": q, "answer": answer})
        elif agentic:
            status = st.status("Searching the literature…", expanded=True)

            def on_event(ev):
                if ev["type"] == "tool_call":
                    args = ", ".join(f"{key}={val!r}" for key, val in ev["args"].items())
                    status.write(f"→ `{ev['name']}`({args})")
                else:
                    status.write(f"   ↳ {ev['result']}")

            result = agent_mod.run_agent(
                q, k=k, max_steps=max_steps, on_event=on_event, history=prior
            )
            status.update(label=f"Done — {result['steps']} step(s)", state="complete")
            st.markdown(result["answer"])
            render_trace(result["trace"])
            st.session_state.history.append(
                {"question": q, "answer": result["answer"], "trace": result["trace"]}
            )
        else:
            with st.spinner("Retrieving and answering…"):
                answer, sources = one_shot_answer(q, k, mode, abstract_only, history=prior)
            st.markdown(answer)
            render_sources(sources)
            st.session_state.history.append(
                {"question": q, "answer": answer, "sources": sources}
            )
