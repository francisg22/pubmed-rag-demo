PubMed RAG demo - example outputs
=================================

Corpus: ~400+ PubMed articles across 7+ clinical topics (metformin/CKD,
statins, atrial fibrillation/anticoagulation, HFpEF, SGLT2 inhibitors,
hypertension, COPD), embedded with OpenAI text-embedding-3-small (1536-d).
The count grows as you load more topics, so it may differ from older captures.

Pipeline:
  - Retrieval: hybrid (semantic embeddings + BM25 keyword, fused by RRF).
    Embeddings are built on title+abstract only.
  - Context: at answer time, open-access FULL TEXT is fetched from PubMed
    Central for the top hits where available (~1/4-1/3 of articles; the rest
    fall back to the abstract). This enriches what can be quoted; it does NOT
    change retrieval.
  - Answer (two modes):
      * one-shot (ask)   -- one fixed search, then answer
      * agentic (agent)  -- the model drives retrieval via tools, in a loop
    Both: gpt-4o-mini, grounded only in retrieved sources, every claim with a
    [PMID] citation AND a verbatim supporting quote. A Streamlit GUI (app.py)
    wraps both modes with a sources panel and a live agent trace.

Files
-----
01_metformin_ckd.txt                       metformin safety in moderate CKD
02_sglt2_heart_failure.txt                 SGLT2 inhibitors & HF outcomes
03_statins_primary_prevention_elderly.txt  statins, primary prevention >75
04_doac_vs_warfarin_af.txt                 DOACs vs warfarin in AF
05_hfpef_treatment.txt                     treatments for HFpEF
06_copd_exacerbation.txt                   acute COPD exacerbation
07_hypertension_bp_target.txt              recommended BP target
08_abstention_pneumonia.txt                out-of-corpus -> abstains
09_semantic_vs_keyword.txt                 dense vs lexical retrieval
10_clarified_doac_elderly.txt              clarifying-question intake
11_adversarial_ignore_sources.txt          jailbreak attempt -> refused
12_prompt_injection.txt                    poisoned source -> ignored
13_full_text_vs_abstract.txt               full-text vs abstract-only
14_agentic_multistep.txt                   agent tool-call trace + answer
15_gui_conversation.txt                    GUI 4-turn session (incl. memory)

Safety behaviors demonstrated: grounding, per-claim PMID + verbatim quote,
abstention (#08), jailbreak refusal (#11), prompt-injection resistance (#12),
and model-driven multi-step retrieval that stays grounded (#14). #12 also shows
a limit: the model neutralizes the injected command but still cites the poisoned
source -- corpus trust still matters.
