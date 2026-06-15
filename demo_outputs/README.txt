PubMed RAG demo - example outputs
=================================

379 articles
     379 embedded with openai:text-embedding-3-small:1536 (years 2003-2026)

Corpus: ~379 PubMed abstracts across 7 clinical topics (metformin/CKD,
statins, atrial fibrillation/anticoagulation, HFpEF, SGLT2 inhibitors,
hypertension, COPD).

Pipeline:
  - Retrieval: hybrid (OpenAI text-embedding-3-small, 1536-d, + BM25, RRF).
    Embeddings are built on title+abstract.
  - Context: at answer time, open-access FULL TEXT is fetched from PubMed
    Central for the top hits where available (~1/4-1/3 of articles; rest
    fall back to the abstract). This enriches what can be quoted; it does
    NOT change retrieval.
  - Answer: gpt-4o-mini, grounded only in the provided sources. Every
    claim carries a [PMID] citation AND a verbatim supporting quote.

Files
-----
01_metformin_ckd.txt                       metformin safety in moderate CKD
02_sglt2_heart_failure.txt                 SGLT2 inhibitors & HF hospitalization
03_statins_primary_prevention_elderly.txt  statins for primary prevention >75
04_doac_vs_warfarin_af.txt                 DOACs vs warfarin in atrial
fibrillation
05_hfpef_treatment.txt                     treatments for HFpEF
06_copd_exacerbation.txt                   managing an acute COPD exacerbation
07_hypertension_bp_target.txt              recommended BP target
08_abstention_pneumonia.txt                OUT-OF-CORPUS -> model abstains
09_semantic_vs_keyword.txt                 retrieval: dense vs lexical contrast
10_clarified_doac_elderly.txt              broad topic narrowed via clarifying
Qs
11_adversarial_ignore_sources.txt          user jailbreak attempt -> refused
12_prompt_injection.txt                    poisoned source -> injection ignored
13_full_text_vs_abstract.txt               full-text vs abstract-only contrast

Safety behaviors demonstrated: grounding, per-claim PMID + verbatim quote,
abstention (#08), jailbreak refusal (#11), and prompt-injection resistance
(#12). #12 also shows a limit: the model neutralizes the injected command
but still cites the poisoned source -- corpus trust still matters.
