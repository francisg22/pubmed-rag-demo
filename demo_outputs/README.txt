PubMed RAG demo — example outputs
=================================

379 articles
     379 embedded with openai:text-embedding-3-small:1536 (years 2003-2026)

Corpus: ~379 PubMed abstracts across 7 clinical topics (metformin/CKD,
statins, atrial fibrillation/anticoagulation, HFpEF, SGLT2 inhibitors,
hypertension, COPD). Embeddings: OpenAI text-embedding-3-small (1536-d).
Answers: gpt-4o-mini, grounded only in retrieved abstracts, PMID-cited.

Files
-----
01_metformin_ckd.txt                  metformin safety in moderate CKD
02_sglt2_heart_failure.txt            SGLT2 inhibitors & HF hospitalization
03_statins_primary_prevention_elderly.txt  statins for primary prevention >75
04_doac_vs_warfarin_af.txt            DOACs vs warfarin in atrial fibrillation
05_hfpef_treatment.txt                treatments that improve HFpEF outcomes
06_copd_exacerbation.txt              managing an acute COPD exacerbation
07_hypertension_bp_target.txt         recommended BP target in hypertension
08_abstention_pneumonia.txt           OUT-OF-CORPUS: model correctly abstains
09_semantic_vs_keyword.txt            retrieval: dense vs lexical contrast

Note: every grounded answer ends with 'Draft for clinician review' and
cites PMIDs; #08 shows the system declining when the corpus lacks support.
