# Content-validity rating PILOT — drawn 2026-10-06

Per `ContentValidityRatings_edited.docx` (analysis repo root, owner-approved
plan): 40-ish narrative pilot BEFORE the final draw, to measure CJ rate and
agreement, settle decision rules 2 (individual non-compliance: 2 vs 3) and 7
(disputes/boilerplate: CJ vs unfavourable), and tune anchors.

- **PILOT seed = 11** (sampler `dynclim/experiments/sample_validity_ratings.py`).
  The FINAL-draw seed must be different, recorded in the protocol before
  drawing, and the final draw must exclude this pilot's eids
  (`rating_key_seed11.csv` column eid → pass via --exclude-eids).
- n = 42 (10 per industry; msha 12 from cell-balance rounding), stratified
  industry × indicator-score tercile × structured severity, exact-duplicate
  boilerplate collapsed, <5-word strings excluded, --llm-dev 0 (no LLM split
  at pilot stage).
- Protocol illustration narratives excluded via
  `validity_study/protocol_example_eids.txt` (7 eids — the uniquely
  identifiable examples; formulaic rail phrases shared by 45-579 reports were
  NOT eid-excluded since boilerplate is collapsed and the phrase is the
  population, not the example; two aviation illustrations are abridged
  paraphrases with no verbatim match — collision risk in a 10-draw ~nil).
- `rating_sheet_seed11.xlsx` / `.csv`: blinded (rating_id, industry, text,
  blank rating_1to5 / cant_judge / notes), blocked by industry, shuffled
  within block. **Raters must not open `rating_key_seed11.csv`** (id → eid,
  consensus + reference-config scores, strata).

Next: both raters rate all 42 → reliability (weighted kappa/ICC, CJ
agreement, CJ prevalence by industry) → settle rules 2 & 7 + anchor edits →
record final seed in protocol → final draw (75/industry, or 150 for
msha+rail pending the pilot CJ rate).
