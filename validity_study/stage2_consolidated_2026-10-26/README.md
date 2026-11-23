# Stage-2 rating — CONSOLIDATED (built 2026-10-26)

**Status 2026-11-09 (after the second review meeting).** All decisions are in.

- `RATER_GUIDE.md` is **v1.1, FROZEN 2026-11-09**: a five-step procedure
  for human raters and the model. Read-only snapshot
  `RATER_GUIDE_v1.1_frozen_2026-11-09.md`; SHA-256 in
  `RATER_GUIDE_CHANGELOG.md` (decision record, rule mapping, post-freeze
  log). Earlier versions: `RATER_GUIDE_v1.0_2026-10-26.md` (stage-2
  ratings were made under it) and `RATER_GUIDE_v1.1_annotated_2026-11-09.md`
  (the owner's meeting annotations).
- `stage2_consensus_v1.1.csv` has the final call on each of the 128
  discordant narratives (`apply_q_outcomes.py`, from
  `stage2_review_v1.1.csv`); seven are marked as judgment calls. Narratives
  outside the worklist keep their v1.0 ratings and are not re-rated.
- `stage2_review_v1.1.xlsx` is the meeting workbook, kept for the record.
- `aviation_check_2026-11-09/` is the optional 15-narrative aviation check
  (see below); not a reported result.
- `llm_check_input_128.csv` is the input for the model's check against
  `stage2_consensus_v1.1.csv` (`validity_study/LLM_PIPELINE.md`, step 2). Result 2026-11-09: weighted κ 0.94, CJ κ 0.84
  (`validity_study/llm/CHECK_128_RESULTS.md`).

These 319 narratives are the development set for the guide; the frozen
guide is tested on a fresh sample
(`docs/validity_study/rating_protocol_draft.md` §5). The instructions below
are the ones the stage-2 ratings were made under.

ONE sheet per rater containing every narrative that survived any stage-1
screen (rateable per at least one screener): 319 narratives.

| Industry | First draw (seed 20261008) | Mining/rail top-up (20261019) | Nuclear top-up (20261022) | Total |
|---|---|---|---|---|
| Mining | 54 | 39 | — | 93 |
| Rail | 50 | 31 | — | 81 |
| Aviation | 81 | — | — | 81 |
| Nuclear | 36 | — | 28 | 64 |

Supersedes `_superseded/stage2_2026-10-19` (221) and `_superseded/stage2_topup_2026-10-22`
(70) — do NOT rate those. Rating ids from different draws collided, so ids
here carry a draw suffix (e.g. `R0123_20261019`); the key maps them back.

Files
- `rating_sheet_raterA_consolidated.{csv,xlsx}` and `rating_sheet_raterB_consolidated.{csv,xlsx}` —
  identical content, blocked by industry, shuffled within: rating_id, industry,
  text, `rating_1to5`, `cant_judge`, `notes`.
- `rating_key_consolidated.csv` — eid, draw, strata, screen outcomes (incl. 49
  screen-discordant items flagged), consensus + champion-config scores.
  KEEP FROM RATERS until rating is complete.

Instructions
- One rating per narrative: 1 (strongly unfavourable) to 5 (strongly
  favourable) on how favourably it portrays the organisation's safety
  practice. CJ remains available (`cant_judge` = Y); for the 49 items the
  screeners disagreed on, your consensus is the final CJ call.
- Use the full scale WITHIN each industry: reserve 1 for the clearly worst
  narratives you see in that industry and 5 for the clearly best.
- Decision rules per the protocol: 1 (only what is stated or attributed),
  8 (mandatory reporting is baseline), and the nuclear sub-rules ADOPTED
  2026-10-26: 8a (TS-/procedure-required responses are baseline), 8b
  (equipment/instrument failure is CJ unless a human/organisational cause
  or maintenance practice is stated; recurrence clause pending), 8c (report
  genres with no practice content — non-work on-site events, Part 21,
  routine spill/environmental and scheduled-outage notifications — are CJ).
  The screen was deliberately inclusive on these borderlines, so expect to
  CJ a fair share of the nuclear block.
- Expected effort: ~1.5 min per narrative → ~8 hours per rater; split across
  sittings by industry block.
