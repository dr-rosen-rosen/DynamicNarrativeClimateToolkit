# Stage-1 CJ screen — final draw (seed 20261008, drawn 2026-10-08)

1,400 narratives: mining 800, rail 400, aviation 100, nuclear 100. Excluded
the 7 protocol-example eids and the 42 pilot eids. Mining and rail drawn with
length weights (60% longest quartile / 25% third / 15% shorter half) within
score-tercile x severity cells; aviation and nuclear unweighted.

Files
- `screen_sheet_reviewerA_seed20261008.{csv,xlsx}` — 840 narratives (560 own + 280 overlap)
- `screen_sheet_reviewerB_seed20261008.{csv,xlsx}` — 840 narratives (560 own + 280 overlap)
- `screen_key_seed20261008.csv` — rating_id -> eid, reviewer (A / B / both), strata,
  length stratum, inclusion weight, consensus score, main-text champion scores.
  KEEP AWAY FROM REVIEWERS until stage 2 is complete.

Reviewer instructions (stage 1 only)
- For each narrative mark `cant_judge_YN` = Y if the narrative conveys no
  information about safety practice (mechanism of injury or failure only);
  N if there is anything to rate. Apply the protocol's decision rules 1 and 8:
  no inference from event type or severity; mandatory reporting language is
  not practice information.
- Notes are welcome on borderline items; they feed the consensus pass.
- Blocked by industry, shuffled within. ~10 seconds per decision is the
  expected pace; 840 items ~ 2.5 hours.

After return: CJ reliability on the 280 overlap items, prevalence by industry
(weighted by `incl_weight`), top-up draw for mining if rateable yield is thin,
then stage-2 rating sheets for both raters on all non-CJ narratives.
