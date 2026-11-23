# Model coding pipeline (protocol §5.3–5.5)

Built 2026-11-09. Two scripts, run by file path under `.llmVenv` (Python 3.12 with the
Anthropic SDK; the toolkit venv is Python 3.9 and cannot run the current SDK):

- `dynclim/experiments/llm_rating.py` — prompt, pool draw, rating (live or batch), check.
- `dynclim/experiments/draw_heldout_sample.py` — the stratified held-out draw and rater sheets.

Credentials: `ANTHROPIC_API_KEY` exported in the shell that runs the scripts. Nothing is stored in the repo.

## Fixed settings

| Setting | Value | Where recorded |
|---|---|---|
| Guide | `RATER_GUIDE.md` v1.1, sha256 `c24d9b0b…a1b1`; the script refuses to run if the file's hash differs | protocol §5.3, `RATER_GUIDE_CHANGELOG.md` |
| Prompt | a short frame + the guide verbatim + an output schema; sha256 printed by `prompt` and written into every output row | protocol §5.3 (after the 128-item check) |
| Model | `claude-opus-5-5`, pinned by id | protocol §5.3 |
| Thinking / effort | adaptive thinking (cannot be turned off on this model); `effort: high` | protocol §5.3 |
| Sampling | none (temperature is not settable on this model); run-to-run agreement from `--repeats 3` on the held-out sample | protocol §5.3 |
| Output | structured JSON: `cant_judge`, `rating` 1–5 or null, `content_present` (A/B/C/D), `tone_adjustment`, `basis` | guide §6 |
| Fallbacks | none: a refusal or error is recorded as an error row, never answered by another model | this file |
| Caching | the system prompt (guide) is cached across requests; cache reads are logged per row | this file |

## Steps

1. **Prompt check** — `llm_rating.py prompt` prints the system prompt and both hashes.
2. **Check on the settled stage-2 calls** (prompt development; in-sample):
   `llm_rating.py rate --input validity_study/stage2_consolidated_2026-10-26/llm_check_input_128.csv --out validity_study/llm/check_128.jsonl`
   then `llm_rating.py check --ratings validity_study/llm/check_128.jsonl`. Reports CJ agreement and
   kappa, exact and within-one agreement, weighted kappa and mean difference against `stage2_consensus_v1.1.csv`,
   by industry, plus mean tokens per request for the cost estimate. Only the frame text may change in
   response; the guide is frozen. Record the final prompt hash in protocol §5.3 before step 3.
3. **Pools** — one random draw per industry from the paper's analysis scope, excluding the 2,149 eids
   drawn before, duplicates and strings under five words:
   `llm_rating.py pool --industry msha --n <N> --seed 20261110 --out validity_study/llm/pool_msha.csv` (and rail, nrc, aviation; one seed, recorded).
   Proposed N, to confirm after the cost estimate from step 2: mining 6,000, rail 4,000, nuclear 1,500,
   aviation 800. Mining and rail need the large pools because about 4–7% of narratives are rateable and
   favourable ones are rarer still.
4. **Code the pools** in batch mode (half price, asynchronous):
   `llm_rating.py rate --input validity_study/llm/pool_msha.csv --out validity_study/llm/pool_ratings.jsonl --mode batch`
   for each pool (same `--out`), then `llm_rating.py collect --out validity_study/llm/pool_ratings.jsonl --wait`.
   Batches are tracked in `pool_ratings.batches.json`; `rate` and `collect` resume.
5. **Held-out draw** — `draw_heldout_sample.py --pool validity_study/llm/pool_*.csv --ratings validity_study/llm/pool_ratings.jsonl --n 75 --seed <seed> --out validity_study/heldout_<date>`.
   Strata on the model's call (CJ, 1–2, 3, 4–5), equal targets, shortfalls redistributed; inclusion weights
   in the key. Writes identical blind sheets for both raters and a key that stays with the owner.
6. **Model on the held-out sample** — `rate --input <heldout items> --repeats 3` (live mode is fine for 300).
7. **Second model** (robustness, different provider): same prompt, same held-out sample and pools, one
   supplementary table. Not built here; the prompt text is provider-neutral.

## Outputs

JSONL rows carry `item_id`, `repeat`, `model`, `effort`, `prompt_sha256`, `guide_sha256`, the parsed
rating fields, `call` (CJ or 1–5), `stop_reason`, `request_id`, token counts, or an `error` field.
Rows with an error are retried on the next `rate` run.
