# Narrative rating protocol — construct validity study (DRAFT v0.1 for team discussion, 2026-10-06)

**Revision 2026-11-06:** the analysis plan (§5) was rewritten after the stage-2 results were seen. The study is now framed as exploratory, with a development phase (pilot, stage 1, stage 2), a dated freeze of the rating guide, and a test phase on fresh narratives. §1, §6 and §8 carry matching notes. The original §5 is kept as §5.6.

**Second revision 2026-11-06 (after the first review meeting, under deadline):** the 319 are not re-rated in full under guide v1.1 and no post-consensus validity estimate is computed on them (§5.2). Human validity under the final rules comes from the held-out sample (§5.5). The held-out sample is at least 60 per industry and the reading rule is pooled (§5.4). A 15-narrative aviation check precedes the freeze.

Companion to `dynclim/experiments/sample_validity_ratings.py`. Responds to PNAS Nexus R1.1 (does the indicator capture evaluations of organisational safety practice rather than event severity?) and R2 (what is being measured; LLM-based coding as a measurement path).

## 1. What the study has to deliver

Three claims for the revised manuscript, from one sample and one rubric:

1. **Worked examples** — narratives with their segments, matched survey items, sentiment, and the roll-up to a report score (R2: "connect the measurements to example narratives").
2. **Validity against human judgment** — blind ratings of how favourably a narrative portrays organisational safety practice correlate with the indicator, and the association survives conditioning on structured event severity (mining, rail; coarse for nuclear; unavailable for aviation).
3. **LLM feasibility pilot** — a frontier model applying the same rubric agrees with human raters about as well as they agree with each other, and the validity result replicates on thousands of narratives, including at the organisation-period level where the indicator is actually used.

**Revised 2026-11-06.** Claims 2 and 3 are now exploratory and change as follows (§5):

- Claim 2 rests on the 319 stage-2 narratives as a development set. It reports the stage-2 reliability as returned and the pre-consensus validity estimate. It makes no reliability claim for the final rules.
- Claim 3 carries the reliability claim. Both raters and the model code a fresh held-out sample under the frozen guide, giving human-to-human and human-to-model agreement out of sample, and a blind human validity estimate under the final rules. The model then supplies the document-level and organisation-period validity analysis at scale.

## 2. Unit, blinding, raters

- **Unit:** the whole narrative (one report). Raters see the narrative text and its industry, nothing else.
- **Raters:** two, blind to indicator scores and to the sampling strata. Both rate every narrative (or a reliability subset of ≥100 if time is short).
- **Order:** narratives are blocked by industry (register differs sharply) and shuffled within block.

## 3. The rating

One question per narrative:

> **Based only on what this narrative conveys, how favourably does it portray the organisation's safety practice?**
> Safety practice here means the state of safety as it is practised in that workplace: hazard recognition and control, procedures and whether they are followed, supervision, training and competence, communication and coordination, equipment and housekeeping, and the organisation's response to events (reporting, investigation, correction).

Scale:

| Rating | Anchor |
|---|---|
| **1 — Strongly unfavourable** | Practice portrayed as clearly deficient: procedures absent or ignored, known hazards left uncontrolled, supervision or training plainly lacking, dismissive or concealing response. |
| **2 — Somewhat unfavourable** | A lapse or deficiency is indicated (inadequate precaution, missing control, non-compliance) without mitigation or response. |
| **3 — Mixed / neutral** | Practice-relevant content is present but balanced or ambiguous: a lapse paired with an adequate response; controls present but not used; competing cues. |
| **4 — Somewhat favourable** | Practice shown working: hazard recognised and controlled, procedure followed, prompt reporting or response, corrective steps taken. |
| **5 — Strongly favourable** | Proactive, conservative practice: stop-work or conservative decisions, thorough investigation, learning actions, explicit management commitment. |
| **CJ — Can't judge** | The narrative conveys no information about safety practice (mechanism of injury or failure only). |

**CJ is expected to be common in mining and rail.** Random draws suggest 6–8 of 10 narratives there describe the event mechanism only ("He was cutting a flap off a piece of belt when the knife blade broke, cutting his wrist"). CJ is the correct answer for these; do not infer practice from severity or from the mere fact that an event occurred.

### Decision rules

**Superseded 2026-11-06.** The rules below are those in force through stage 2 (guide v1.0). The current rules, with the decisions from the 2026-11-06 review meeting, are in `validity_study/stage2_consolidated_2026-10-26/RATER_GUIDE.md` (v1.1, final draft for freeze, rewritten 2026-11-09 as a five-step procedure; decision record in `RATER_GUIDE_CHANGELOG.md`). This list will be replaced with the frozen text when the guide is frozen (§5.3).

1. **Rate practice, not severity, and only what is stated** *(sharpened from pilot, 2026-10-08)*. Rate only what the narrative states or carries by an explicit attribution ("cause was…", "failed to…"). An event mechanism that could reflect a lapse but has no stated cause — wide gage, a shove over a switch, even a fatal head-on collision — is CJ, however severe. Severity comes from structured data.
2. **Frontline compliance counts.** Individual non-compliance (not wearing provided PPE, not following a procedure) is part of practice: rate 2, or 3 if the organisation's provision or response is also described.
3. **Response actions count.** Prompt reporting, stop-work, investigation, corrective action, retraining, or conservative reporting move the rating toward 4–5 even when the triggering event was a lapse — subject to rule 8.
4. **Equipment or material failure alone is CJ** unless the narrative speaks to inspection, maintenance, or the response.
5. **Attributions are evidence.** "Cause was inadequate precautions" or "lack of awareness" are unfavourable; "immediately reported", "documented in the corrective action program" are favourable.
6. **Multiple cues:** weigh the overall portrayal; use 3 for genuinely balanced narratives rather than averaging mechanically.
7. **Disputes and boilerplate** ("submits this report solely to terminate the citation") — treat as CJ unless practice is described. *Open for discussion.*
8. **Mandatory reporting is baseline, not favourable** *(new from pilot, 2026-10-08)*. Notifying the NRC inspector, filing an Event Notification, or an FRA/MSHA report is the required act and does not raise the rating. A favourable response needs discretionary action beyond the requirement: conservative reporting, corrective-action-program entry, stop-work, investigation findings, retraining.
8a. **Required operational responses are baseline** *(adopted 2026-10-26)*. Actions the technical specifications or procedures require — declaring a system inoperable, entering an LCO/TS action statement, posting a required fire watch or other compensatory measure, tripping the reactor and stabilising the plant — are what any licensee must do and are CJ on their own. Rateable only when the narrative adds discretionary or evaluative content: a conservative action beyond the requirement, a stated cause that implicates practice, or corrective action beyond the immediate.
8b. **Equipment or instrument failure is CJ unless practice is stated** *(adopted 2026-10-26)*. Rateable when the narrative states a human or organisational cause (tagging/configuration-control error, test performance, work planning) or describes maintenance or inspection practice. Recurrence counts only when the narrative connects it — "despite the earlier corrective action", "the cause of the previous event had not been identified", "the repair did not resolve" — not when it merely notes the same thing happened before *(adopted 2026-10-26; keeps 8b consistent with rule 1)*.
8d. **(Rail, PROPOSED)** FRA-mandated post-accident toxicological testing ("crew was tested", "all tests were negative") is baseline, like a notification.
9. **(PROPOSED) Third-party conduct is not the organisation's practice.** Motorists at crossings, trespassers, shippers, other railroads' crews: CJ for the reporting organisation unless its own controls or response are evaluated — devices "activated and functioning" is baseline; devices "not operating" is unfavourable; a crew's prompt emergency application on sighting the hazard is a response under rule 3. Worked through against all 19 rail and 14 nuclear screen discords in `validity_study/stage2_consolidated_2026-10-26/DISCUSSION_rail_nuclear.md`; rater-facing summary in `RATER_GUIDE.md` there.
8c. **Report genres with no practice content are CJ** *(adopted 2026-10-26)*. Non-work-related events on site, Part 21 supplier-defect notifications, routine offsite spill/environmental notifications, and scheduled-outage notifications are CJ unless the narrative evaluates practice.

### Industry illustrations (abridged real narratives; final examples are excluded from the rating sample via `--exclude-eids`)

**Mining**
- 4: "A roof fall occurred in the #4 entry. The roof had been supported with resin bolts and cable bolts. Ventilation was not affected; no personnel or equipment affected. Immediately reported." — controls described, prompt reporting.
- 3: "Draw rock fell and struck his right forearm. He was not wearing the Kevlar arm sleeves provided." — PPE provided (favourable) but not worn (unfavourable). *Discussion point: 2 or 3?*
- 2: "Employee slipped and fell checking a sensor. Lack of awareness was the cause of the accident. PPE was not a factor." — attribution to a lapse, no control or response described.
- CJ: "Employee was climbing out of the loader, stepped on a frozen clod of mud, twisting his left ankle."

**Rail**
- 1–2: "Crew derailed two engines and six cars because they failed to comply with restricted speed, resulting in a rear-end collision." — plain non-compliance.
- 2: "Railcars rolled out of the yard track and impacted another cut. Cause was inadequate precautions while kicking railcars."
- 3: "Heavy dynamic braking caused a run-in. Action to prevent: engineer has been counseled on use of dynamic brake." — lapse plus response.
- 4: "Crew noticed sparks and smoke so they immediately stopped the train. A thermite weld had broken in the curve."
- CJ: "Pulling off #9 and derailed 8th to 11th cars account broken rail."

**Nuclear** (reactor events only; narratives are long and usually describe the licensee's response)
- 4: "The licensee is conservatively making this report although no specific impacts are indicated. The individual's access has been terminated; the event is documented in the corrective action program; the resident inspector has been notified."
- 2 (paraphrased pattern): "The condition had existed undetected since the last outage; the procedure did not require the check; operators were unaware of the configuration." — latent deficiency, no mitigation described.

**Aviation** (ATC/ASRS; first-person, evaluative)
- 1: "Primary radar has been out of service for weeks. ATIS was not advertising the outage, which makes me wonder if it ever did." — uncontrolled known hazard, communication failure.
- 2: "Due to extreme noise from maintenance in the facility and the lack of a radar associate, the readback was missed and separation was lost."
- 1–2: "The city is supposed to call before ship traffic suspends Runway 35 arrivals. This is rarely done correctly and is left to the local controller — contrary to the agreed order."
- 4 (pattern): conflict recognised early, coordinated with the adjacent facility, corrective action issued before separation was lost.

## 4. Sampling (implemented in `sample_validity_ratings.py`)

- Scope matches the paper's analyses: mining = coal; nuclear = power-reactor events; rail and aviation = all.
- Stratification: industry × indicator-score tercile × structured severity (mining: no-injury / low / mid / high; rail: damage-only / injury / fatal). Equal cells, random top-up to the per-industry target.
- The stratification score is a **consensus** across the 81 weighted-average/attention configurations (mean within-config z-score), so no single specification is privileged; a reference configuration's raw score is carried in the key.
- Exact-duplicate boilerplate collapsed; trivial strings (<5 words) excluded.
- Default 75 per industry (300). **Recommend 150 for mining and rail** given the expected CJ rate (unrateable narratives take ~10 seconds).
- The final seed is recorded here before the draw: `seed = ______` (dry run used seed 0; do not reuse).

## 5. Analysis plan

**Revised 2026-11-06, after the stage-2 results were seen.** Stage-2 agreement on the 1–5 rating was acceptable only in mining (quadratic weighted kappa 0.64; nuclear 0.31, rail 0.24, aviation 0.06), and agreement on CJ was poor (pooled κ 0.19). The disagreements traced to rules that were missing or applied unevenly (`validity_study/stage2_consolidated_2026-10-26/STAGE2_RESULTS_2026-11-05.md` and `DISCUSSION_stage2_rules.md` there). The study is therefore exploratory and runs in two phases separated by a dated freeze of the rating guide. The plan written before any data were seen is kept in §5.6.

### 5.1 Phases and what each can support

| Phase | Material | What it supports |
|---|---|---|
| Development | Pilot (42), stage-1 screens, stage 2 (319) | A rating guide that can be applied consistently; exploratory pre-consensus validity; labelled examples for checking the model prompt |
| Freeze | Guide v1.1, model and prompt, dated here | The boundary: nothing before it counts as a test of the rules |
| Test | Fresh held-out narratives, coded by both raters and the model | Reliability of the frozen rules (human-to-human, human-to-model); validity at scale from the model |

The 319 cannot test the final rules, because the rules were written from the items the raters disagreed on. Agreement on those items after consensus is reported as a description of the consensus set and never as reliability.

### 5.2 Development phase (the 319)

1. **Stage-2 reliability is reported as returned**: CJ agreement and κ; exact and within-one agreement, quadratic weighted kappa, ICC and Spearman on narratives both raters rated; by industry and pooled, with bootstrap intervals. It is presented as the result that motivated the rule changes and is not replaced by any later figure.
2. **Pre-consensus validity is the primary development-phase estimate.** Spearman correlation of the mean of the two independent ratings with the consensus score, within industry, with intervals; repeated for the champion configurations; controlling structured severity in mining and rail (unavailable in aviation, length only in nuclear). These numbers were computed before any rule changed.
3. **Consensus on the discordant narratives only** *(revised 2026-11-06, second revision)*. The thirteen decisions (D1–D13) were recorded at the 2026-11-06 meeting and written into guide v1.1 (draft). The 128 discordant narratives are settled by those decisions or, for the 37 they do not settle, at a second meeting (`stage2_review_v1.1.xlsx`). The 319 are **not** re-rated in full under v1.1: narratives outside the worklist keep their v1.0 ratings. Original ratings are kept unchanged in the merged file.
4. **No post-consensus validity estimate is computed on the 319.** The set would mix two versions of the rules, and one rater saw the pre-consensus validity results before consensus. The human validity estimate under the final rules comes from the held-out sample (§5.5 item 3), where both raters are blind.
5. **Labelled set for checking the model prompt** (§5.3): the 128 settled calls, plus the agreed narratives that no changed rule touches (about 91: the 191 outside the worklist less the 64 aviation narratives and 36 that match a changed rule by keyword). CJ prevalence is reported in one sentence per industry, with CJ treated as missing.
6. **Aviation check before the freeze.** Aviation agreement was at chance in stage 2 and its rule changed most. Once the aviation questions are settled, each rater independently rates 15 aviation narratives drawn at random (seed 20261109) from the 64 outside the worklist, and the two compare. If the ratings split in opposite directions again, the rule is revised before the freeze. This is a check and is not reported as a result.

### 5.3 Freeze

Recorded here before the held-out sample is drawn and before the model codes any narrative outside the development set:

- guide v1.1 frozen 2026-11-09, `validity_study/stage2_consolidated_2026-10-26/RATER_GUIDE.md`, sha256 `c24d9b0ba0187ddf4d300ea2e7d18a322b43fc2c3597d259804d6d30b640a1b1`
- model `claude-opus-5-5`, adaptive thinking, `effort: high`, no sampling parameters, no fallbacks (recorded 2026-11-09; pipeline in `validity_study/LLM_PIPELINE.md`); prompt sha256 `9be9c2caac6a784e6633d40062a6a95c45137dab4f7de5b3b672923d597b509c` (frozen 2026-11-09 after the 128-item check: pooled weighted κ 0.94 with the consensus calls, CJ κ 0.84; `validity_study/llm/CHECK_128_RESULTS.md`)
- pools drawn 2026-11-09 with seed 20261110 (mining 6,000, rail 4,000, nuclear 1,500, aviation 800; simple random from the analysis scope, all 2,149 prior eids excluded) and coded in batch mode the same day, no errors; held-out draw seed 20261111, 75 per industry, `validity_study/heldout_2026-11-09/`

Rules:

- The model prompt is the frozen guide text, the rating question, and an output format. Prompt development uses only the pilot and the 319. Model agreement with consensus on those items is in-sample and is reported as such.
- One pinned model version with fixed settings (temperature 0 where the model exposes it). Three repeat runs on the held-out sample give run-to-run agreement.
- No rule changes after the freeze. Problems found during the test phase are logged for a later version and are not applied to test-phase ratings.
- The manuscript states that the rule proposals were drafted with model assistance from the discordant items and then decided by the raters.

### 5.4 Test phase: reliability under the frozen rules

1. **Pool.** A random sample per industry from the paper's analysis scope (§4), excluding all 2,149 narratives drawn so far (three stage-1 draws, the pilot, the protocol examples). The model codes the whole pool as CJ or 1–5. Pool size is set by how rare rateable narratives are (weighted stage-1 prevalence: mining 4%, rail 7%, nuclear about a third, aviation 82%); see open decision 11.
2. **Held-out human sample.** Drawn from the pool stratified on the model's call: CJ, 1–2, 3, and 4–5. 75 per industry (300 in all; owner, 2026-11-09). Equal strata (about 19 each), with short strata filled from the others; in mining the model called only 10 pool narratives 3, so that stratum has 10 and the others 21–22. Inclusion weights are kept so that results can be reweighted to the pool.
3. **Why stratify on the model's call.** Hand-screening found 4–7% of mining and rail narratives rateable, so a random held-out sample would again be nearly all CJ. Stratifying on the model's call fixes that without the circularity rejected on 2026-10-08 (§7), because the raters also code narratives the model called CJ. Model errors in both directions are observable.
4. **Rating.** Both raters code every held-out narrative under guide v1.1, blind to the model's call, the strata, the indicator and each other. No discussion before the agreement statistics are computed. At about 1.5 minutes per narrative, 240 narratives is at most six hours per rater.
5. **Human-to-human agreement.** The statistics of §5.2 item 1, by industry and pooled, unweighted (as designed) and inclusion-weighted (as in the pool). **The pooled figure is the reliability figure for the frozen rules.** At 60 per industry about 120 narratives are expected to be rated by both raters (using the stage-2 both-rated shares), which gives a pooled interval on weighted kappa of roughly ±0.10 to ±0.15. By-industry figures rest on about 20 to 40 narratives each and are reported descriptively.
6. **Human-to-model agreement.** The same statistics for the model against each rater and against the mean of the two, set beside the human-to-human figures. The benchmark from §1 is unchanged: the model agrees with the raters about as well as they agree with each other.
7. **Reading rule (fixed 2026-11-10, before either held-out sheet was returned or compared).** Model-coded results are interpreted only if pooled held-out human-to-human agreement is adequate: quadratic weighted kappa of at least 0.60 on narratives both raters rated, pooled across industries, unweighted (as designed). Agreement on the cannot-judge decision is reported beside it and is not part of the rule. Otherwise they are reported descriptively, without a validity interpretation. The sample is too small to apply this industry by industry; an industry where agreement is visibly poor is flagged in the text.
   **Outcome (2026-11-20, both sheets complete, analysed once).** Pooled weighted kappa 0.53 (95% interval 0.40 to 0.66; 118 rated by both); cannot-judge kappa 0.37. The rule is not met: model-coded results are reported descriptively. By industry: mining 0.89, nuclear 0.77, rail 0.55, aviation 0.22 (flagged). Tables: `validity_study/heldout_2026-11-09/HELDOUT_RESULTS.md`.

### 5.5 Validity at scale (exploratory)

1. **Document level.** Model rating against the consensus score and the champion configurations, within industry, on all pool narratives the model rated; controlling structured severity in mining and rail.
2. **Organisation-period level.** Mean model rating per organisation-period against the indicator for the same unit. The minimum number of rated narratives per unit is fixed before the analysis: `minimum = ______`.
3. **Held-out human ratings against the indicator**, inclusion-weighted. This is the human validity estimate under the final rules, and it is blind: neither rater has seen the indicator for these narratives. Mean of the two ratings on narratives both rated, against the consensus score; pooled on within-industry ranks as the primary figure, by industry descriptively. Small samples.
4. All of §5.5 is labelled exploratory in the manuscript.

### 5.6 Original plan (2026-10-06, written before any data were seen)

1. Inter-rater reliability: weighted kappa / ICC on the 1–5 ratings; agreement on CJ.
2. Validity: Spearman correlation between mean rating and the consensus score; repeated for the reference configuration. Reported pooled and by industry, with CIs.
3. Severity-conditioned validity: partial correlation (or rank regression) controlling structured severity in mining and rail; coarse in nuclear; stated as unavailable for aviation.
4. CJ prevalence reported in one sentence per industry; CJ treated as missing.
5. LLM: same question as a frozen prompt (dev set = 12 per industry; test = remainder). Agreement with each rater vs. rater–rater agreement. Then ~2,000–5,000 narratives per industry for document-level and organisation-period validity. Single frontier model, pinned version, temperature 0, reported as exploratory.

## 6. Expectations to state up front

- Document-level correlations will be **modest** — the indicator is designed to be aggregated over many reports. The organisation-period result from the LLM expansion is the level that matters.
- High CJ in mining/rail is a finding about the corpora, not a failure of the study.
- *(Added 2026-11-06.)* The pilot agreement (weighted kappa 0.84 on 14 narratives) did not hold at stage 2. Reliability of the final rules is unknown until the held-out sample is coded (§5.4), and may remain poor in rail and nuclear, where the CJ boundary has been unstable in every round.

## 7. Pilot results (seed 11, 42 narratives, two raters; 2026-10-08)

| Result | Value |
|---|---|
| Both rated | n = 14: within one point 100%, exact 50%, quadratic weighted kappa 0.84, Spearman 0.66; neither rater used 3 |
| CJ agreement | 32 of 42 (18 both CJ, 14 both rated); 10 discordant |
| CJ rate by industry (MR / MK) | aviation 0% / 0%; nuclear 70% / 10%; rail 80% / 60%; mining 100% / 100% |
| Discordant pattern A (6 of 10) | Nuclear notifications rated 4 by one rater for notifying the inspector, CJ by the other — resolved by rule 8 |
| Discordant pattern B (4 of 10) | Rail events where a lapse was inferred from the event type without a stated cause — resolved by sharpened rule 1 (all four become CJ) |

**Mining was the design problem — resolved by human screening, not an LLM screen (decided 2026-10-08).** With CJ on 12 of 12, a random draw of 75 or 150 yields no rateable mining narratives. An LLM content screen was rejected as circular (the human sample would be conditional on the LLM's notion of content, and the LLM then validated on that sample). Instead: a two-stage human workflow. Stage 1 — all four industries screened for CJ by two reviewers, evenly split, 20% overlap for CJ reliability; mining and rail oversampled with length weights (60% longest quartile / 25% third / 15% shorter half — practice cues run 3.5% in long vs 0.2% in short mining narratives; length is indicator-independent and inclusion weights are kept for population prevalence). Stage 2 — both raters rate every survivor. The LLM is evaluated afterwards against the human-coded set on both the CJ decision and the rating.

**Pilot association with the indicator** (14 both-rated): none detectable (Spearman +0.05 vs consensus, +0.40 vs one reference config) — 9 of 10 aviation narratives were rated 1–2 while the indicator spread them across terciles. Hence: raters use the full scale *within* industry (1 = clearly worst seen in that industry), and the validity analysis is within industry. The key now carries every main-text champion configuration's raw score (not config 37) alongside the consensus.

**Final stage-1 draw: seed 20261008 (2026-10-08)** → `validity_study/stage1_2026-10-08/`: mining 800, rail 400, aviation 100, nuclear 100 (1,400); reviewers A/B 840 each (560 own + 280 overlap); 7 protocol-example and 42 pilot eids excluded. Merged pilot ratings: `validity_study/pilot_2026-10-06/pilot_ratings_merged.csv`.

**Stage-1 screen results (returned 2026-10-19).** 1,399 of 1,400 marked. Overlap (n = 280): CJ agreement 88.6%, Cohen's kappa 0.58; reviewer B more inclusive (28 B-rateable/A-CJ vs 4 reverse). By industry: aviation 95%, mining 92%, rail 88%, **nuclear 55%** (9 of 20 discordant — the rule-8 boundary still read differently; consensus pass needed).

| Industry | Screened | Survivors (rateable by ≥1) | Discordant overlap | Raw yield | Population prevalence (weighted) |
|---|---|---|---|---|---|
| Mining | 800 | 54 | 12 | 6.8% | 4.1% |
| Rail | 400 | 50 | 10 | 12.5% | 6.7% |
| Aviation | 99 | 81 | 1 | 82% | 82% |
| Nuclear | 100 | 36 | 9 | 36% | 36% |

Length weighting worked (42/54 mining and 44/50 rail survivors from the longest quartile). Stage 2 proceeds on all 221 survivors; discordant screens are included and flagged, and stage-2 raters keep a CJ option so their consensus is the final CJ call. **Top-up drawn 2026-10-19 (seed 20261019)** → `validity_study/stage1_topup_2026-10-19/`: +400 mining and +200 rail from the longest quartile only, all 1,449 prior eids excluded; reviewers A/B 360 each (240 own + 120 overlap); expected ≈ +30 / +25 survivors for ~1 hour of screening per reviewer; survivors join stage 2 as a second sheet via `--stage2`.

**Top-up screen results (returned 2026-10-22).** 599 of 600 marked. Yield: mining 39 (9.8% of the longest quartile), rail 31 (15.6%); 12 discordant. Overlap (n = 120): agreement 90.0%, κ 0.45 — mining 96%, rail 78% (9 of 40 discordant; rule-1 inference boundary still reads differently in rail). **Pooled across both draws (n = 400 overlap): agreement 89.0%, κ 0.56.** Pooled survivors: mining 93, rail 81, aviation 81, nuclear 36 = **291** for stage 2. Second stage-2 sheet: `validity_study/stage2_topup_2026-10-22/rating_sheet_raterA|B_seed20261019.xlsx` (70 items). Merged top-up screens: `validity_study/stage1_topup_2026-10-19/screen_merged_seed20261019.csv`.

**Nuclear top-up drawn 2026-10-22 (seed 20261022)** → `validity_study/stage1_topup_nrc_2026-10-22/`: 100 reactor-event narratives, tercile-stratified, no length weighting, all 2,049 prior eids excluded; reviewers A/B 60 each (40 own + 20 overlap); expected ≈ +35 survivors; rule 8 reiterated in the README given the first draw's 55% nuclear overlap agreement.

**Nuclear top-up results (returned 2026-10-26).** 28 survivors of 100 (28%). Overlap (n = 20): agreement 75%, κ 0.00 — and the A/B gap **inverted** (A 40% rateable vs B 18% outside the overlap; first draw A 15% / B 45%): each reviewer over-corrected after the rules discussion. All five discords are A-rateable/B-CJ on notification-genre reports (loss of offsite communications capability, chemical-spill notifications, scheduled TSC ventilation maintenance, an automatic trip from RCP loss). Pooled nuclear screening reliability across both draws (n = 40 overlap): agreement 65%, κ 0.07 — similar counts rateable (A 9, B 11) but only 3 in common, i.e. chance-level agreement beyond base rates; rail pooled κ 0.22 (n = 120) is the other weak spot; all industries pooled (n = 420): 88.3%, κ 0.52. Nuclear sub-rules **adopted 2026-10-26** (recurrence clause pending): **8a** TS-/procedure-required responses (declaring inoperable, LCO entry, required fire watch, trip and stabilise) are baseline unless discretionary or evaluative content is added; **8b** equipment/instrument failure is CJ unless the narrative states a human or organisational cause (tagging, configuration control, test performance), a recurrence or missed detection, or maintenance/inspection practice; **genre clarification** non-work-related on-site events and Part 21 defect notifications are CJ unless practice is evaluated.

**Consolidated stage 2 (built 2026-10-26)** → `validity_study/stage2_consolidated_2026-10-26/`: ONE sheet per rater with all **319** survivors (mining 93, rail 81, aviation 81, nuclear 64; 49 screen-discordant items flagged in the key; ids carry a draw suffix where they collided across draws). The two earlier partial stage-2 sheets are retired under `validity_study/_superseded/`. Stage-2 sheets: `validity_study/stage2_2026-10-19/rating_sheet_raterA|B_seed20261008.xlsx` (identical content; 1–5 plus CJ; full scale within industry). Merged screens: `validity_study/stage1_2026-10-08/screen_merged_seed20261008.csv`.

## 8. Open decisions

1. Adopt sharpened rule 1 and new rule 8; re-adjudicate the 10 discordant pilot items by consensus.
2. Mining and rail: LLM content screen before the final draw, or bundle rating per organisation-period?
3. Per-industry targets after the screen decision.
4. Rule 2 (individual non-compliance): 2 or 3 when the organisation provided the control?
5. Rule 7 (disputes/boilerplate): CJ or unfavourable?
6. Which frontier model for the screen and the LLM pilot?
7. Worked-example figure: main text or SI?

Added 2026-11-06 with the revised plan (§5):

8. Closed 2026-11-09: guide v1.1 frozen (all decisions and the eight rewrite resolutions confirmed). The MSHA regulation behind the mining required-acts entry was not checked against the regulation text; noted as a limitation.
9. Held-out sample size. At least 60 per industry (owner, 2026-11-06); final number on 2026-11-09. For scale, the stage-2 interval on weighted kappa was about ±0.12 at 71 both-rated narratives and ±0.23 at 32.
10. What counts as adequate pooled held-out human agreement for the reading rule in §5.4 item 7, fixed before the held-out ratings are compared. **Closed 2026-11-10 (owner): pooled quadratic weighted kappa ≥ 0.60; see §5.4 item 7. `analyze_heldout.py --adequate-qwk 0.60`.**
11. Pool size per industry. Pools are simple random draws (not length-weighted), so the inclusion weights are the stratum ratios alone. Proposed: mining 6,000, rail 4,000, nuclear 1,500, aviation 800; confirm after the 128-item check gives tokens per request. The organisation-period analysis needs far more, possibly the full corpus.
12. ~~Who makes the consensus calls on the 319.~~ Closed 2026-11-06: no post-consensus validity estimate is computed, so the calls are made jointly.
13. Where the validity results sit in the manuscript: main text with an exploratory label, or supplement.
