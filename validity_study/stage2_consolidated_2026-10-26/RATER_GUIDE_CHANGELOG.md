# Rater guide: change log and decision record

Companion to `RATER_GUIDE.md` (v1.1). Raters and the model do not need this file.

Versions kept in this folder:

- `RATER_GUIDE_v1.0_2026-10-26.md`: the guide the stage-2 ratings were made under.
- `RATER_GUIDE_v1.1_annotated_2026-11-09.md`: the 2026-11-06 draft with the owner's meeting annotations (decisions, additions, Q1–Q6 responses).
- `RATER_GUIDE.md`: v1.1 rewritten as a procedure and **frozen 2026-11-09**. Read-only snapshot: `RATER_GUIDE_v1.1_frozen_2026-11-09.md`.
  SHA-256: `c24d9b0ba0187ddf4d300ea2e7d18a322b43fc2c3597d259804d6d30b640a1b1`

## Where the rules went

The rewrite replaces the numbered rules with five steps. Mapping from the v1.1 draft:

| Draft rule | Now |
|---|---|
| 1 only what is stated; crew action is an attribution | Step 1 (mechanism only), step 2 preamble, step 2 A |
| 2 frontline compliance | Step 2 A; step 3 (A only = 2) |
| 3, 3a, 3b responses; discipline; crew response | Step 2 B and its "not a response" list; step 3 |
| 4 equipment failure | Step 1 |
| 5 attributions are evidence | Step 2 preamble |
| 6 using 3, 1 and 2 | Step 3 table |
| 7 disputes, boilerplate, dismissiveness | Step 2 D; step 1 (form language) |
| 8, 8a–8f required acts, voluntary notifications, report timing | Required-acts list; step 2 B |
| 9, 9a, 9b third parties, contractors, aviation | Step 1; §1 definition of the organisation; industry notes |
| 10 tone; conditions; conduct | Step 4 (tone); 10b and 10c absorbed into steps 2 and 3 |
| 11 mining checklists | Step 1 (form language); step 2 preamble (hedging); industry notes |
| 12 fitness for duty | Step 2 A and B; required-acts list; industry notes |
| 13 potential risk; Q3 test | Step 2 C |

## Decisions from the 2026-11-06 meeting

| Decision | Outcome |
|---|---|
| D1 Nuclear notifications and required responses | Baseline, CJ |
| D2 Voluntary and courtesy notifications | 4 when the subject of the report; never 5 for reporting alone |
| D3 Fitness-for-duty reports | Rateable (proposal to make them CJ rejected); access restriction is required, not favourable; applies in all industries |
| D4 Conditions found by the licensee | Potential risk not yet an event = 4; with analysis or action = 5; actual failure in a mandatory report = CJ |
| D5 Aviation reports | Rate conditions and conduct; tone and self-criticism are evidence (draft rule that they are baseline rejected); generalised to all industries |
| D6 Who the organisation is in aviation | Reporter's operation; third parties CJ unless implications for ATC-controlled practice; pilot miscommunication CJ, controller miscommunication rateable |
| D7 Post-accident testing | Baseline |
| D8 Third parties in rail | Adopted; contractors count only when stated to be under the railroad's authority |
| D9 Discipline | Not a response; rate the lapse |
| D10 Crew response | Alone 4; with a stated organisational lapse 3 |
| D11 Mining checklists | "Not a factor" lines are form language (CJ if nothing else); named cause 2; hedged factor counts as certain; specific control in use 4 |
| D12 Required responses in mining | Baseline; report timing excluded in every industry |
| D13 Using 3; 1 against 2 | 3 needs lapse plus discretionary response or control that worked; 1 needs more than a single lapse |
| Rule 7 | Disputing an injury, minimising severity, or dismissiveness toward reporting = 2 |

## Outcomes of the open questions (owner, 2026-11-09)

| | Outcome |
|---|---|
| Q1 fitness for duty | Access restriction is required, not favourable; a discretionary response (review of past work) is. Numbers: see resolution 2 below. |
| Q2 rule-13 boundary | Accepted as proposed (first candidate on each item). |
| Q3 aviation hazard reports | Test accepted: did a safety event occur? No event, hazard with what should change = 4; event = rate conditions and conduct. |
| Q4 tone | Never rateable alone; moves the content rating by at most one point. |
| Q5 mining | Questioning the injury or minimising severity = 2. |
| Q6 two items | R0079 (top-up) 2; R0531 (top-up) 2. |

## Resolutions applied in the rewrite (2026-11-09), confirmed by the owner at the freeze (2026-11-09)

1. **Hazard test against the anchor for 1.** The Q3 test alone would make a known hazard left uncontrolled (the protocol's example of a 1: radar out for weeks) a 4. Added to step 2 C: if the narrative says the organisation already knew about the hazard and left it, that is A (more than a single lapse), not C.
2. **Fitness for duty, the numbers.** Failure alone or with required measures = 2; with a discretionary response = 3 (keeps the step 3 table consistent). No failure, alcohol surrendered, no concerns = 4. The annotated draft said "favourable" for the response case; 4 would break the lapse-plus-response = 3 rule.
3. **Pilot miscommunication against the recommendations clause.** The clarification wins: a pilot's lapse is CJ even when the reporter recommends a change to ATC practice. The snowplough narrative (R0283) is CJ.
4. **Hedged causes outside checklists.** "Most likely human error" is treated like a hedged checklist factor: 2.
5. **Anchor for 3.** "Controls present but not used" is dropped; PPE provided but not worn is 2.
6. **Counselling** is placed with retraining as a response (it is instruction, not punishment). The annotated draft left it unplaced.
7. **Report timing** is excluded in every industry (confirmed).
8. **Rule 8f and the regulation.** Still not checked against the regulation text; accepted on the meeting's understanding.

## Not re-rated

Narratives outside the 128-item worklist keep their v1.0 ratings (protocol §5.2, second revision). The final calls on the 128 are in `stage2_consensus_v1.1.csv`.

## Post-freeze log

Problems found after 2026-11-09 go here, dated, for a later version. They are not applied to the frozen guide or to any rating made under it.

- 2026-11-09, from the model check (`validity_study/llm/CHECK_128_RESULTS.md`):
  - **Step 1 contractor clause and fitness for duty.** The clause "contractors count as the organisation only when stated to be under the organisation's authority" was applied to contract employees who failed fitness-for-duty tests (R0049, R0614), giving CJ where the consensus has 2. Intended reading: anyone with site or plant access is the organisation for fitness-for-duty purposes; the clause is about work crews.
  - **"Retracted notifications" in the step 1 genre gate.** Makes a retracted potential-risk report CJ (R0092) where the consensus has 4. The decision D1 said only that a retraction is not evidence either way.
  - **Worked example against consensus.** The dynamic-braking example (lapse, then practices modified) is 3 in the guide; the consensus call on R0850, the same pattern, is 4. The guide is internally consistent (A and B = 3); the label is the odd one out.
  - **Breadth of B.** "An investigation with findings", a root-cause review under way, or a voluntary LER were read as discretionary responses in five nuclear and aviation narratives where the consensus treated them as expected (R0966, R0028, R0074, R0736, R0027). A later version could say that an investigation counts only when its findings are stated.
