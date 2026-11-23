# Stage-2 Rater Guide (v1.0, 2026-10-26)

Companion to `rating_sheet_rater{A,B}_consolidated.xlsx` (319 narratives). Full protocol: shared page "Narrative Rating Protocol" / `docs/validity_study/rating_protocol_draft.md`.

## The question

**Based only on what this narrative conveys, how favourably does it portray the organisation's safety practice?**

Safety practice = the state of safety as practised in that workplace: hazard recognition and control, procedures and whether they are followed, supervision, training and competence, communication and coordination, equipment and housekeeping, and the organisation's response to events (reporting, investigation, correction).

## The scale

| Rating | Anchor |
|---|---|
| 1 — Strongly unfavourable | Practice clearly deficient: procedures absent or ignored, known hazards uncontrolled, supervision or training plainly lacking, dismissive or concealing response. |
| 2 — Somewhat unfavourable | A lapse or deficiency is indicated (inadequate precaution, missing control, non-compliance) without mitigation or response. |
| 3 — Mixed / neutral | Practice-relevant content present but balanced: a lapse paired with an adequate response; controls present but not used; competing cues. |
| 4 — Somewhat favourable | Practice shown working: hazard recognised and controlled, procedure followed, prompt reporting or response, corrective steps taken. |
| 5 — Strongly favourable | Proactive, conservative practice: stop-work or conservative decisions, thorough investigation, learning actions, explicit management commitment. |
| CJ — Can't judge | No information about safety practice (mechanism of injury or failure only). |

**Use the full scale within each industry.** Aviation narratives are filed to flag problems and read unfavourable; nuclear notifications read as institutional responses. Within an industry, reserve 1 for the clearly worst narratives you see there and 5 for the clearly best. Analysis is within industry.

## Decision rules (all adopted unless marked)

1. **Rate practice, not severity, and only what is stated.** Rate only what the narrative states or carries by an explicit attribution ("cause was…", "failed to…", "was not wearing…", "operated before the route was lined"). An event mechanism that *could* reflect a lapse but has no stated cause — wide gage, a shove over a switch, a head-on collision "under investigation" — is CJ, however severe. **A stated crew action or omission that caused the event IS an attribution** and is rateable (usually 2).
2. **Frontline compliance counts.** Not wearing provided PPE, not following a procedure: rate 2, or 3 if the organisation's provision or response is also described. *(Team: 2 or 3 when the control was provided?)*
3. **Response actions count.** Prompt reporting, stop-work, placing the train in emergency on noticing a problem, investigation, corrective action ("practices have been modified to prevent recurrence"), retraining or conservative reporting move the rating toward 4–5 even when the trigger was a lapse — subject to rules 8 and 8a.
4. **Equipment or material failure alone is CJ** unless the narrative speaks to inspection, maintenance or the response — see 8b.
5. **Attributions are evidence.** "Inadequate precautions", "lack of awareness", "failed to comply" are unfavourable; "immediately reported", "documented in the corrective action program" are favourable.
6. **Multiple cues:** weigh the overall portrayal; use 3 for genuinely balanced narratives.
7. **Disputes and boilerplate** ("submits this report solely to terminate the citation"): CJ unless practice is described. *(Team: CJ or unfavourable?)*
8. **Mandatory reporting is baseline.** Notifying the inspector, filing an Event Notification or FRA/MSHA report is the required act and does not raise the rating. Favourable needs discretionary action beyond the requirement.
   - **8a. Required operational responses are baseline.** Declaring a system inoperable, entering an LCO/TS action, posting a required fire watch or compensatory measure, tripping and stabilising the plant — CJ on their own. Rateable only with added discretionary or evaluative content (conservative action beyond requirement, stated cause implicating practice, corrective action beyond the immediate).
   - **8b. Equipment or instrument failure is CJ unless practice is stated.** Rateable when a human or organisational cause is stated (tagging/configuration-control error, test performance, work planning) or maintenance/inspection practice is described. **Recurrence counts only when the narrative connects it** — "despite the earlier corrective action", "the cause of the previous event had not been identified", "the repair did not resolve" — not when it merely notes that the same thing happened before.
   - **8c. Report genres with no practice content are CJ.** Non-work-related events on site, Part 21 supplier-defect notifications, routine offsite spill/environmental notifications, scheduled-outage notifications — unless practice is evaluated.
   - **8d (rail, PROPOSED — discuss).** FRA-mandated post-accident toxicological testing ("crew was tested", "all tests were negative") is baseline, like a notification: not evidence either way.
9. **Third-party conduct is not the organisation's practice (PROPOSED — discuss).** Motorists at crossings, trespassers, shippers, other railroads' crews: their behaviour is CJ for the reporting organisation. Rateable only when the organisation's *own* controls or response are evaluated — crossing devices "were activated and functioning" is baseline (CJ); devices "were not operating" is unfavourable; a crew's prompt emergency application on sighting the hazard is a response (rule 3).

## Procedure

- Both raters rate every narrative; the sheets are identical. Work in industry blocks; several sittings are fine.
- Enter 1–5 in `rating_1to5`, or Y in `cant_judge` (not both). Notes on borderlines feed the consensus pass.
- 49 items were screen-discordant (flagged in the key, invisible to you); your two ratings decide them.
- Do not look up the structured data or the indicator scores while rating.
