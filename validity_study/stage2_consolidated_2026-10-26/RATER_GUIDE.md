# Narrative Rating Guide, version 1.1

**Status:** FROZEN 2026-11-09. Do not edit. Problems found after this date are logged in `RATER_GUIDE_CHANGELOG.md` for a later version and are not applied. The SHA-256 of this file is recorded there and in the protocol (§5.3).

This guide is used by two human raters and by a language model, with the same text. Everything you need is here. The history of the rules (`RATER_GUIDE_CHANGELOG.md`) and the full protocol (`docs/validity_study/rating_protocol_draft.md`) are not needed to rate.

## 1. The question

**Based only on what this narrative conveys, how favourably does it portray the organisation's safety practice?**

*Safety practice* is the state of safety as practised in that workplace: hazard recognition and control, procedures and whether they are followed, supervision, training and competence, communication and coordination, equipment and housekeeping, and the organisation's response to events (investigation, correction, learning).

*The organisation* is the reporting operation and the system it works in. For a controller's report it is the air traffic operation the reporter works in, including adjacent facilities, procedures and automation.

You see the narrative text and its industry, nothing else. Do not use structured data, indicator scores, or anything you know about the operator.

## 2. The answer

One of: **CJ** (can't judge), or **1 to 5**.

| Rating | Meaning |
|---|---|
| CJ | The narrative conveys no information about safety practice: mechanism of injury or failure only, or only acts that are required of every organisation. |
| 1 | Strongly unfavourable. More than a single frontline lapse: a known hazard left uncontrolled, a repeated problem, supervision that ignored it, or a dismissive or concealing response. |
| 2 | Somewhat unfavourable. A single lapse or deficiency is stated, with no response or only a required one. |
| 3 | Mixed. Both a stated lapse and a discretionary response or a control that worked, in the same narrative. |
| 4 | Somewhat favourable. Practice shown working: a hazard recognised and controlled, a procedure followed, a discretionary response, a change to how the work is done, or a potential risk identified before it caused harm. |
| 5 | Strongly favourable. Proactive, conservative practice: stop-work or conservative decisions, in-depth analysis or investigation, learning actions, explicit management commitment. |

Use the full scale **within each industry**: 1 is the clearly worst narrative you see in that industry, 5 the clearly best.

Uncertainty is not 3. If unsure, choose the nearest rating or CJ and write a note.

## 3. The procedure

Work through the steps in order.

### Step 1. Is there anything to rate? If any of these describes the whole narrative, answer CJ.

- **Mechanism only.** The narrative says what happened but not why, and nothing about practice. A wide gage, a shove over a switch, a broken rail, a head-on collision "under investigation", a slip on a frozen clod of mud. Severity does not matter: a fatality with no stated cause is CJ.
- **Equipment or material failure with no stated cause** and no description of inspection, maintenance or a discretionary response. A card failure, a relief valve lifting, an instrument spike attributed to electrical noise.
- **Required acts only.** Acts that every organisation must perform tell you nothing about this one. See the list in step 2.
- **Third-party conduct only.** Motorists, trespassers, shippers, customers, other railroads, pilots, airlines, the military, other airspace users. Their conduct is not the organisation's practice. A pilot who does not switch frequency or misses a readback is a third party. Contractors count as the organisation only when the narrative states they were working under the organisation's authority or instruction.
- **Report genres with no practice content.** Non-work-related events on site, supplier-defect (Part 21) notifications, routine spill or environmental notifications, planned-outage and scheduled-maintenance notifications, retracted notifications.
- **Form language only.** A checklist of things that were "not a factor", a generic disclaimer ("did not violate any laws", "all protocols were followed"), a required post-accident test result ("all tests were negative").

A narrative passes this step if it contains at least one of the four kinds of content in step 2.

### Step 2. What does the narrative state?

Note which of these are present. Only what is **stated or explicitly attributed** counts ("cause was…", "failed to…", "was not wearing…", "proficiency was a factor"). A hedged attribution counts the same as a certain one ("may have been a factor", "most likely human error").

**A. A lapse or deficiency of the organisation or its people.**
Non-compliance, a missing or inadequate control, a procedure not followed, a stated human or organisational cause (lack of awareness, improper lifting, a tagging error, a jumper installed contrary to procedure, work planning, undersized spare parts, worn bolts), a crew action or omission that caused the event ("failed to drop pin", "handbrake was not set"), a failed, refused or admitted fitness-for-duty test, a condition stated to be currently affecting safety.
Also note whether the narrative shows **more than a single lapse**: a known hazard left uncontrolled, the same problem recurring and the narrative connecting it ("the earlier repair did not resolve it", "this is the second time"), concerns raised and not acted on, supervision that ignored it, or a dismissive or concealing response.

**B. A discretionary response or a control that worked.**
Stop-work, denying service to regain control, placing the train in emergency on seeing a hazard, a colleague speaking up, a go-around issued in time, an investigation with findings, a change to how the work is done ("practices have been modified to prevent recurrence"), retraining, a review of the person's past work, conservative action beyond the requirement, a voluntary or courtesy notification that is the subject of the report.
**Not a response:** anything in the required-acts list below, discipline of an individual (suspension, dismissal, decertification), a recommendation written into the report, the act of filing the report, the timing of a report ("immediately reported", "reported the same day").

**C. A potential risk identified before it caused harm.**
A condition that could compromise safety but has not caused an event, found by the organisation's own review, analysis or extent-of-condition check, or raised by the reporter: an unanalysed design condition, an unsealed flood penetration found during an investigation, a hazard the reporter identifies with what should change. **Test: did a safety event occur in the narrative** (an injury, a loss of separation, a near miss, an incursion, an operational error, a failure on demand)? If yes, this is not C: rate what the event shows under A and B. If the narrative says the organisation already knew about the hazard and left it, that is A (more than a single lapse), not C.

**D. Dismissiveness.**
The reporter disputes the injury, minimises its severity ("a bulging disc which everyone has"), or is dismissive of the reporting requirement ("submitting this so MSHA doesn't shut down my mine"). This is unfavourable information about reporting culture.

**Required acts (never favourable, CJ on their own):** notifying the inspector or regulator; filing an Event Notification, LER, FRA or MSHA report; declaring a system inoperable; entering an LCO or technical-specification action; posting a required fire watch or compensatory measure; declaring an emergency classification and evacuating under the emergency plan; tripping and stabilising the plant; restricting or revoking access after a fitness-for-duty failure; posting, barricading, cribbing or "dangering off" a fall area after the fall; FRA-mandated post-accident testing; the timing of any report.

### Step 3. Assign the rating from what is present.

| Present | Rating |
|---|---|
| D only, or A with D | 2 (1 if A also shows more than a single lapse) |
| A only, a single lapse, with no response or a required one | 2 |
| A showing more than a single lapse | 1 |
| A and B together | 3 |
| B only, or C | 4 |
| B or C with depth: stop-work or a conservative decision, in-depth analysis or investigation, a specific fix implemented, learning actions, explicit management commitment | 5 |

Never 5 for reporting alone, even a voluntary notification.

### Step 4. Adjust for tone, at most one point.

Tone and self-criticism are evidence of climate but are never rateable on their own. After step 3, move the rating by **at most one point**:

- **Up** when the account is reflective and self-critical ("the fault lies entirely with me… I should have placed a closed-runway strip on my board").
- **Down** when the tone is frustration at concerns raised and not acknowledged or acted on, or resignation about a problem nobody fixes. If the narrative *states* that concerns were raised and ignored, that is already A (more than a single lapse); do not count it twice.

A constructive recommendation is not tone and does not move the rating.

### Step 5. Record

Enter 1 to 5 in `rating_1to5`, or Y in `cant_judge`, not both. A short note on borderlines helps.

## 4. Industry notes

**Nuclear (event notifications).** Most reports are required notifications and are CJ on their own. Rateable content is a stated human or organisational cause (usually 2), a condition found by the licensee's own review before any event (4, or 5 with analysis and corrective action), a voluntary or courtesy notification that is the subject of the report (4), or a change the licensee made to its procedures (4). A fitness-for-duty failure is 2; with a discretionary response such as a review of the person's past work, 3. Alcohol found and surrendered with no failure and no concerns is 4. Equipment found inoperable by routine surveillance or testing is an actual failure (CJ) unless a cause or practice is stated; a condition identified by analysis before any demand is a potential risk (4).

**Mining (accident narratives).** Most narratives are mechanism only (CJ). A named worker cause, including a hedged one, is 2. "Not a factor" lines and compliance disclaimers are form language; if nothing else is present, CJ. A specific control described in use is 4. Dangering off a fall area and notifying MSHA are required; an area dangered off *in anticipation* of a fall is 4. Disputing or minimising an injury is 2. Report timing never counts.

**Rail (accident narratives).** A stated crew action or omission is 2. Discipline is not a response: rate the lapse, 2. A crew's emergency application on seeing a hazard is 4 alone, 3 with a stated lapse by the organisation. Crossing devices "activated and functioning" and listed protection are baseline (CJ); devices "not operating" is unfavourable. Post-accident testing is not evidence. A contractor is the organisation only when the narrative states it worked under the railroad's authority. Worn, missing or broken components of the organisation's own equipment, stated as the cause, are a deficiency (2).

**Aviation (controller reports).** Every report is voluntary and most end with a recommendation; neither filing nor recommending raises the rating. Apply the step 2 test: if a safety event occurred, rate the conditions and the conduct it shows (a rushed briefing, a distracted controller, a sector overloaded by design, a colleague who spoke up, a go-around in time); if no event occurred and the reporter identifies a hazard with what should change, it is 4. A hazard the organisation knew about and left (an outage for weeks, a second occurrence, a supervisor who would not act) is 1. A pilot's miscommunication is third-party (CJ); a controller's is rateable. Pilots, airlines and the military are third parties unless the report has clear implications for practices under air traffic control's own control, including arrival and departure corridor rules.

## 5. Worked examples

| Industry | Narrative (abridged) | Steps | Rating |
|---|---|---|---|
| Mining | Employee was climbing out of the loader, stepped on a frozen clod of mud, twisting his ankle. | Mechanism only. | CJ |
| Mining | Knife slipped, laceration. "Lack of awareness was the cause. PPE was not a factor, equipment was not a factor, proficiency was a factor." | A: named cause. Checklist lines ignored. | 2 |
| Mining | Rib rolled, fatal injuries. "Reported to MSHA hot line on the day of the accident." | Mechanism; report timing does not count. | CJ |
| Mining | "Management was aware of adverse roof conditions… the area was dangered off in anticipation of a fall." | C: hazard recognised and controlled before the event. | 4 |
| Mining | "After investigation he has had back problems for some time… CT scan shows mild bulging disc which everyone has." | D: minimises the injury. | 2 |
| Rail | "Crew failed to comply with restricted speed, resulting in a rear-end collision. Engineer dismissed, conductor suspended." | A: stated lapse. Discipline is not a response. | 2 |
| Rail | "Head crew observed switch improperly lined and immediately placed train brakes in emergency." | A (switch not lined) and B (emergency application). | 3 |
| Rail | "Crew noticed sparks and smoke so they immediately stopped the train. A thermite weld had broken." | B only. | 4 |
| Rail | Truck struck at crossing; "gates were operating normally" per download and video. | Third party; devices functioning is baseline. | CJ |
| Rail | "Heavy dynamic braking caused a run-in. Practices have been modified to prevent reoccurrence." | A and B (change to the work). | 3 |
| Nuclear | Reactor tripped on loss of two coolant pumps; "Operations responded and stabilized the plant." Four-hour report. | Required acts only. | CJ |
| Nuclear | "Maintenance workers mistakenly drained the oil from the 2B safety injection pump" while the other train was tagged out. | A: stated human cause. | 2 |
| Nuclear | Positive random test for alcohol; "the employee's access to the plant has been restricted." | A; access restriction is required. | 2 |
| Nuclear | Refused random test; access denied; "licensee will write a condition report and review his past work." | A and B. | 3 |
| Nuclear | Fire dampers "have not been analyzed to withstand high air flows from a tornado… compensatory measures have been developed." No event. | C with analysis and action. | 4 or 5 |
| Nuclear | "This event is being voluntarily reported… monitoring of the wells will be increased to a monthly frequency." | B: voluntary notification is the subject. | 4 |
| Aviation | Controller cleared an aircraft to a closed runway; alerted by the adjacent controller and an AMASS alert; go-around issued. "The fault lies entirely with me… I should have placed a closed-runway strip." | Event. A (lapse) and B (controls worked) = 3; self-critical, up one. | 4 |
| Aviation | Sector combined from three into one; "I denied practice approaches to regain control… this sector is unsafe and needs to be split." | Event (unsafe workload). A (design deficiency) and B (stop-work) = 3; reflective, up one. | 4 |
| Aviation | "No chain of notification exists for start/end time of the TFR… this is unsafe and confusing." No event. | C: hazard identified, what should change. | 4 |
| Aviation | Radar failure; "the radar problems are becoming more numerous. Most are not reported by FAA officials." | Event; A, known problem not acted on. | 1 |
| Aviation | "The supervisor would not inform the maintenance technician, I had to do it. This is the second time." | A: repeated, supervision ignored it. | 1 |
| Aviation | Pilot did not switch to tower frequency and landed with a snowplough on the runway; reporter suggests tower inform centre when ploughs are out. | Third party (pilot miscommunication); recommendation does not change that. | CJ |
| Aviation | Pilot taxied past the hold line; the assist called it and the takeoff clearance was cancelled before the alarm. | Third-party lapse; B: the team's response worked. | 4 |

## 6. For model application

The model receives this guide verbatim, the industry, and the narrative, and returns the rating (CJ or 1 to 5), the step-2 content it found (A, B, C, D), and one sentence of basis. Same rules, same scale, no extra instructions. The model is run with a pinned version and fixed settings recorded in the protocol.
