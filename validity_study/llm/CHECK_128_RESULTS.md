# Model check on the 128 settled stage-2 narratives (2026-11-09)

In-sample prompt check (protocol §5.3): the model rated the 128 discordant stage-2 narratives
under the frozen guide, and its calls were compared with the final consensus calls in
`stage2_consensus_v1.1.csv`. These narratives are the ones the rules were developed on, so this
is a check that the prompt applies the guide as the raters do, not a reliability estimate.

Settings: `claude-opus-5-5`, adaptive thinking, effort high, structured output, one run per
narrative, no fallbacks. Guide sha256 `c24d9b0b…a1b1`; prompt sha256 `9be9c2ca…b509c`.
Ratings: `check_128.jsonl`. No errors, no refusals.

## Agreement with the consensus calls

| | n | CJ agreement | CJ κ | Both rated | Exact | Within 1 | Weighted κ | Mean model − human |
|---|---|---|---|---|---|---|---|---|
| All | 128 | 0.92 | 0.84 | 73 | 0.90 | 1.00 | 0.94 | −0.01 |
| Aviation | 17 | 1.00 | 1.00 | 14 | 0.86 | 1.00 | 0.95 | 0.00 |
| Mining | 30 | 0.90 | 0.79 | 17 | 1.00 | 1.00 | 1.00 | 0.00 |
| Nuclear | 47 | 0.87 | 0.74 | 23 | 0.87 | 1.00 | 0.92 | +0.04 |
| Rail | 34 | 0.97 | 0.94 | 19 | 0.89 | 1.00 | 0.91 | −0.11 |

For scale: the two human raters' stage-2 agreement on these same industries, before the rules
were revised, was weighted κ 0.47 pooled and CJ κ 0.19. The comparison is not like for like
(the consensus calls were made with the revised rules and the model sees the same rules), but
the model is applying the frozen guide the way the consensus did.

Confusion (rows human, columns model):

| | 1 | 2 | 3 | 4 | CJ |
|---|---|---|---|---|---|
| 1 | 2 | | | | |
| 2 | | 41 | 3 | | 3 |
| 3 | | 2 | 5 | | |
| 4 | | | 2 | 18 | 1 |
| CJ | | 3 | 1 | 2 | 45 |

## The 17 disagreements

Every disagreement is within one point or on the CJ boundary, and the model's stated basis cites
the guide correctly in each case. They fall into four groups.

1. **The guide and the consensus disagree with each other (3).** The model followed the guide.
   - R0850 (rail): "practices modified to prevent recurrence" after a stated lapse. Guide: A and B = 3, and the guide's own worked example (dynamic braking, counselled) says 3. Consensus: 4. The consensus label should probably be 3.
   - R0092 (nuclear): unsealed conduit, later retracted. The step 1 genre list includes "retracted notifications", so the model said CJ. Consensus: 4 (potential risk). Either the gate is too broad or the label is wrong.
   - R0019 (nuclear): Part 21 notice where the licensee did its own evaluation and entered it in the corrective action program. Model read that as a discretionary response (4); consensus CJ.
2. **Contractor clause applied to fitness-for-duty (2).** R0049 and R0614 are contract employees who failed fitness-for-duty tests. The model applied the step 1 contractor clause (CJ unless stated to be under the organisation's authority). The guide should say that anyone with plant access is the organisation for fitness-for-duty purposes; the contractor clause is about work crews.
3. **The model finds a discretionary response where the consensus saw a required one (5).** R0966, R0028, R0074, R0736, R0027: an investigation, a root-cause review, a voluntary LER or a conflict alert that fired, read as B, giving 3 where the consensus gave 2, 4 or CJ. The guide's B list includes "an investigation with findings"; the consensus was stricter about what counts.
4. **Judgment items (7).** R0977 (self-criticism not enough for 3), R0536 (maintenance cause stated: 2 vs CJ), R0209 (neutral medical review: CJ vs 2), R1134 (retraining read as post-event: 4 vs CJ), R1259 (missing collar as a deficiency: 2 vs CJ), R0223 (improper loading with no loader named: 2 vs CJ), R0970 (contractor lapse without a response: 2 vs 3). Two of these were already flagged as judgment calls in the consensus file.

## Cost and throughput

Mean per request: 6,366 cached guide tokens read, 413 uncached input tokens, 367 output tokens
(thinking included). At list prices this is about one cent per narrative live, half a cent in
batch mode. Live throughput with four workers: 128 narratives in about 3.5 minutes.

Estimates for the proposed pools (mining 6,000, rail 4,000, nuclear 1,500, aviation 800; mining
and rail narratives are shorter than these 128, nuclear and aviation longer): about $60 in batch
mode, under $130 live. Coding the full analysis corpora (about 290,000 narratives) would be in
the region of $1,500 in batch mode.

## What follows

- The prompt is frozen as run here (sha256 `9be9c2caac6a784e6633d40062a6a95c45137dab4f7de5b3b672923d597b509c`). The three groups above are guide or label issues, not prompt issues; they are logged in `RATER_GUIDE_CHANGELOG.md` for a later guide version and not applied.
- Decide whether the three consensus labels in group 1 are corrected to match the frozen guide (they are development labels, not frozen).
- Approve the pool sizes and the batch spend, then code the pools and draw the held-out sample.
