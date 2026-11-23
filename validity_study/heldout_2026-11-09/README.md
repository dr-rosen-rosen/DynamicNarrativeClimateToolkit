# Held-out sample (drawn 2026-11-09, seed 20261111)

300 narratives, 75 per industry, stratified on the model's call (protocol §5.4). Sheets are identical and blind; `heldout_key.csv` holds the strata, the model's call and the indicator scores. KEEP THE KEY FROM RATERS.

```
stratum   1-2   3  4-5  CJ
industry                  
aviation   19  19   19  18
msha       22  10   21  22
nrc        19  18   19  19
rail       19  18   19  19
```

## Files

- `heldout_sheet_raterA.xlsx`, `heldout_sheet_raterB.xlsx`: identical, blind. Columns rating_id (H0001-H0300), industry, text, rating_1to5, cant_judge, notes. Blocked by industry, shuffled within. 43,849 words in all.
- `heldout_key.csv`: eid, severity, stratum, the model's pool call, inclusion weight (pool stratum size / sample stratum size), pool population and seed, consensus and champion indicator scores. KEEP FROM RATERS until both sheets are returned.
- `heldout_items_for_model.csv`: the 300 narratives as model input; `model_ratings_x3.jsonl`: three independent model runs per narrative (run-to-run agreement, protocol §5.3).

## Model calls on the pools (for the record)

| Industry | Pool | CJ | 1 | 2 | 3 | 4 | 5 | Rateable share |
|---|---|---|---|---|---|---|---|---|
| Mining | 6,000 | 5,597 | 2 | 281 | 10 | 102 | 8 | 6.7% |
| Rail | 4,000 | 2,660 | 7 | 1,202 | 41 | 90 | 0 | 33.5% |
| Nuclear | 1,500 | 1,020 | 3 | 199 | 79 | 142 | 57 | 32.0% |
| Aviation | 800 | 55 | 214 | 189 | 199 | 139 | 4 | 93.1% |

The human stage-1 screen found about 4% of mining, 7% of rail, 30% of nuclear and 82% of aviation
narratives rateable. The model agrees in mining, nuclear and aviation but calls five times as many
rail narratives rateable, nearly all of them 2. The held-out sample will show whether the raters
agree with those calls: 19 of the 75 rail narratives come from the model's 1-2 stratum.

## Model run-to-run agreement (3 independent live runs, 2026-11-09)

900 requests, no errors. Identical call in all three runs: 88% pooled (aviation 79%, mining 91%,
nuclear 91%, rail 91%); within one point across all three: 94%; CJ decision identical across
the three runs: 96%. All three runs equal the batch call that defined the stratum for 85% of
narratives, so the strata are stable but not fixed: a narrative's stratum reflects one draw of
the model, which is why the sample includes the model's CJ calls and the analysis weights by
stratum rather than treating the model's call as truth.
