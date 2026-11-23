# Held-out sample: results (2026-11-20 15:18)

Script `dynclim/experiments/analyze_heldout.py`, seed 20261122, 2000 bootstrap resamples within industry x stratum. "As designed" is unweighted; "as in pool" applies the inclusion weights, normalised within industry.

Sheet completion: rater A 100%, rater B 100%. Both sheets complete.


## 1. Model run-to-run agreement (three independent runs)

|  | n | Identical call | Within one point | CJ decision identical | All three = pool call |
|---|---|---|---|---|---|
| Pooled | 300 | 88% | 94% | 96% | 85% |
| Mining | 75 | 91% | 95% | 95% | 85% |
| Rail | 75 | 91% | 95% | 95% | 89% |
| Nuclear | 75 | 91% | 96% | 97% | 88% |
| Aviation | 75 | 79% | 92% | 97% | 77% |


## 2. Design: strata by industry

```
stratum   1-2   3  4-5  CJ
industry                  
Aviation   19  19   19  18
Mining     22  10   21  22
Nuclear    19  18   19  19
Rail       19  18   19  19
```


## 3. Human-to-human agreement under the frozen guide

|  | Weighting | n | CJ rate (first) | CJ rate (second) | CJ agreement | CJ κ | Both rated | Exact | Within 1 | Weighted κ | Spearman | Mean difference |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Pooled | as designed | 300 | 34% | 55% | 67% | 0.37 | 118 | 60% | 83% | 0.53 | 0.51 | +0.38 |
| Pooled | as in pool | 300 | 54% | 68% | 79% | 0.57 | 118 | 61% | 78% | 0.46 | 0.47 | +0.53 |
| Mining | as designed | 75 | 57% | 64% | 75% | 0.47 | 20 | 75% | 100% | 0.89 | 0.86 | -0.05 |
| Mining | as in pool | 75 | 87% | 92% | 94% | 0.66 | 20 | 89% | 100% | 0.95 | 0.94 | -0.08 |
| Rail | as designed | 75 | 21% | 61% | 57% | 0.24 | 28 | 61% | 86% | 0.55 | 0.58 | +0.46 |
| Rail | as in pool | 75 | 45% | 83% | 63% | 0.29 | 28 | 87% | 98% | 0.65 | 0.57 | +0.14 |
| Nuclear | as designed | 75 | 37% | 61% | 68% | 0.39 | 26 | 77% | 92% | 0.77 | 0.79 | +0.00 |
| Nuclear | as in pool | 75 | 73% | 76% | 81% | 0.51 | 26 | 84% | 94% | 0.83 | 0.85 | +0.08 |
| Aviation | as designed | 75 | 19% | 33% | 69% | 0.22 | 44 | 43% | 68% | 0.22 | 0.24 | +0.75 |
| Aviation | as in pool | 75 | 9% | 23% | 79% | 0.24 | 44 | 47% | 68% | 0.26 | 0.32 | +0.78 |


Reading rule (pooled weighted κ ≥ 0.6): **not met**.


## 4. Human-to-model agreement (mean over the three model runs; mean difference is model minus rater)

|  | Weighting | n | CJ agreement | CJ κ | Both rated | Exact | Within 1 | Weighted κ | Spearman | Mean difference | κ min | κ max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Rater A, Pooled | as designed | 300 | 81% | 0.57 | 180 | 75% | 93% | 0.76 | 0.77 | +0.06 | 0.75 | 0.78 |
| Rater A, Pooled | as in pool | 300 | 87% | 0.73 | 180 | 69% | 88% | 0.65 | 0.65 | -0.07 | 0.64 | 0.66 |
| Rater A, Mining | as designed | 75 | 68% | 0.39 | 28 | 73% | 93% | 0.79 | 0.78 | +0.17 | 0.78 | 0.81 |
| Rater A, Mining | as in pool | 75 | 89% | 0.39 | 28 | 75% | 93% | 0.73 | 0.71 | +0.11 | 0.73 | 0.73 |
| Rater A, Rail | as designed | 75 | 84% | 0.58 | 51 | 92% | 100% | 0.94 | 0.95 | +0.05 | 0.92 | 0.96 |
| Rater A, Rail | as in pool | 75 | 71% | 0.44 | 51 | 99% | 100% | 0.98 | 0.98 | +0.01 | 0.97 | 0.98 |
| Rater A, Nuclear | as designed | 75 | 87% | 0.70 | 46 | 80% | 99% | 0.87 | 0.89 | +0.10 | 0.84 | 0.90 |
| Rater A, Nuclear | as in pool | 75 | 93% | 0.84 | 46 | 84% | 99% | 0.92 | 0.94 | +0.09 | 0.90 | 0.93 |
| Rater A, Aviation | as designed | 75 | 87% | 0.60 | 54 | 57% | 83% | 0.53 | 0.51 | -0.02 | 0.50 | 0.56 |
| Rater A, Aviation | as in pool | 75 | 93% | 0.49 | 54 | 55% | 80% | 0.52 | 0.50 | -0.15 | 0.50 | 0.53 |
| Rater B, Pooled | as designed | 300 | 65% | 0.34 | 124 | 57% | 86% | 0.59 | 0.61 | +0.47 | 0.58 | 0.60 |
| Rater B, Pooled | as in pool | 300 | 84% | 0.65 | 124 | 61% | 88% | 0.60 | 0.62 | +0.45 | 0.58 | 0.61 |
| Rater B, Mining | as designed | 75 | 64% | 0.34 | 24 | 68% | 92% | 0.74 | 0.80 | +0.35 | 0.71 | 0.76 |
| Rater B, Mining | as in pool | 75 | 93% | 0.43 | 24 | 86% | 97% | 0.86 | 0.90 | +0.16 | 0.85 | 0.87 |
| Rater B, Rail | as designed | 75 | 65% | 0.36 | 28 | 59% | 83% | 0.51 | 0.55 | +0.51 | 0.50 | 0.52 |
| Rater B, Rail | as in pool | 75 | 84% | 0.59 | 28 | 86% | 98% | 0.62 | 0.55 | +0.15 | 0.62 | 0.62 |
| Rater B, Nuclear | as designed | 75 | 58% | 0.24 | 26 | 75% | 95% | 0.81 | 0.84 | +0.08 | 0.76 | 0.84 |
| Rater B, Nuclear | as in pool | 75 | 79% | 0.47 | 26 | 79% | 95% | 0.85 | 0.88 | +0.15 | 0.82 | 0.87 |
| Rater B, Aviation | as designed | 75 | 75% | 0.38 | 44 | 40% | 79% | 0.40 | 0.47 | +0.73 | 0.38 | 0.41 |
| Rater B, Aviation | as in pool | 75 | 80% | 0.22 | 44 | 51% | 83% | 0.47 | 0.53 | +0.59 | 0.46 | 0.48 |


Supplementary: the pool call that defined the strata.

|  | Weighting | n | CJ agreement | CJ κ | Both rated | Exact | Within 1 | Weighted κ | Spearman | Mean difference |
|---|---|---|---|---|---|---|---|---|---|---|
| Model pool call vs rater A | as designed | 300 | 82% | 0.57 | 183 | 73% | 93% | 0.75 | 0.76 | +0.10 |
| Model pool call vs rater A | as in pool | 300 | 87% | 0.74 | 183 | 70% | 89% | 0.69 | 0.69 | -0.08 |
| Model pool call vs rater B | as designed | 300 | 65% | 0.33 | 126 | 56% | 87% | 0.60 | 0.62 | +0.44 |
| Model pool call vs rater B | as in pool | 300 | 83% | 0.64 | 126 | 64% | 89% | 0.63 | 0.64 | +0.41 |


## 5. Confusion (rows rater A, columns rater B; CJ included)

```
call_B  1   2  3   4  CJ
call_A                  
1       7   3  0   0   1
2       3  39  0   2  38
3       8  12  5   6  13
4       4   6  0  20  29
5       0   0  0   3   0
CJ      1   6  1   9  84
```


## 6. Validity: rating against the consensus indicator score (Spearman, within industry)

Positive means narratives rated more favourable carry higher indicator scores. Exploratory; small samples. Effective n is the Kish effective sample size under the inclusion weights; where it is far below n, the weighted estimate and its interval rest on a few heavily weighted narratives and should not be read.


| Rating |  | Weighting | n | Effective n | ρ | 2.5% | 97.5% | ρ, length controlled | ρ, severity and length controlled |
|---|---|---|---|---|---|---|---|---|---|
| Mean of two raters | Pooled (within-industry ranks) | as designed | 118 | 118.00 | +0.37 | +0.20 | +0.52 | — | — |
| Mean of two raters | Pooled (within-industry ranks) | as in pool | 118 | 63.84 | +0.38 | +0.16 | +0.57 | — | — |
| Mean of two raters | Mining | as designed | 20 | 20.00 | +0.31 | -0.12 | +0.67 | +0.30 | +0.32 |
| Mean of two raters | Mining | as in pool | 20 | 2.71 | +0.74 | +0.55 | +0.90 | +0.72 | +0.80 |
| Mean of two raters | Rail | as designed | 28 | 28.00 | -0.10 | -0.48 | +0.32 | -0.19 | -0.15 |
| Mean of two raters | Rail | as in pool | 28 | 11.85 | +0.08 | -0.20 | +0.31 | -0.04 | -0.04 |
| Mean of two raters | Nuclear | as designed | 26 | 26.00 | +0.39 | +0.04 | +0.68 | +0.07 | — |
| Mean of two raters | Nuclear | as in pool | 26 | 24.05 | +0.41 | +0.05 | +0.71 | +0.09 | — |
| Mean of two raters | Aviation | as designed | 44 | 44.00 | +0.25 | -0.05 | +0.51 | +0.25 | — |
| Mean of two raters | Aviation | as in pool | 44 | 36.45 | +0.31 | -0.02 | +0.58 | +0.30 | — |
| Rater A | Pooled (within-industry ranks) | as designed | 199 | 199.00 | +0.30 | — | — | — | — |
| Rater A | Pooled (within-industry ranks) | as in pool | 199 | 95.99 | +0.28 | — | — | — | — |
| Rater A | Mining | as designed | 32 | 32.00 | +0.28 | — | — | +0.31 | +0.24 |
| Rater A | Mining | as in pool | 32 | 4.53 | +0.73 | — | — | +0.74 | +0.68 |
| Rater A | Rail | as designed | 59 | 59.00 | +0.11 | — | — | +0.02 | +0.04 |
| Rater A | Rail | as in pool | 59 | 23.13 | +0.30 | — | — | +0.13 | +0.15 |
| Rater A | Nuclear | as designed | 47 | 47.00 | +0.25 | — | — | +0.16 | — |
| Rater A | Nuclear | as in pool | 47 | 41.98 | +0.29 | — | — | +0.18 | — |
| Rater A | Aviation | as designed | 61 | 61.00 | +0.27 | — | — | +0.26 | — |
| Rater A | Aviation | as in pool | 61 | 47.55 | +0.27 | — | — | +0.27 | — |
| Rater B | Pooled (within-industry ranks) | as designed | 135 | 135.00 | +0.39 | — | — | — | — |
| Rater B | Pooled (within-industry ranks) | as in pool | 135 | 68.38 | +0.40 | — | — | — | — |
| Rater B | Mining | as designed | 27 | 27.00 | +0.21 | — | — | +0.23 | +0.17 |
| Rater B | Mining | as in pool | 27 | 3.17 | +0.71 | — | — | +0.70 | +0.71 |
| Rater B | Rail | as designed | 29 | 29.00 | -0.23 | — | — | -0.20 | -0.18 |
| Rater B | Rail | as in pool | 29 | 12.01 | +0.12 | — | — | +0.10 | +0.08 |
| Rater B | Nuclear | as designed | 29 | 29.00 | +0.38 | — | — | +0.07 | — |
| Rater B | Nuclear | as in pool | 29 | 15.37 | +0.31 | — | — | +0.08 | — |
| Rater B | Aviation | as designed | 50 | 50.00 | +0.29 | — | — | +0.29 | — |
| Rater B | Aviation | as in pool | 50 | 39.27 | +0.33 | — | — | +0.32 | — |
| Model (mean of three runs) | Pooled (within-industry ranks) | as designed | 210 | 210.00 | +0.19 | +0.05 | +0.33 | — | — |
| Model (mean of three runs) | Pooled (within-industry ranks) | as in pool | 210 | 103.81 | +0.26 | +0.08 | +0.42 | — | — |
| Model (mean of three runs) | Mining | as designed | 47 | 47.00 | +0.15 | -0.14 | +0.44 | +0.15 | -0.04 |
| Model (mean of three runs) | Mining | as in pool | 47 | 33.90 | +0.18 | -0.11 | +0.47 | +0.18 | -0.05 |
| Model (mean of three runs) | Rail | as designed | 52 | 52.00 | +0.12 | -0.18 | +0.41 | +0.00 | +0.00 |
| Model (mean of three runs) | Rail | as in pool | 52 | 20.97 | +0.09 | -0.09 | +0.31 | -0.08 | -0.08 |
| Model (mean of three runs) | Nuclear | as designed | 54 | 54.00 | +0.13 | -0.16 | +0.42 | +0.03 | — |
| Model (mean of three runs) | Nuclear | as in pool | 54 | 48.63 | +0.16 | -0.14 | +0.46 | +0.05 | — |
| Model (mean of three runs) | Aviation | as designed | 57 | 57.00 | +0.27 | +0.00 | +0.51 | +0.27 | — |
| Model (mean of three runs) | Aviation | as in pool | 57 | 47.40 | +0.31 | +0.06 | +0.55 | +0.31 | — |


### Best-performing configurations (mean of two raters; secondary, uncorrected)

| Configuration |  | Weighting | n | Effective n | ρ | 2.5% | 97.5% | ρ, length controlled | ρ, severity and length controlled |
|---|---|---|---|---|---|---|---|---|---|
| score_cfg8 | Aviation | as designed | 44 | 44.00 | +0.15 | -0.15 | +0.43 | +0.15 | — |
| score_cfg8 | Aviation | as in pool | 44 | 36.45 | +0.13 | -0.20 | +0.43 | +0.14 | — |
| score_cfg10 | Aviation | as designed | 44 | 44.00 | +0.02 | -0.29 | +0.33 | +0.02 | — |
| score_cfg10 | Aviation | as in pool | 44 | 36.45 | +0.06 | -0.30 | +0.41 | +0.08 | — |
| score_cfg26 | Aviation | as designed | 44 | 44.00 | +0.11 | -0.19 | +0.39 | +0.11 | — |
| score_cfg26 | Aviation | as in pool | 44 | 36.45 | +0.12 | -0.21 | +0.43 | +0.13 | — |
| score_cfg59 | Aviation | as designed | 44 | 44.00 | +0.16 | -0.13 | +0.41 | +0.16 | — |
| score_cfg59 | Aviation | as in pool | 44 | 36.45 | +0.20 | -0.11 | +0.47 | +0.19 | — |
| score_cfg62 | Aviation | as designed | 44 | 44.00 | +0.16 | -0.12 | +0.40 | +0.17 | — |
| score_cfg62 | Aviation | as in pool | 44 | 36.45 | +0.15 | -0.13 | +0.40 | +0.17 | — |
| score_cfg40 | Mining | as designed | 20 | 20.00 | +0.46 | +0.13 | +0.74 | +0.45 | +0.26 |
| score_cfg40 | Mining | as in pool | 20 | 2.71 | +0.74 | +0.59 | +0.88 | +0.74 | +0.78 |
| score_cfg46 | Mining | as designed | 20 | 20.00 | +0.05 | -0.43 | +0.47 | +0.08 | +0.34 |
| score_cfg46 | Mining | as in pool | 20 | 2.71 | +0.44 | +0.01 | +0.72 | +0.40 | +0.51 |
| score_cfg47 | Mining | as designed | 20 | 20.00 | +0.41 | +0.18 | +0.64 | +0.39 | +0.24 |
| score_cfg47 | Mining | as in pool | 20 | 2.71 | +0.72 | +0.62 | +0.84 | +0.72 | +0.77 |
| score_cfg89 | Mining | as designed | 20 | 20.00 | +0.45 | +0.13 | +0.70 | +0.43 | +0.29 |
| score_cfg89 | Mining | as in pool | 20 | 2.71 | +0.75 | +0.62 | +0.87 | +0.75 | +0.81 |
| score_cfg131 | Mining | as designed | 20 | 20.00 | +0.54 | +0.30 | +0.74 | +0.53 | +0.39 |
| score_cfg131 | Mining | as in pool | 20 | 2.71 | +0.82 | +0.74 | +0.89 | +0.82 | +0.87 |
| score_cfg29 | Nuclear | as designed | 26 | 26.00 | -0.09 | -0.49 | +0.28 | +0.07 | — |
| score_cfg29 | Nuclear | as in pool | 26 | 24.05 | -0.08 | -0.50 | +0.30 | +0.09 | — |
| score_cfg102 | Nuclear | as designed | 26 | 26.00 | +0.68 | +0.44 | +0.86 | +0.30 | — |
| score_cfg102 | Nuclear | as in pool | 26 | 24.05 | +0.71 | +0.45 | +0.89 | +0.30 | — |
| score_cfg119 | Nuclear | as designed | 26 | 26.00 | +0.65 | +0.38 | +0.88 | +0.23 | — |
| score_cfg119 | Nuclear | as in pool | 26 | 24.05 | +0.69 | +0.38 | +0.90 | +0.20 | — |
| score_cfg120 | Nuclear | as designed | 26 | 26.00 | +0.65 | +0.42 | +0.85 | +0.34 | — |
| score_cfg120 | Nuclear | as in pool | 26 | 24.05 | +0.70 | +0.48 | +0.89 | +0.37 | — |
| score_cfg124 | Nuclear | as designed | 26 | 26.00 | +0.38 | +0.02 | +0.69 | +0.11 | — |
| score_cfg124 | Nuclear | as in pool | 26 | 24.05 | +0.38 | -0.02 | +0.70 | +0.06 | — |
| score_cfg30 | Rail | as designed | 28 | 28.00 | +0.23 | -0.15 | +0.55 | +0.27 | +0.26 |
| score_cfg30 | Rail | as in pool | 28 | 11.85 | +0.28 | -0.04 | +0.55 | +0.23 | +0.22 |
| score_cfg39 | Rail | as designed | 28 | 28.00 | +0.11 | -0.28 | +0.47 | +0.10 | +0.07 |
| score_cfg39 | Rail | as in pool | 28 | 11.85 | +0.15 | -0.08 | +0.36 | +0.08 | +0.03 |
| score_cfg127 | Rail | as designed | 28 | 28.00 | -0.35 | -0.61 | -0.07 | -0.36 | -0.33 |
| score_cfg127 | Rail | as in pool | 28 | 11.85 | -0.22 | -0.50 | -0.08 | -0.26 | -0.24 |


## 7. Validity at scale: model rating against the indicator on the pools (protocol 5.5 item 1)

|  | Score | Pool | Rated by model | ρ | 2.5% | 97.5% | ρ, length controlled | ρ, severity and length controlled |
|---|---|---|---|---|---|---|---|---|
| Mining | score_consensus | 6000 | 403 | +0.31 | +0.21 | +0.40 | +0.31 | +0.19 |
| Mining | score_cfg40 | 6000 | 403 | +0.28 | — | — | +0.28 | +0.10 |
| Mining | score_cfg46 | 6000 | 403 | +0.27 | — | — | +0.27 | +0.23 |
| Mining | score_cfg47 | 6000 | 403 | +0.18 | — | — | +0.18 | +0.05 |
| Mining | score_cfg89 | 6000 | 403 | +0.24 | — | — | +0.24 | +0.06 |
| Mining | score_cfg131 | 6000 | 403 | +0.24 | — | — | +0.24 | +0.09 |
| Rail | score_consensus | 4000 | 1340 | +0.13 | +0.08 | +0.17 | +0.06 | +0.06 |
| Rail | score_cfg30 | 4000 | 1340 | +0.07 | — | — | +0.06 | +0.07 |
| Rail | score_cfg39 | 4000 | 1340 | +0.09 | — | — | +0.07 | +0.07 |
| Rail | score_cfg127 | 4000 | 1340 | +0.00 | — | — | -0.03 | -0.03 |
| Nuclear | score_consensus | 1500 | 480 | +0.10 | +0.01 | +0.19 | +0.01 | — |
| Nuclear | score_cfg29 | 1500 | 480 | -0.20 | — | — | -0.04 | — |
| Nuclear | score_cfg102 | 1500 | 480 | +0.23 | — | — | +0.02 | — |
| Nuclear | score_cfg119 | 1500 | 480 | +0.29 | — | — | +0.08 | — |
| Nuclear | score_cfg120 | 1500 | 480 | +0.20 | — | — | -0.01 | — |
| Nuclear | score_cfg124 | 1500 | 480 | +0.16 | — | — | +0.06 | — |
| Aviation | score_consensus | 800 | 745 | +0.39 | +0.32 | +0.45 | +0.39 | — |
| Aviation | score_cfg8 | 800 | 745 | +0.14 | — | — | +0.14 | — |
| Aviation | score_cfg10 | 800 | 745 | +0.25 | — | — | +0.25 | — |
| Aviation | score_cfg26 | 800 | 745 | +0.14 | — | — | +0.14 | — |
| Aviation | score_cfg59 | 800 | 745 | +0.26 | — | — | +0.26 | — |
| Aviation | score_cfg62 | 800 | 745 | +0.14 | — | — | +0.14 | — |
