# `linguistic_features.py` — Design Spec

> Working spec for adding lexical and syntactic features to the `dynclim` Python
> module. Drop this into a code session as a reference. The R-side layer
> architecture has changed since this spec was written; the **Python feature
> extraction is independent of that change** and is what this doc covers. Layer
> integration on the R side is left out deliberately and should be handled
> separately once the R architecture is stable.

---

## 1. Goals

Add a new module, `linguistic_features.py`, that produces per-segment and
per-report lexical and syntactic features for safety incident narratives, using
spaCy's POS tagger, morphologizer, and dependency parser. **No LIWC.** Outputs
must be:

1. **Segment-level by default**, with report-level aggregation as a separate
   step (parallel to how similarity/sentiment work today).
2. **Aggregatable using climate-feature weights**, so a downstream multiverse
   spec can ask for "passive voice rate weighted by management-commitment
   similarity" — not just unconditional rates.
3. **Conventionally consistent with `temporal_features.py`**: pure functions for
   the math, an extractor class for the cached/stateful pieces, dataclass-based
   config, no hidden global state.

---

## 2. Module placement

```
dynclim/
  temporal_features.py        # existing
  ews.py                      # existing
  linguistic_features.py      # NEW — this spec
  feature_aggregation.py      # NEW (optional split) — segment→report rollups
```

If you want a single file, put aggregation at the bottom of
`linguistic_features.py`. The split is only worth doing if the aggregation
helpers grow large enough to deserve their own namespace.

The module depends on `spacy` (already a project dependency) and `pandas`/
`numpy`. **No new heavy deps.** Optionally, `pyphen` for syllable counts if you
ever want readability indices — leave out of v1.

---

## 3. Feature set

Five families. Implement family-by-family so you can ship after any one. Family
A is the minimum viable contribution; B–E are increments.

### A. Passive voice (highest priority — this replaces the old detector)

Detected via spaCy dependency labels. A sentence (or clause) is passive iff it
contains `nsubjpass` **or** `auxpass`. For finer detection, also flag
`csubjpass` (rare).

Per-segment outputs:
- `passive_n` — count of passive constructions
- `passive_per_sent` — passives / number of sentences in segment
- `passive_per_clause` — passives / number of finite clauses (clauses ≈ verbs
  with a subject dependency)
- `passive_rate_tokens` — passive verb tokens / total tokens (most robust
  across segment lengths)

Pick **one primary** for the multiverse and treat the others as robustness
specs. I'd default to `passive_per_sent` and use `passive_rate_tokens` as the
length-normalized check.

### B. Function-word / style (Pennebaker core, POS-derived)

All computed as **rate per token**, with raw counts retained for downstream
re-aggregation.

| Feature | spaCy detection |
|---|---|
| `pron_i_rate` | `tag_ in {"PRP","PRP$"}` + lemma in `{"i","me","my","mine","myself"}` |
| `pron_we_rate` | lemma in `{"we","us","our","ours","ourselves"}` |
| `pron_you_rate` | lemma in `{"you","your","yours","yourself","yourselves"}` |
| `pron_3rd_rate` | lemma in `{"he","she","it","they","him","her","them","his","hers","its","their","theirs"}` |
| `article_rate` | `pos_ == "DET"` + lemma in `{"a","an","the"}` |
| `prep_rate` | `pos_ == "ADP"` |
| `cconj_rate` | `pos_ == "CCONJ"` |
| `sconj_rate` | `pos_ == "SCONJ"` |
| `aux_rate` | `pos_ == "AUX"` |
| `modal_rate` | `tag_ == "MD"` |
| `neg_rate` | `dep_ == "neg"` OR lemma in `{"not","no","never","none","nothing","nobody","nowhere","neither","nor"}` |
| `quant_rate` | small curated list: `{"all","any","some","many","few","several","most","both","each","every"}` |

These do not require LIWC. The pronoun decomposition is the single most useful
piece for Pennebaker-style analysis.

### C. Tense and aspect

From spaCy morph features (`token.morph.get("Tense")`, `.get("Aspect")`,
`.get("VerbForm")`). Per-segment rates over finite verbs:

- `past_rate` — `Tense=Past`
- `pres_rate` — `Tense=Pres`
- `future_rate` — modal `will` / `shall` directly governing a `VERB`, or
  `going to` + `VERB` heuristic (regex on lemmas in dependency window)
- `progressive_rate` — `Aspect=Prog`
- `perfect_rate` — `Aspect=Perf`

Future tense in English is annoying; document the heuristic clearly and
include it as a separate multiverse dimension (compute / don't compute).

### D. POS rates and lexical diversity

- `noun_rate`, `verb_rate`, `adj_rate`, `adv_rate`, `propn_rate`
- `content_word_ratio` — content (NOUN+VERB+ADJ+ADV+PROPN) / function (rest)
- `ttr` — type-token ratio, lemmas
- `mattr` — moving-average type-token ratio, window=50 tokens (length-robust);
  fall back to TTR if segment < 50 tokens

Flag `mattr` as the recommended diversity measure for cross-industry
comparisons since FRA narratives are short and NRC LERs are long.

### E. Syntactic complexity

Per-segment, averaged over sentences (skip sentences < 3 tokens):

- `mean_sent_len_tokens`
- `mean_dep_distance` — mean of `abs(token.i - token.head.i)` over non-root
  tokens
- `max_tree_depth` — depth of dependency tree (recursive walk from root)
- `clause_density` — finite verbs / sentences (proxy for clauses per sentence)
- `subord_rate` — `mark` dependencies / clauses

### F. (Optional) Pennebaker composites

Once A–E are stable, expose two derived composites:

- `analytical_thinking` ≈ standardized combination favoring article and prep
  rates over pronoun and aux rates (Pennebaker's formula uses 6 categories;
  cite the paper, document the exact weights you use)
- `clout` ≈ favors `we` and `you` over `i`

Keep these **optional and clearly labeled as approximations** — the components
themselves are what the multiverse should sweep over. Composites are a
convenience for narrative reporting, not for model fitting.

---

## 4. Granularity: segment-level primary, report-level by aggregation

**Compute features per segment.** Aggregate to report level as a separate,
configurable step. This mirrors how similarity and sentiment work in the
existing pipeline.

Why segment level is primary:
- Symmetry with existing pipeline (one less special case)
- Enables climate-weighted aggregation (the interesting analyses)
- Naturally normalizes for report length across industries

Why you still need report-level rollups:
- Outcome models live at report-quarter or site-quarter level
- Some features (MATTR, syntactic complexity) need a minimum token count;
  use report-level fallback when segment is too short

### Stability floor

Define a minimum-token threshold (`min_tokens_for_complexity`, default 20) below
which complexity and diversity features are returned as `NaN` rather than
computed. Rate-style features (everything in B, C, and most of D) are stable on
short segments and should always be returned.

### Aggregation methods (this is a multiverse dimension)

The aggregator takes a per-segment feature DataFrame and optional weights
vector and returns one row of report-level features.

1. **`mean`** — unweighted across segments
2. **`length_weighted`** — weighted by segment token count (defensible default)
3. **`climate_weighted`** — weighted by climate similarity to a target dimension
4. **`climate_top_k`** — restrict to top-k climate-similar segments, then mean
5. **`climate_threshold`** — restrict to segments above similarity threshold τ,
   then mean

For rates that are already token-normalized (everything ending in `_rate`),
length-weighting recovers the report-level rate exactly. For counts, sum;
for ratios, length-weight. Document this in the aggregator.

---

## 5. API

Match `temporal_features.py` conventions: dataclass config, pure compute
functions, optional class wrapper for stateful caching.

```python
# linguistic_features.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Sequence, Literal, Union
import numpy as np
import pandas as pd
import spacy
from spacy.tokens import Doc

AggMethod = Literal["mean", "length_weighted", "climate_weighted",
                    "climate_top_k", "climate_threshold"]

@dataclass
class LinguisticConfig:
    # Which feature families to compute
    compute_passive: bool = True
    compute_style: bool = True       # Family B
    compute_tense: bool = True       # Family C
    compute_pos_rates: bool = True   # Family D part 1
    compute_diversity: bool = True   # Family D part 2 (TTR, MATTR)
    compute_complexity: bool = True  # Family E
    compute_composites: bool = False # Family F (off by default)

    # Stability floors
    min_tokens_for_complexity: int = 20
    min_tokens_for_diversity: int = 20
    mattr_window: int = 50

    # spaCy model
    spacy_model: str = "en_core_web_sm"

    # Whether to include raw counts alongside rates (useful for re-aggregation)
    include_counts: bool = True


@dataclass
class AggregationConfig:
    method: AggMethod = "length_weighted"
    # for climate_weighted / climate_top_k / climate_threshold:
    k: Optional[int] = None
    threshold: Optional[float] = None
    # if True, treat weights as already-normalized; else normalize to sum to 1
    normalize_weights: bool = True


# --- Pure feature functions on a spaCy Doc -----------------------------------

def passive_features(doc: Doc) -> dict[str, float]: ...
def style_features(doc: Doc) -> dict[str, float]: ...
def tense_features(doc: Doc) -> dict[str, float]: ...
def pos_rate_features(doc: Doc) -> dict[str, float]: ...
def diversity_features(doc: Doc, mattr_window: int = 50,
                       min_tokens: int = 20) -> dict[str, float]: ...
def complexity_features(doc: Doc, min_tokens: int = 20) -> dict[str, float]: ...
def composite_features(features: dict[str, float]) -> dict[str, float]: ...


# --- Top-level extractor on text or Doc --------------------------------------

def extract_linguistic_features(
    doc_or_text: Union[str, Doc],
    config: LinguisticConfig = LinguisticConfig(),
    nlp: Optional[spacy.Language] = None,
) -> dict[str, float]:
    """
    Compute one row of linguistic features for a single segment or report.
    If passed a string, parses with `nlp` (or loads config.spacy_model).
    Returns a flat dict of feature_name -> value, with NaN for features that
    fall below their stability floor.
    """
    ...


def extract_segment_features(
    segments: Sequence[str],
    config: LinguisticConfig = LinguisticConfig(),
    nlp: Optional[spacy.Language] = None,
) -> pd.DataFrame:
    """
    Batch-parse segments with nlp.pipe and return a DataFrame:
    one row per segment, columns = features, plus 'segment_index' and 'n_tokens'.
    """
    ...


# --- Aggregation: segment-level -> report-level ------------------------------

def aggregate_segment_features(
    segment_features: pd.DataFrame,
    weights: Optional[Sequence[float]] = None,
    config: AggregationConfig = AggregationConfig(),
    segment_lengths: Optional[Sequence[int]] = None,
) -> dict[str, float]:
    """
    Roll up segment-level features to report level.
    - weights: optional per-segment weights (e.g., climate similarities)
    - segment_lengths: required for length_weighted (use 'n_tokens' from
      extract_segment_features output)
    """
    ...


# --- Stateful wrapper for batch jobs (parallels SafetyFeatureExtractor) ------

class LinguisticFeatureExtractor:
    """
    Holds a loaded spaCy pipeline so it isn't reloaded per call.
    Provides .extract(text), .extract_segments(segments), .extract_report(text,
    segmenter, aggregation_config, weights=None).
    """
    def __init__(self, config: LinguisticConfig = LinguisticConfig()):
        self.config = config
        self.nlp = spacy.load(config.spacy_model)

    def extract(self, text: str) -> dict[str, float]: ...
    def extract_segments(self, segments: Sequence[str]) -> pd.DataFrame: ...
    def extract_report(
        self,
        text: str,
        segmenter,                                # a Segmenter from safety_feature_toolkit
        agg_config: AggregationConfig = AggregationConfig(),
        weights: Optional[Sequence[float]] = None,
    ) -> dict[str, float]: ...
```

### Why this shape

- **`extract_linguistic_features(doc_or_text, ...)`** is the one function the
  rest of the system actually calls per segment. Everything else is plumbing.
- The family functions (`passive_features`, `style_features`, ...) are
  separately callable so they can be unit-tested without spinning up the full
  pipeline.
- **`extract_segment_features`** uses `nlp.pipe()` for batching — much faster
  than parsing one segment at a time. This is non-negotiable for the multiverse.
- **`aggregate_segment_features`** is a pure function on a DataFrame so it can
  be sweeped over in the multiverse without touching the spaCy pipeline.
- **`LinguisticFeatureExtractor.extract_report`** is sugar that composes
  segmenter + extract_segment_features + aggregate_segment_features, useful for
  one-off scoring.

### Naming conventions

- All feature keys use snake_case
- Rates end in `_rate` (token-normalized) or `_per_sent` / `_per_clause`
- Raw counts end in `_n`
- All values are `float`; missing/undefined values are `np.nan`, never 0

---

## 6. Implementation notes

### 6.1 spaCy pipeline efficiency

The default `en_core_web_sm` pipeline includes `tok2vec, tagger, parser,
attribute_ruler, lemmatizer, ner`. You **do not need NER** for any of these
features. Disable it at load:

```python
nlp = spacy.load("en_core_web_sm", disable=["ner"])
```

Use `nlp.pipe(segments, batch_size=64, n_process=1)` for batch parsing. Don't
crank `n_process` above 1 inside an already-parallel multiverse runner — fork
contention will dominate.

### 6.2 What "sentence" means

`spacy.Doc.sents` requires either the parser or the senter. With the parser
enabled (default), `sents` works. If you ever swap to a parser-less pipeline
for speed, add `senter` explicitly. Don't roll your own sentence splitter for
linguistic feature counting — sentence boundaries must match what spaCy uses
internally, or `mean_sent_len_tokens` will be inconsistent with `clause_density`.

### 6.3 Edge cases that will bite you

1. **Empty segments / single-token segments.** Return all-NaN dict, not an
   error. The aggregator should drop NaN before computing means.
2. **Segments with no finite verbs.** Tense rates are undefined (NaN), not 0.
   This matters — log narratives often contain noun fragments.
3. **All-caps text** (some FRA narratives). spaCy's tagger handles this
   reasonably but accuracy drops. Document this; don't lowercase preemptively
   because it kills proper noun detection.
4. **Numeric tokens and IDs.** Exclude `pos_ in {"NUM","SYM","PUNCT","X"}`
   from the token denominator for rate calculations? **No** — keep them in the
   denominator so rates are comparable to standard reporting. But exclude them
   from `ttr` / `mattr` (lemmatize and filter to alphabetic).
5. **Negation scope.** `neg_rate` counts negation markers, not negated
   propositions. Don't try to do scope resolution; flag it as a known
   limitation.

### 6.4 Climate-weighted aggregation: the alignment problem

The weights vector (climate similarities) and the segment-features DataFrame
must be aligned by segment order. Enforce this in `aggregate_segment_features`:
require `len(weights) == len(segment_features)` and assume position-based
correspondence. The caller is responsible for aligning them, and the obvious
caller is `LinguisticFeatureExtractor.extract_report`.

If a segment was dropped (e.g., empty after segmentation), drop the
corresponding weight too. This is a frequent bug source — write a test for it.

### 6.5 Cache compatibility

The existing `ImprovedCachedSafetyProcessor` caches segment lists and segment
embeddings keyed on text. For linguistic features, the natural cache key is
segment text → feature dict. This is cheap to cache because the per-segment
dict is small. **In v1, don't bother with caching** — measure first. spaCy
parsing of small segments is fast enough that the multiverse overhead is
likely elsewhere.

### 6.6 Output schema for downstream R consumption

When `extract_report` returns to a per-report DataFrame for the R pipeline,
column names should be: `ling__<family>__<feature>` for namespacing, e.g.
`ling__style__pron_i_rate`, `ling__passive__passive_per_sent`. This keeps the
R-side join clean and visually distinguishes linguistic features from temporal
(`temp__...`) and EWS (`ews__...`) features. If those prefixes aren't already
in use, propose them as part of this change.

---

## 7. Testing

Minimum viable test set before integrating:

1. **Sanity** — one hand-crafted segment with known passives, known pronouns,
   known POS distribution. Assert exact values.
2. **Empty / degenerate input** — empty string, single token, single
   punctuation. Should return dict with NaN, not crash.
3. **Aggregation correctness** — for rate features with `length_weighted`,
   verify that aggregating segment rates by token weights recovers the rate
   computed on the concatenated text. (This is the math check that catches
   off-by-one segment-alignment bugs.)
4. **Weights alignment** — feed a deliberately misaligned weight vector,
   verify it raises rather than silently miscomputing.
5. **Performance smoke test** — 1000 NRC-length segments through
   `extract_segment_features` should complete in < 30 seconds on a laptop.
   If not, the spaCy disable list or batch_size is wrong.

---

## 8. Build order

1. `passive_features` + tests. Ship this first; it replaces the old detector.
2. `extract_linguistic_features` skeleton with passive-only.
3. `style_features` + `pos_rate_features` (these share token iteration; write
   them together).
4. `extract_segment_features` with `nlp.pipe`.
5. `aggregate_segment_features` with `mean` and `length_weighted` only.
6. `tense_features`, `diversity_features`, `complexity_features`.
7. Climate-weighted aggregation methods.
8. `LinguisticFeatureExtractor` wrapper.
9. (Defer) Pennebaker composites — only after components are validated.

After step 5 you have a working end-to-end pipeline for passive voice alone,
which is enough to begin R-side integration work in parallel.

---

## 9. Explicit non-goals for this module

- **No LIWC.** Approximate the dimensions you need with POS + small lexicons,
  document it as such.
- **No semantic features.** Embeddings and similarity live elsewhere.
- **No layer integration.** That belongs in the R pipeline and depends on the
  new R layer architecture, which is out of scope here.
- **No outcome modeling.** This module produces features; CV and multiverse
  runners consume them.
- **No "linguistic style score" as a single number.** The point is to expose
  components and let the multiverse sweep over them. Composites (Family F) are
  optional and clearly labeled as approximate.

---

## 10. Open questions — RESOLVED (2026-06-21)

All four are settled in the as-built module; recorded here so the spec matches
the code.

1. **Feature prefix convention** — adopted `ling__<family>__<feature>`,
   implemented in `namespace_features()`. Families: `passive`, `style`,
   `tense`, `pos`, `diversity`, `complexity`, `composite`.
2. **Future tense detection** — kept. Heuristic (modal `will`/`shall`
   governing a VERB, or `going to` + VERB) is implemented and gated behind
   `LinguisticConfig.compute_future` so it's a clean multiverse on/off
   dimension.
3. **Composite formulas** — two forms. Per-span `composite_features()` is a raw
   convenience (documented as scale-dominated, not for modeling). The form to
   use is corpus-level `standardize_composites(df)`: z-score each component rate
   across the corpus, then combine with documented +/-1 weights — this is what
   "standardized combination" means and is the only scale at which it's
   meaningful. Components/weights live in `_ANALYTICAL_COMPONENTS` /
   `_CLOUT_COMPONENTS`. Still labelled an approximation, NOT LIWC.
4. **MATTR window** — kept at 50. Confirmed on real data: aviation reports are
   median ~320 tokens (p10 ~158), 0% fall back to TTR at report level. Two-tier
   design computes diversity at report level only, so short FRA/MSHA *segments*
   never reach this path; short *reports* (common in rail/MSHA) correctly return
   NaN below `min_tokens_for_diversity`.

## 11. As-built status (2026-06-21)

- Module: `dynclim/features/linguistic/linguistic_features.py` — families A–F
  implemented; 13/13 tests pass (`test_linguistic_features.py`).
- Driver: `dynclim/features/linguistic/compute_linguistic_features.py` — reads
  `dynclim/data/processed/<industry>/events.parquet`, computes whole-narrative
  features in a single `nlp.pipe` parse per report, adds corpus-standardized
  composites, and writes `linguistic_features_<industry>.parquet` keyed by
  `eid` with `ling__` columns. Industries wired: **aviation, rail, msha**.
  **nrc excluded** until narratives are re-scraped (lowercased + word-glued).
- The module lives under `dynclim/features/linguistic/` (a package), not at the
  `dynclim/` root as §2 originally sketched.
- Climate-weighted segment aggregation (the per-`(eid, segment_index)` climate
  emitter) is the next phase and is intentionally NOT part of this driver.

> Env note: a clean run needs spaCy + `en_core_web_sm` + pandas + pyarrow on a
> NumPy < 2 interpreter. The system Python 3.12 currently has a NumPy 2 / SciPy
> mismatch that breaks `import dynclim` (via `sentence_transformers`); the
> `~/.virtualenvs/r-spacyr` env (NumPy 1.26) was used for validation.
