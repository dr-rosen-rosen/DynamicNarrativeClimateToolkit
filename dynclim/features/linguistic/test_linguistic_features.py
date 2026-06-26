"""
Tests for linguistic_features. Runnable with pytest or as a plain script:
    python -m dynclim.features.linguistic.test_linguistic_features
Loads spaCy once at module level (model load dominates runtime).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import spacy

from dynclim.features.linguistic.linguistic_features import (
    LinguisticConfig,
    AggregationConfig,
    LinguisticFeatureExtractor,
    passive_features,
    style_features,
    pos_rate_features,
    complexity_features,
    standardize_composites,
    extract_linguistic_features,
    extract_segment_features,
    aggregate_segment_features,
)

NLP = spacy.load("en_core_web_sm", disable=["ner"])
CFG = LinguisticConfig()


def test_passive_sanity():
    doc = NLP("The valve was opened by the operator. The technician closed the door.")
    pf = passive_features(doc)
    # one passive clause ("was opened") across two sentences
    assert pf["passive_n"] == 1.0
    assert pf["passive_per_sent"] == 0.5
    assert 0.0 < pf["passive_rate_tokens"] < 1.0


def test_style_pronoun_decomposition():
    doc = NLP("I told you that we saw it.")
    sf = style_features(doc)
    assert sf["pron_i_n"] == 1.0
    assert sf["pron_you_n"] == 1.0
    assert sf["pron_we_n"] == 1.0
    assert sf["pron_3rd_n"] == 1.0  # "it"


def test_pos_rates_sum_sane():
    doc = NLP("The quick brown fox jumped over the lazy dog.")
    pr = pos_rate_features(doc)
    assert pr["noun_rate"] > 0
    assert pr["verb_rate"] > 0
    # content_word_ratio is content/function and must be finite & positive here
    assert np.isfinite(pr["content_word_ratio"]) and pr["content_word_ratio"] > 0


def test_empty_and_single_token_return_nan_not_crash():
    for bad in ["", "   ", "."]:
        feats = extract_linguistic_features(bad, CFG, NLP)
        assert isinstance(feats, dict) and len(feats) > 0
        assert all((v != v) for v in feats.values()), f"non-NaN for input {bad!r}"


def test_empty_schema_matches_full_schema():
    full = extract_linguistic_features(NLP("the inspector verified the valve was opened safely twice"), CFG, NLP)
    empty = extract_linguistic_features("", CFG, NLP)
    assert set(full.keys()) == set(empty.keys())


def test_tense_undefined_without_finite_verbs():
    # noun fragment: no finite verb -> tense rates NaN, not 0
    feats = extract_linguistic_features(NLP("Hydraulic pressure loss near pump bay"), CFG, NLP)
    assert feats["past_rate"] != feats["past_rate"]  # NaN


def test_length_weighted_recovers_rate():
    """
    length_weighted aggregation of a token-rate must satisfy
    aggregated_rate * total_tokens == summed count (internal consistency),
    and approximately equal the rate on the concatenated text.
    """
    segs = [
        "The pump was inspected by the crew.",
        "The operator reported a leak and we isolated the line.",
        "Maintenance replaced the seal the next morning.",
    ]
    sf = extract_segment_features(segs, CFG, NLP)
    agg = aggregate_segment_features(
        sf, config=AggregationConfig(method="length_weighted"))

    total_tokens = sf["n_tokens"].sum()
    # passive_rate_tokens * total_tokens should equal summed passive_n
    recovered = agg["passive_rate_tokens"] * total_tokens
    assert abs(recovered - agg["passive_n"]) < 1e-6

    # ... and be close to the whole-text rate (boundary tokenization differs slightly)
    whole = extract_linguistic_features(NLP(" ".join(segs)), CFG, NLP)
    assert abs(agg["passive_rate_tokens"] - whole["passive_rate_tokens"]) < 0.05


def test_weights_misalignment_raises():
    segs = ["The valve was opened.", "We closed it."]
    sf = extract_segment_features(segs, CFG, NLP)
    try:
        aggregate_segment_features(
            sf, weights=[0.5], config=AggregationConfig(method="climate_weighted"))
    except ValueError:
        return
    raise AssertionError("misaligned weights did not raise")


def test_climate_weighted_runs():
    segs = ["The valve was opened by the crew.", "We closed it.", "Pressure dropped."]
    sf = extract_segment_features(segs, CFG, NLP)
    agg = aggregate_segment_features(
        sf, weights=[0.9, 0.1, 0.5], config=AggregationConfig(method="climate_weighted"))
    assert "passive_per_sent" in agg
    # top_k=1 keeps only the highest-weight segment
    agg_k = aggregate_segment_features(
        sf, weights=[0.9, 0.1, 0.5], config=AggregationConfig(method="climate_top_k", k=1))
    assert np.isfinite(agg_k["passive_rate_tokens"])


def test_extract_report_independent_mode():
    text = ("The pump was inspected by the crew. The operator reported a leak. "
            "We isolated the line and replaced the seal the following morning.")
    ext = LinguisticFeatureExtractor(CFG)
    rep = ext.extract_report(text)
    assert rep["n_segments"] >= 2
    assert "ttr" in rep  # Tier-2 diversity computed on whole report
    assert np.isfinite(rep["passive_rate_tokens"])


def test_perf_smoke():
    base = ("The operator reported that the valve was opened during the morning "
            "shift and the crew isolated the affected line before the leak spread.")
    segs = [base] * 1000
    ext = LinguisticFeatureExtractor(CFG)
    t0 = time.time()
    sf = ext.extract_segments(segs)
    dt = time.time() - t0
    assert len(sf) == 1000
    assert dt < 30.0, f"too slow: {dt:.1f}s for 1000 segments"


def test_complexity_ignores_short_sentence_fragments():
    """
    A long clause followed by many sub-min_sent_tokens fragments must yield the
    same complexity values as the long clause alone - fragments are skipped for
    ALL complexity measures (dep distance, clause density, subordination), not
    just sentence length.
    """
    main = ("The operator reported that the valve was opened during the morning "
            "shift because the crew had isolated the affected line.")
    cfg = LinguisticConfig(min_tokens_for_complexity=5, min_sent_tokens=4)
    base = complexity_features(NLP(main), min_tokens=5, min_sent_tokens=4)
    withfrag = complexity_features(NLP(main + " OK. Yes. No."), min_tokens=5, min_sent_tokens=4)
    for k in ("mean_dep_distance", "clause_density", "subord_rate", "mean_sent_len_tokens"):
        assert np.isclose(base[k], withfrag[k], equal_nan=True), f"{k} changed with fragments"


def test_standardize_composites_corpus_zscored():
    """
    Corpus composites z-score each component across reports, so analytical_thinking
    and clout are mean~0 over the corpus and a high-`we` report scores above a
    high-`i` report on clout.
    """
    rows = [
        extract_linguistic_features(NLP("I think I saw that I was wrong about it."), CFG, NLP),
        extract_linguistic_features(NLP("We inspected the valve and you closed the line."), CFG, NLP),
        extract_linguistic_features(NLP("The operator opened the valve at the station."), CFG, NLP),
    ]
    df = pd.DataFrame(rows)
    out = standardize_composites(df)
    assert "clout" in out.columns and "analytical_thinking" in out.columns
    assert abs(np.nanmean(out["clout"])) < 1e-9          # z-scored -> mean 0
    assert out["clout"].iloc[1] > out["clout"].iloc[0]   # we/you-heavy > i-heavy


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        t0 = time.time()
        fn()
        print(f"  PASS {fn.__name__} ({time.time()-t0:.2f}s)")
        passed += 1
    print(f"\n{passed}/{len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()
