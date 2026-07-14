"""
Equivalence test: vectorized scorer vs frozen per-report goldens
================================================================

Verifies that vectorized_scorer reproduces the exact outputs of the
paper-1-lineage per-report scoring path (captured in scoring_goldens/ by
gen_scoring_goldens.py) on 309 fixed reports x 15 production configs.

History — power_attention arms were originally excluded from the
comparison. Investigation (2026-07-10) found that torch 2.0.1's
`nanquantile` returns incorrect values on MPS for any multi-column tensor.
PowerAttentionScorer's GPU path used it for the attn_quantile prefilter (a
q=0.0 no-op by spec), so every power_attention config scored on this
machine's MPS applied an arbitrary, garbage per-item similarity threshold
before weighting. Fixed 2026-07-13: compute_composite_gpu short-circuits
q<=0 and computes any real quantile on CPU, so the legacy path now
implements the SPEC on every device. The goldens were regenerated against
the fixed path and ALL 15 configs are compared (rtol/atol allow only
float-noise at top-k tie boundaries). Pre-fix goldens capturing the
corrupted power behavior remain in git history and at the
`paper1-submission` tag.

Run:
    .dynClimVenv/bin/python dynclim/features/safety/test_scoring_equivalence.py
    (or via pytest)
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
GOLD_DIR = HERE / "scoring_goldens"

RTOL, ATOL = 1e-4, 1e-6


def _build_processor():
    from dynclim.features.safety.seg_core import SegmentationStore
    from dynclim.features.safety.unified_processor import Processor
    from dynclim.utils import loadDomainsFromYaml
    from tokenizers import Tokenizer

    repo = HERE.parents[2]
    proc = Processor(
        domain_definitions=loadDomainsFromYaml(
            repo / "config/domains/safety_domains.yaml"),
        embedding_model_name="all-MiniLM-L6-v2",
        use_acceleration=True,
        sentiment_method="vader",
    )
    proc.store = SegmentationStore(str(GOLD_DIR / "seg_cache_goldens.sqlite"))
    proc.tok = Tokenizer.from_pretrained("bert-base-uncased")
    return proc


def test_vectorized_matches_goldens():
    from dynclim.features.safety import SegmentationConfig
    from dynclim.features.safety.gen_scoring_goldens import build_config_grid
    from dynclim.features.safety.vectorized_scorer import (
        prepare_corpus, score_configs)

    grid = build_config_grid()
    labels = {g[0] for g in grid}

    golden = pd.read_parquet(GOLD_DIR / "goldens.parquet")
    golden = golden[golden["config_label"].isin(labels)].reset_index(drop=True)
    reports = pd.read_parquet(GOLD_DIR / "reports.parquet")

    proc = _build_processor()
    seg_cfg = SegmentationConfig(method="sentence", min_length=5,
                                 backend="auto")
    corpus = prepare_corpus(proc, list(zip(reports["eid"], reports["text"])),
                            seg_cfg)
    out = score_configs(proc, corpus, grid)

    key = ["config_label", "eid", "item_key"]
    merged = golden.merge(out, on=key, how="outer", suffixes=("_g", "_v"),
                          indicator=True)
    missing = merged[merged["_merge"] != "both"]
    assert len(missing) == 0, (
        f"{len(missing)} rows differ in coverage:\n"
        f"{missing[key + ['_merge']].head(10)}")

    items = merged[~merged["item_key"].str.startswith("domain::")]
    doms = merged[merged["item_key"].str.startswith("domain::")]

    report = {}
    for col in ("final_score", "sent_signal", "strength_signal"):
        g, v = items[f"{col}_g"].to_numpy(), items[f"{col}_v"].to_numpy()
        report[col] = np.nanmax(np.abs(g - v)) if len(g) else 0.0
        assert np.allclose(g, v, rtol=RTOL, atol=ATOL, equal_nan=True), (
            f"{col}: max abs diff {report[col]:.2e} exceeds tolerance; "
            f"worst rows:\n"
            f"{items.assign(d=np.abs(g - v)).nlargest(5, 'd')[key + [f'{col}_g', f'{col}_v', 'd']]}")

    g, v = doms["final_score_g"].to_numpy(), doms["final_score_v"].to_numpy()
    report["domain_final"] = np.nanmax(np.abs(g - v)) if len(g) else 0.0
    assert np.allclose(g, v, rtol=RTOL, atol=ATOL), (
        f"domain final: max abs diff {report['domain_final']:.2e}")

    ns_mismatch = (items["n_segments_g"].astype(int)
                   != items["n_segments_v"].astype(int))
    assert ns_mismatch.sum() == 0, (
        f"{ns_mismatch.sum()} n_segments mismatches:\n"
        f"{items[ns_mismatch][key + ['n_segments_g', 'n_segments_v']].head(10)}")

    print(f"EQUIVALENCE OK — {len(merged):,} rows, "
          f"{golden['config_label'].nunique()} configs, "
          f"{golden['eid'].nunique()} reports")
    for k, m in report.items():
        print(f"  max |Δ| {k}: {m:.2e}")


if __name__ == "__main__":
    test_vectorized_matches_goldens()
