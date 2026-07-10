"""
Golden-output generator for scoring-path equivalence testing
=============================================================

Captures the CURRENT (paper-1, `paper1-submission`-tagged lineage) scoring
path's exact outputs on a fixed report sample so that any optimized
re-implementation can be verified to reproduce them. Run ONCE on the
unmodified scoring code; the goldens are then committed and treated as
frozen truth.

Sample = 300 real OSHA ITA narratives (fixed seed; text saved WITH the
goldens so regeneration never depends on the events parquet) + synthetic
edge cases (empty, single-token, punctuation-only, unicode, duplicated
sentences, very long).

Configs = the 18 seg/threshold/composite combos the ExperimentRunner
crosses for production runs (MiniLM encoder, vader sentiment): top_k
k∈{3,5} × 12 post_threshold composites + sentinel top_k(99999) × 6
apply_over='all' composites.

Outputs (dynclim/features/safety/scoring_goldens/):
    reports.parquet   eid, text
    goldens.parquet   config_label, eid, item_key, final_score,
                      sent_signal, strength_signal, n_segments
                      + domain rows (item_key='domain::<name>', final only)

Usage:
    python dynclim/features/safety/gen_scoring_goldens.py
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent / "scoring_goldens"

EDGE_CASES = [
    ("edge_empty", ""),
    ("edge_whitespace", "   \n\t  "),
    ("edge_one_token", "Fell."),
    ("edge_punct_only", "... !!! ???"),
    ("edge_single_sentence", "Employee slipped on wet floor and injured left knee."),
    ("edge_two_identical",
     "Worker was struck by falling pipe. Worker was struck by falling pipe."),
    ("edge_unicode",
     "Employee's hand was caught — crushed between café door and jamb; 3° burn."),
    ("edge_long", " ".join(
        f"Employee number {i} reported feeling unsafe near the conveyor belt "
        f"on level {i % 7} and supervisor was notified immediately."
        for i in range(120))),
    ("edge_redacted",
     "Cleaning the [REDACTED] roller. Caught right hand in the [REDACTED]. "
     "Amputation of two fingers. Machinery."),
]


def build_config_grid():
    """The 18-combo axis exactly as ExperimentRunner crosses it."""
    from dynclim.features.safety import CompositeConfig, ThresholdConfig

    post_composites = [
        ("wa_0.5_post", CompositeConfig(method="weighted_average",
                                        similarity_weight=0.5,
                                        sentiment_weight=0.5,
                                        apply_over="post_threshold")),
        ("att_t0.15_post", CompositeConfig(method="attention_weight",
                                           temperature=0.15,
                                           apply_over="post_threshold")),
        ("att_t1.0_post", CompositeConfig(method="attention_weight",
                                          temperature=1.00,
                                          apply_over="post_threshold")),
        ("pow_p0.5_post", CompositeConfig(method="power_attention",
                                          power=0.5,
                                          apply_over="post_threshold")),
        ("pow_p2.0_post", CompositeConfig(method="power_attention",
                                          power=2.0,
                                          apply_over="post_threshold")),
        ("wa_0.5_all", None),  # placeholder replaced below
    ]
    # threshold-independent ('all') composites use the sentinel threshold
    all_composites = [
        ("wa_0.5_all", CompositeConfig(method="weighted_average",
                                       similarity_weight=0.5,
                                       sentiment_weight=0.5,
                                       apply_over="all")),
        ("att_t0.15_all", CompositeConfig(method="attention_weight",
                                          temperature=0.15, apply_over="all")),
        ("att_t1.0_all", CompositeConfig(method="attention_weight",
                                         temperature=1.00, apply_over="all")),
        ("pow_p0.5_all", CompositeConfig(method="power_attention",
                                         power=0.5, apply_over="all")),
        ("pow_p2.0_all", CompositeConfig(method="power_attention",
                                         power=2.0, apply_over="all")),
    ]
    post_composites = [pc for pc in post_composites if pc[1] is not None]

    thresholds = [("k3", ThresholdConfig(method="top_k", k=3)),
                  ("k5", ThresholdConfig(method="top_k", k=5))]
    sentinel = ("kall", ThresholdConfig(method="top_k", k=99999))

    grid = []
    for tl, tc in thresholds:
        for cl, cc in post_composites:
            grid.append((f"{tl}__{cl}", tc, cc))
    for cl, cc in all_composites:
        grid.append((f"{sentinel[0]}__{cl}", sentinel[1], cc))
    return grid


def main() -> None:
    from dynclim.features.safety import SegmentationConfig
    from dynclim.features.safety.unified_processor import Processor
    from dynclim.utils import loadDomainsFromYaml

    OUT_DIR.mkdir(exist_ok=True)

    ev = pd.read_parquet(
        REPO / "dynclim/data/processed/osha/events.parquet",
        columns=["eid", "narrative"]).sample(300, random_state=20260710)
    reports = pd.DataFrame({
        "eid": list(ev["eid"].astype(str)) + [e for e, _ in EDGE_CASES],
        "text": list(ev["narrative"].astype(str)) + [t for _, t in EDGE_CASES],
    })
    reports.to_parquet(OUT_DIR / "reports.parquet", index=False)

    domain_definitions = loadDomainsFromYaml(
        REPO / "config/domains/safety_domains.yaml")
    proc = Processor(
        domain_definitions=domain_definitions,
        embedding_model_name="all-MiniLM-L6-v2",
        use_acceleration=True,
        sentiment_method="vader",
    )
    # Wire the segmentation store + tokenizer exactly as ExperimentRunner does
    # (runner.py: processor.store = self.store; processor.tok = self.tok).
    from dynclim.features.safety.seg_core import SegmentationStore
    from tokenizers import Tokenizer
    proc.store = SegmentationStore(str(OUT_DIR / "seg_cache_goldens.sqlite"))
    proc.tok = Tokenizer.from_pretrained("bert-base-uncased")
    seg_cfg = SegmentationConfig(method="sentence", min_length=5,
                                 backend="auto")
    report_data = list(zip(reports["eid"], reports["text"]))

    rows = []
    for label, thresh_cfg, comp_cfg in build_config_grid():
        print(f"--- {label}")
        # _process_sequential is the exact per-report path the runner drives
        # (process_reports -> _process_sequential); we call it directly to get
        # the typed ReportFeatureResult objects rather than the DataFrame.
        results = proc._process_sequential(report_data, seg_cfg, thresh_cfg,
                                           comp_cfg)
        for r in results:
            for item_key, ir in (r.item_results or {}).items():
                rows.append({
                    "config_label": label,
                    "eid": r.report_id,
                    "item_key": item_key,
                    "final_score": ir.final_score,
                    "sent_signal": ir.sent_signal,
                    "strength_signal": ir.strength_signal,
                    "n_segments": ir.n_segments,
                })
            for dom, score in (r.domain_scores or {}).items():
                rows.append({
                    "config_label": label,
                    "eid": r.report_id,
                    "item_key": f"domain::{dom}",
                    "final_score": score,
                    "sent_signal": float("nan"),
                    "strength_signal": float("nan"),
                    "n_segments": -1,
                })

    golden = pd.DataFrame(rows)
    golden.to_parquet(OUT_DIR / "goldens.parquet", index=False)
    print(f"wrote {len(golden):,} golden rows for "
          f"{golden['config_label'].nunique()} configs "
          f"x {golden['eid'].nunique()} reports -> {OUT_DIR}")


if __name__ == "__main__":
    main()
