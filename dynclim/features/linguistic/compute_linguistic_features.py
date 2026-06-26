"""
compute_linguistic_features.py
------------------------------
Driver that runs the linguistic feature extractor over a processed events
parquet and emits one per-report feature parquet keyed by report id (`eid`),
with columns namespaced `ling__<family>__<feature>` for clean R-side joins.

This is the lexical/syntactic analog of the R-side
`9a_compute_whole_narrative_vader.py` template: compute a per-report value,
key it by event id, hand it to the R substitution-windowing trick (drop the
column into the climate results parquet, re-run panel prep, window it exactly
like the climate score). See HANDOFF_feature_layers.md.

Whole-narrative extraction, single parse per report
---------------------------------------------------
Each report is parsed ONCE (`nlp.pipe` batching) and all families - including
Tier-2 diversity (TTR/MATTR) - are computed on that one Doc. This is the
efficient path for per-report features; it deliberately does NOT use
`LinguisticFeatureExtractor.extract_report`, which double-parses (segments +
whole-report diversity) and only matters for the climate-weighted segment
aggregation path.

Pennebaker composites are computed at CORPUS level via `standardize_composites`
(z-score components across all reports, then combine) rather than per-report,
which is the only scale at which "standardized combination" is meaningful.

Data caveats (see project memory / preprocessing audit)
-------------------------------------------------------
- aviation (ASRS): clean mixed-case. Primary validation target.
- rail (FRA): ~all-caps in source; tagger/parser accuracy degrades. Rate-style
  features remain usable; treat syntactic-complexity features with caution.
- msha: mixed-case injury narratives, often short; complexity/diversity will be
  NaN for sub-floor reports (expected, not an error).
- nrc: NOT supported here - stored narratives are lowercased + word-glued at
  scrape time and must be re-scraped before syntactic features are valid.

Usage
-----
    python -m dynclim.features.linguistic.compute_linguistic_features \\
        --industry aviation
    python -m dynclim.features.linguistic.compute_linguistic_features \\
        --industry msha --batch-size 128 --n-process 4
    # quick smoke on a subset:
    python -m dynclim.features.linguistic.compute_linguistic_features \\
        --industry rail --limit 500 --no-save --summary
"""
from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from dynclim.features.linguistic.linguistic_features import (
    LinguisticConfig,
    extract_linguistic_features,
    namespace_features,
    standardize_composites,
    _load_nlp,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IndustrySpec:
    """Where an industry's narratives live and how to read them."""
    events_path: str
    narrative_col: str
    id_col: str = "eid"
    all_caps: bool = False
    note: str = ""


# Defaults resolve against dynclim/data/processed/<industry>/. nrc is omitted
# on purpose (text quality); add it once narratives are re-scraped with case
# and word-boundary fixes.
INDUSTRY_SPECS: Dict[str, IndustrySpec] = {
    "aviation": IndustrySpec(
        "dynclim/data/processed/aviation/events.parquet", "narrative",
        note="ASRS, clean mixed-case",
    ),
    "rail": IndustrySpec(
        "dynclim/data/processed/rail/events.parquet", "narrative",
        all_caps=True, note="FRA, mostly ALL-CAPS - parser accuracy degrades",
    ),
    "msha": IndustrySpec(
        "dynclim/data/processed/msha/events.parquet", "narrative",
        note="MSHA mining injury narratives, often short",
    ),
}


def compute_features(
    df: pd.DataFrame,
    spec: IndustrySpec,
    config: Optional[LinguisticConfig] = None,
    add_composites: bool = True,
    batch_size: int = 64,
    n_process: int = 1,
    log_every: int = 5000,
    prefix: str = "ling",
) -> pd.DataFrame:
    """
    Compute per-report linguistic features for the rows of `df`.

    Returns a DataFrame with the id column (as str) plus `<prefix>__...`
    namespaced feature columns - one row per input row, same order. Rows whose
    narrative is empty/degenerate get an all-NaN feature row (never dropped, so
    the join back to the event panel stays complete).
    """
    # Composites are computed at corpus level below, never per-report; force the
    # per-doc path off so extract_linguistic_features doesn't emit raw versions.
    base = config or LinguisticConfig()
    cfg = LinguisticConfig(**{**base.__dict__, "compute_composites": False})

    nlp = _load_nlp(cfg.spacy_model)
    texts = df[spec.narrative_col].fillna("").astype(str).tolist()

    rows: List[Dict[str, float]] = []
    t0 = time.time()
    for i, doc in enumerate(nlp.pipe(texts, batch_size=batch_size, n_process=n_process)):
        rows.append(extract_linguistic_features(doc, cfg, nlp))
        if log_every and (i + 1) % log_every == 0:
            rate = (i + 1) / (time.time() - t0)
            logger.info("  %d/%d reports (%.0f/s)", i + 1, len(texts), rate)

    feats = pd.DataFrame(rows, index=df.index)

    # Corpus-level standardized Pennebaker composites (Family F). Done here, on
    # the raw style-rate columns, before namespacing.
    if add_composites:
        feats = standardize_composites(feats)

    feats = feats.rename(columns=_namespace_map(feats.columns, prefix))

    out = pd.DataFrame({spec.id_col: df[spec.id_col].astype(str).values})
    out = pd.concat([out.reset_index(drop=True), feats.reset_index(drop=True)], axis=1)
    logger.info("computed %d features for %d reports in %.1fs (%.1f/s)",
                feats.shape[1], len(out), time.time() - t0,
                len(out) / max(time.time() - t0, 1e-9))
    return out


def _namespace_map(columns, prefix: str) -> Dict[str, str]:
    """Build a column-rename map using the module's family namespacing."""
    probe = {c: 0.0 for c in columns}
    renamed = namespace_features(probe, prefix=prefix)
    # namespace_features preserves insertion order, so zip is positional-safe.
    return dict(zip(probe.keys(), renamed.keys()))


def run(
    industry: str,
    events_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    config: Optional[LinguisticConfig] = None,
    add_composites: bool = True,
    batch_size: int = 64,
    n_process: int = 1,
    limit: Optional[int] = None,
    save: bool = True,
) -> pd.DataFrame:
    """Load an industry's events, compute features, and (optionally) save."""
    if industry not in INDUSTRY_SPECS:
        raise ValueError(
            f"unknown industry {industry!r}; known: {sorted(INDUSTRY_SPECS)}. "
            "nrc is intentionally excluded until narratives are re-scraped."
        )
    spec = INDUSTRY_SPECS[industry]
    path = events_path or spec.events_path

    logger.info("[%s] %s", industry, spec.note)
    if spec.all_caps:
        logger.warning("[%s] narratives are ALL-CAPS; syntactic-complexity and "
                       "proper-noun-sensitive features are less reliable.", industry)

    df = pd.read_parquet(path, columns=[spec.id_col, spec.narrative_col])
    n_raw = len(df)
    # Keep rows with usable text; report (don't silently drop) what's excluded.
    has_text = df[spec.narrative_col].fillna("").astype(str).str.strip().ne("")
    df = df[has_text].reset_index(drop=True)
    logger.info("[%s] %d/%d rows have non-empty narrative", industry, len(df), n_raw)

    dup = df[spec.id_col].duplicated().sum()
    if dup:
        logger.warning("[%s] %d duplicate %s values - join key is not unique.",
                       industry, dup, spec.id_col)

    if limit:
        df = df.head(limit)
        logger.info("[%s] limited to first %d rows", industry, len(df))

    out = compute_features(
        df, spec, config=config, add_composites=add_composites,
        batch_size=batch_size, n_process=n_process,
    )

    if save:
        out_dir = Path(output_dir or Path(path).parent)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"linguistic_features_{industry}.parquet"
        out.to_parquet(out_path, index=False)
        logger.info("[%s] wrote %s (%d rows x %d cols)",
                    industry, out_path, out.shape[0], out.shape[1])
    return out


def _summary(out: pd.DataFrame, id_col: str) -> None:
    feat_cols = [c for c in out.columns if c != id_col]
    print("\n" + "=" * 60)
    print(f"SUMMARY  ({len(out)} reports, {len(feat_cols)} feature columns)")
    print("=" * 60)
    highlights = [c for c in feat_cols if any(
        k in c for k in ("passive__passive_per_sent", "tense__past_rate",
                         "style__pron_i_rate", "diversity__mattr",
                         "complexity__mean_dep_distance", "composite__"))]
    for c in highlights:
        col = out[c]
        print(f"  {c:48s} mean={np.nanmean(col):+.3f}  nan%={100*col.isna().mean():4.0f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute per-report lexical/syntactic features and emit a "
                    "ling__-namespaced parquet keyed by event id.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Industries: " + ", ".join(sorted(INDUSTRY_SPECS)) +
               "  (nrc excluded: needs re-scrape).",
    )
    parser.add_argument("--industry", "-i", required=True,
                        choices=sorted(INDUSTRY_SPECS))
    parser.add_argument("--events", "-e", help="Override events parquet path")
    parser.add_argument("--output", "-o", help="Output dir (default: events dir)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-process", type=int, default=1,
                        help="spaCy n_process; keep 1 inside a parallel runner")
    parser.add_argument("--limit", type=int, help="Process only first N rows")
    parser.add_argument("--no-composites", action="store_true",
                        help="Skip corpus-standardized Pennebaker composites")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--summary", "-s", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    out = run(
        industry=args.industry,
        events_path=args.events,
        output_dir=args.output,
        add_composites=not args.no_composites,
        batch_size=args.batch_size,
        n_process=args.n_process,
        limit=args.limit,
        save=not args.no_save,
    )

    if args.summary:
        _summary(out, INDUSTRY_SPECS[args.industry].id_col)


if __name__ == "__main__":
    main()
