"""
compute_temporal_features.py
----------------------------
Per-report temporal-dynamics features for the lean substitution-windowing rail.

For each organization's ordered event sequence we slide a fixed-COUNT window
(step=1) and compute burstiness / memory / inter-event pace on the trailing
window, anchoring the result to the window's most-recent event id. The output
is one row per event (keyed by `eid`), structurally identical to
compute_linguistic_features.py, so temporal features enter the panel models
through the SAME per-report -> substitution -> windowing path. Config-
INDEPENDENT: depends only on event timing, not on any climate config.

Features (trailing window of `window` events, default 20):
  temp__burstiness    - Goh-Barabasi B = (CV-1)/(CV+1) of inter-event times
  temp__memory        - lag-1 memory coefficient of inter-event times
  temp__cv            - coefficient of variation of inter-event times
  temp__log_mean_iet  - log mean inter-event time (recent pace), days

Gotcha handled: rolling_burstiness_by_count() sorts timestamps internally but
does NOT reorder event_ids, so we pre-sort per org and pass sort=False with
aligned timestamp/eid lists.

Usage:
    python -m dynclim.features.temporal.compute_temporal_features --industry rail
    python -m dynclim.features.temporal.compute_temporal_features --industry msha --window 20
"""
from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from dynclim.features.temporal.temporal_features import rolling_burstiness_by_count

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TempSpec:
    events_path: str
    org_col: str
    time_col: str
    eid_col: str = "eid"
    commodity_filter: Optional[str] = None  # msha: restrict to 'coal' (panel universe)
    note: str = ""


# rail/msha have a clean org id on each event. (aviation's org = airport via a
# many-to-one spatial map; deferred.)
SPECS = {
    "rail": TempSpec("dynclim/data/processed/rail/events.parquet", "org_id", "event_date",
                     note="FRA; org_id per event"),
    "msha": TempSpec("dynclim/data/processed/msha/events.parquet", "mine_id", "accident_date",
                     commodity_filter="coal", note="MSHA coal; mine_id per event"),
}

OUT_COLS = ["eid", "temp__burstiness_goh", "temp__burstiness_kimjo",
            "temp__memory", "temp__cv", "temp__log_mean_iet"]


def _org_temporal(ts: List, eids: List, window: int, unit: str) -> Optional[pd.DataFrame]:
    """
    Rolling-by-count temporal features for one org's pre-sorted event seq.
    Burstiness is emitted in TWO forms: the Goh-Barabasi B (biased toward 0 for
    short sequences) and the Kim-Jo (2016) open-boundary finite-size-corrected
    A_n(r), which is the more appropriate estimator at our ~20-event window.
    Both use identical windows (same step / min_valid), so rows align by position.
    """
    rb = rolling_burstiness_by_count(
        list(ts), window_size=window, step=1, unit=unit, method="goh",
        compute_memory=True, sort=False, event_ids=list(eids))
    if rb.empty or "event_id" not in rb.columns:
        return None
    rk = rolling_burstiness_by_count(
        list(ts), window_size=window, step=1, unit=unit, method="kimjo2016_open",
        compute_memory=False, sort=False, event_ids=list(eids))
    mean_dt = rb["mean_dt"].astype(float)
    return pd.DataFrame({
        "eid": rb["event_id"].astype(str),
        "temp__burstiness_goh": rb["b"].astype(float),
        "temp__burstiness_kimjo": rk["b"].astype(float).to_numpy(),  # same window order
        "temp__memory": rb["M"].astype(float),
        "temp__cv": rb["cv"].astype(float),
        "temp__log_mean_iet": np.log(mean_dt.where(mean_dt > 0)),
    })


def compute(industry: str, window: int = 20, unit: str = "d") -> pd.DataFrame:
    if industry not in SPECS:
        raise ValueError(f"unknown industry {industry!r}; known: {sorted(SPECS)}")
    spec = SPECS[industry]
    cols = {spec.org_col, spec.time_col, spec.eid_col}
    if spec.commodity_filter:
        cols.add("commodity")
    df = pd.read_parquet(spec.events_path, columns=list(cols))

    df[spec.eid_col] = df[spec.eid_col].astype(str)
    df = df.dropna(subset=[spec.org_col, spec.time_col, spec.eid_col])
    df[spec.org_col] = df[spec.org_col].astype(str)
    df[spec.time_col] = pd.to_datetime(df[spec.time_col], errors="coerce")
    df = df.dropna(subset=[spec.time_col])
    n_all = len(df)
    if spec.commodity_filter:
        df = df[df["commodity"].astype(str).str.strip().str.lower() == spec.commodity_filter]
        logger.info("[%s] commodity=%s: %d/%d events", industry, spec.commodity_filter, len(df), n_all)

    df = df.sort_values([spec.org_col, spec.time_col], kind="mergesort")

    parts, n_orgs_used, n_orgs_skip = [], 0, 0
    t0 = time.time()
    for org, g in df.groupby(spec.org_col, sort=False):
        if len(g) < window:
            n_orgs_skip += 1
            continue
        res = _org_temporal(g[spec.time_col].tolist(), g[spec.eid_col].tolist(), window, unit)
        if res is not None and len(res):
            parts.append(res)
            n_orgs_used += 1

    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=OUT_COLS)
    # Each (org, trailing event) is unique under step=1; guard against dup eids anyway.
    if out["eid"].duplicated().any():
        logger.warning("[%s] %d duplicate eids - keeping last", industry, int(out["eid"].duplicated().sum()))
        out = out.drop_duplicates("eid", keep="last")
    logger.info("[%s] %d reports with temporal features across %d orgs "
                "(%d orgs < window=%d skipped) in %.1fs",
                industry, len(out), n_orgs_used, n_orgs_skip, window, time.time() - t0)
    return out


def run(industry: str, window: int = 20, output_dir: Optional[str] = None, save: bool = True) -> pd.DataFrame:
    spec = SPECS[industry]
    logger.info("[%s] %s (window=%d events)", industry, spec.note, window)
    out = compute(industry, window=window)
    if save:
        out_dir = Path(output_dir or Path(spec.events_path).parent)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"temporal_features_{industry}.parquet"
        out.to_parquet(out_path, index=False)
        logger.info("[%s] wrote %s (%d rows x %d cols)", industry, out_path, out.shape[0], out.shape[1])
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Emit per-report temporal-dynamics features keyed by eid.")
    p.add_argument("--industry", "-i", required=True, choices=sorted(SPECS))
    p.add_argument("--window", type=int, default=20, help="trailing event-count window (default 20)")
    p.add_argument("--output", "-o", help="output dir (default: events dir)")
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--summary", "-s", action="store_true")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    out = run(args.industry, window=args.window, output_dir=args.output, save=not args.no_save)
    if args.summary:
        import numpy as _np
        print(f"\n{len(out)} reports")
        for c in OUT_COLS[1:]:
            print(f"  {c:24s} mean={_np.nanmean(out[c]):+.3f}  nan%={100*out[c].isna().mean():.0f}")


if __name__ == "__main__":
    main()
