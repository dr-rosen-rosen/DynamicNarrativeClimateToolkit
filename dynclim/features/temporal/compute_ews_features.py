"""
compute_ews_features.py
-----------------------
Per-report early-warning-signal (EWS) features on the safety-CLIMATE score
series, for the lean substitution-windowing rail.

EWS (critical-slowing-down) indicators describe whether a *time series* is
destabilizing (rising variance / autocorrelation, slowing recovery). Here the
series is each organization's ordered sequence of per-report climate scores
(`overall_final_score`) FROM A CHOSEN CONFIG (the champion for the cell being
tested). For each org we slide a fixed-COUNT window (step=1) over the score
series and anchor each window's indicators to its most-recent event id, giving
one row per report keyed by `eid` -- same shape as the linguistic/temporal
emitters, so EWS enters models through the same per-report -> substitution ->
windowing path.

CONFIG-DEPENDENT: EWS is a property of a specific config's climate signal, so we
emit one parquet per (industry, config_id): ews_features_<industry>_cfg<cid>.parquet.
No outcome leakage (the series is climate, not the outcome).

Features (trailing window of `window` reports, default 20; linear-detrended):
  ews__variance, ews__std, ews__ar1, ews__acf1, ews__return_rate, ews__skew

Gotcha handled: rolling_or_expanding_ews() accepts event_ids but does NOT attach
them, and skips windows with < 8 obs. We pre-sort per org and map each window's
end_idx (right anchor) back to its eid ourselves.

Usage:
    python -m dynclim.features.temporal.compute_ews_features --industry rail --config-id 127
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

from dynclim.features.temporal.ews import rolling_or_expanding_ews, EWSConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EwsSpec:
    events_path: str
    checkpoint_dir: str          # holds _cfg/<cid>/results.parquet
    org_col: str
    time_col: str
    eid_col: str = "eid"
    commodity_filter: Optional[str] = None


_CKPT = "dynclim/notebooks/checkpoints"
SPECS = {
    "rail": EwsSpec("dynclim/data/processed/rail/events.parquet",
                    f"{_CKPT}/rail_04-14-2026", "org_id", "event_date"),
    "msha": EwsSpec("dynclim/data/processed/msha/events.parquet",
                    f"{_CKPT}/msha_06-01-2026", "mine_id", "accident_date",
                    commodity_filter="coal"),
}

# EWS indicators. The base set is stable on short windows; the "full" set adds
# DFA-alpha + spectral measures, which need a LONGER window to estimate (DFA
# returns NaN below ~3 usable scales, i.e. roughly window < 32; use window>=64,
# ideally 100+).
_KEEP_BASE = ["variance", "std", "ar1", "acf1", "return_rate", "skew"]
_KEEP_FULL = _KEEP_BASE + ["dfa_alpha", "spectral_slope", "spectral_ratio"]


def _ews_cfg(full: bool) -> EWSConfig:
    return EWSConfig(detrend="linear",
                     compute_dfa_alpha=full, compute_spectral_slope=full,
                     compute_spectral_ratio=full,
                     compute_mean=False, compute_cv=False, compute_kurtosis=False)


def _load_events(spec: EwsSpec) -> pd.DataFrame:
    cols = {spec.org_col, spec.time_col, spec.eid_col}
    if spec.commodity_filter:
        cols.add("commodity")
    df = pd.read_parquet(spec.events_path, columns=list(cols))
    df[spec.eid_col] = df[spec.eid_col].astype(str)
    df = df.dropna(subset=[spec.org_col, spec.time_col, spec.eid_col])
    df[spec.org_col] = df[spec.org_col].astype(str)
    df[spec.time_col] = pd.to_datetime(df[spec.time_col], errors="coerce")
    df = df.dropna(subset=[spec.time_col])
    if spec.commodity_filter:
        df = df[df["commodity"].astype(str).str.strip().str.lower() == spec.commodity_filter]
    return df.rename(columns={spec.org_col: "org", spec.time_col: "time", spec.eid_col: "eid"})


def compute(industry: str, config_id, window: int = 20, full: bool = False) -> pd.DataFrame:
    if industry not in SPECS:
        raise ValueError(f"unknown industry {industry!r}; known: {sorted(SPECS)}")
    spec = SPECS[industry]
    keep_set = _KEEP_FULL if full else _KEEP_BASE
    cfg = _ews_cfg(full)
    ev = _load_events(spec)[["org", "time", "eid"]]

    src = Path(spec.checkpoint_dir) / "_cfg" / str(config_id) / "results.parquet"
    scores = pd.read_parquet(src, columns=["report_id", "overall_final_score"])
    scores = scores.assign(eid=scores["report_id"].astype(str))[["eid", "overall_final_score"]]

    m = ev.merge(scores, on="eid", how="inner").dropna(subset=["overall_final_score"])
    m = m.sort_values(["org", "time"], kind="mergesort")
    logger.info("[%s cfg=%s w=%d full=%s] %d scored events join events",
                industry, config_id, window, full, len(m))

    parts, n_used = [], 0
    t0 = time.time()
    for org, g in m.groupby("org", sort=False):
        if len(g) < window:
            continue
        r = rolling_or_expanding_ews(
            g["overall_final_score"].tolist(), g["time"].tolist(),
            window_size=window, step=1, mode="rolling", cfg=cfg)
        if r.empty:
            continue
        eids = g["eid"].tolist()
        r = r.assign(eid=[eids[i] for i in r["end_idx"].astype(int)])
        keep = [c for c in keep_set if c in r.columns]
        sub = r[["eid"] + keep].rename(columns={c: f"ews__{c}" for c in keep})
        parts.append(sub)
        n_used += 1

    cols = ["eid"] + [f"ews__{c}" for c in keep_set]
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)
    if out["eid"].duplicated().any():
        out = out.drop_duplicates("eid", keep="last")
    logger.info("[%s cfg=%s w=%d] %d reports with EWS across %d orgs in %.1fs",
                industry, config_id, window, len(out), n_used, time.time() - t0)
    return out


def run(industry: str, config_id, window: int = 20, full: bool = False,
        output_dir: Optional[str] = None, save: bool = True) -> pd.DataFrame:
    spec = SPECS[industry]
    out = compute(industry, config_id, window=window, full=full)
    if save:
        out_dir = Path(output_dir or Path(spec.events_path).parent)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"ews_features_{industry}_cfg{config_id}_w{window}.parquet"
        out.to_parquet(out_path, index=False)
        logger.info("[%s cfg=%s w=%d] wrote %s (%d rows x %d cols)",
                    industry, config_id, window, out_path, out.shape[0], out.shape[1])
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Emit per-report EWS features on a config's climate-score series, keyed by eid.")
    p.add_argument("--industry", "-i", required=True, choices=sorted(SPECS))
    p.add_argument("--config-id", "-c", required=True, help="climate config_id whose per-report scores form the series")
    p.add_argument("--window", type=int, default=20, help="trailing report-count window (default 20; use >=64 for --full)")
    p.add_argument("--full", action="store_true", help="add DFA-alpha + spectral measures (need window>=64)")
    p.add_argument("--output", "-o")
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--summary", "-s", action="store_true")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    out = run(args.industry, args.config_id, window=args.window, full=args.full,
              output_dir=args.output, save=not args.no_save)
    if args.summary:
        keep = _KEEP_FULL if args.full else _KEEP_BASE
        print(f"\n{len(out)} reports")
        for c in [f"ews__{k}" for k in keep]:
            if c in out:
                print(f"  {c:20s} mean={np.nanmean(out[c]):+.4g}  nan%={100*out[c].isna().mean():.0f}")


if __name__ == "__main__":
    main()
