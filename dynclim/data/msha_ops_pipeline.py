"""
MSHA Exposure / Offset Pipeline (employee-hours & coal tonnage)
===============================================================

Builds the exposure denominator for mine-quarter incidence-rate models from the
MSHA Open-Government employment/production sets. Employee-hours give the
industry-standard per-200,000-hours offset; coal tonnage is an alternate
exposure (coal mines only — MNM operators are not required to report production).

Sources (auto-downloaded via :mod:`dynclim.data.msha_sources`):
    MinesProdQuarterly.zip → quarterly hours/production by subunit (2000+)
    MinesProdYearly.zip    → annual rollup, abbreviated headers (backfill)

The quarterly headers match the documentation; the **yearly** file uses
abbreviated headers (CALENDAR_YR, ANNUAL_HRS, ANNUAL_COAL_PROD, AVG_ANNUAL_EMPL,
C_M_IND) — mapped explicitly below.

Outputs (keyed for direct join to events / covariate parquets):
    ops_subunit_quarter.parquet   MINE_ID + CAL_YR + CAL_QTR + SUBUNIT_CD
    ops_mine_quarter.parquet      MINE_ID + CAL_YR + CAL_QTR (subunits summed),
                                  with log_hours / log_tons offsets and a
                                  Mundlak between/within decomposition of
                                  log-hours over each mine's quarterly history.

CLI:
    python -m dynclim.data.msha_ops_pipeline --output data/processed/msha --summary
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd

from dynclim.data.msha_sources import read_dataset

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

COMMODITY_MAP: Dict[str, str] = {"C": "coal", "M": "mnm"}

# Yearly file → canonical column names
YEARLY_RENAME: Dict[str, str] = {
    "CALENDAR_YR":      "CAL_YR",
    "ANNUAL_HRS":       "HOURS_WORKED",
    "ANNUAL_COAL_PROD": "COAL_PRODUCTION",
    "AVG_ANNUAL_EMPL":  "AVG_EMPLOYEE_CNT",
    "C_M_IND":          "COAL_METAL_IND",
    "SUBUNIT_DESC":     "SUBUNIT",
    "STATE_ABBR":       "STATE",
    "CURR_MINE_NM":     "MINE_NAME",
}


@dataclass
class MSHAOpsConfig:
    # Mundlak between/within decomposition (quarterly cadence)
    roll_k:   int = 4    # rolling window (quarters) for the between component
    lag_k:    int = 1    # lag (quarters) — use prior period
    min_hist: int = 2    # minimum quarters of history before features are valid
    output_format: str = "parquet"


class MSHAOpsPipeline:
    """MSHA employment/production → mine-quarter exposure offsets."""

    def __init__(
        self,
        raw_dir: Union[str, Path] = "data/raw/msha",
        output_dir: Union[str, Path] = "data/processed/msha",
        config: Optional[MSHAOpsConfig] = None,
        auto_download: bool = True,
    ):
        self.raw_dir = Path(raw_dir)
        self.output_dir = Path(output_dir)
        self.config = config or MSHAOpsConfig()
        self.auto_download = auto_download
        self._subunit: Optional[pd.DataFrame] = None
        self._mine_quarter: Optional[pd.DataFrame] = None

    # -------------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------------

    def load_quarterly(self) -> pd.DataFrame:
        df = read_dataset(
            "prod_quarterly", raw_dir=self.raw_dir,
            dtype={"MINE_ID": str, "SUBUNIT_CD": str, "COAL_METAL_IND": str},
            download_if_missing=self.auto_download,
        )
        df = df.rename(columns={"CURR_MINE_NM": "MINE_NAME"})
        return self._coerce(df)

    def load_yearly(self) -> pd.DataFrame:
        df = read_dataset(
            "prod_yearly", raw_dir=self.raw_dir,
            dtype={"MINE_ID": str, "SUBUNIT_CD": str, "C_M_IND": str},
            download_if_missing=self.auto_download,
        )
        df = df.rename(columns=YEARLY_RENAME)
        df["CAL_QTR"] = pd.NA   # annual granularity has no quarter
        return self._coerce(df)

    @staticmethod
    def _coerce(df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        out["mine_id"] = df["MINE_ID"].astype(str).str.strip()
        out["subunit_cd"] = df.get("SUBUNIT_CD")
        out["cal_yr"] = pd.to_numeric(df.get("CAL_YR"), errors="coerce").astype("Int64")
        out["cal_qtr"] = pd.to_numeric(df.get("CAL_QTR"), errors="coerce").astype("Int64")
        out["hours_worked"] = pd.to_numeric(df.get("HOURS_WORKED"), errors="coerce")
        out["coal_production"] = pd.to_numeric(df.get("COAL_PRODUCTION"), errors="coerce")
        out["avg_employee_cnt"] = pd.to_numeric(df.get("AVG_EMPLOYEE_CNT"), errors="coerce")
        out["commodity"] = df.get("COAL_METAL_IND").astype(str).str.strip().map(COMMODITY_MAP)
        return out

    # -------------------------------------------------------------------------
    # Aggregate
    # -------------------------------------------------------------------------

    def build_subunit_quarter(self, q: pd.DataFrame) -> pd.DataFrame:
        """One row per MINE_ID + CAL_YR + CAL_QTR + SUBUNIT_CD."""
        q = q[q["cal_qtr"].notna()].copy()
        q["mine_quarter"] = (q["mine_id"] + "_" + q["cal_yr"].astype(str)
                             + "Q" + q["cal_qtr"].astype(str))
        return q.reset_index(drop=True)

    def build_mine_quarter(self, q: pd.DataFrame) -> pd.DataFrame:
        """Sum subunits → one row per MINE_ID + CAL_YR + CAL_QTR."""
        q = q[q["cal_qtr"].notna()].copy()
        agg = (q.groupby(["mine_id", "cal_yr", "cal_qtr"], dropna=False)
                 .agg(hours_worked=("hours_worked", "sum"),
                      coal_production=("coal_production", "sum"),
                      avg_employee_cnt=("avg_employee_cnt", "sum"),
                      commodity=("commodity", "first"),
                      n_subunits=("subunit_cd", "nunique"))
                 .reset_index())
        agg["mine_quarter"] = (agg["mine_id"] + "_" + agg["cal_yr"].astype(str)
                               + "Q" + agg["cal_qtr"].astype(str))
        # Offsets. coal_production is only meaningful for coal mines.
        agg["log_hours"] = np.log(agg["hours_worked"].where(agg["hours_worked"] > 0))
        tons = agg["coal_production"].where(
            (agg["commodity"] == "coal") & (agg["coal_production"] > 0))
        agg["log_tons"] = np.log(tons)
        return self._add_mundlak(agg)

    def _add_mundlak(self, df: pd.DataFrame) -> pd.DataFrame:
        """Between/within decomposition of log-hours over each mine's
        quarterly history (lagged + standardised), mirroring the PHMSA ops
        pipeline so the R side gets a comparable exposure-trend feature."""
        cfg = self.config
        df = df.sort_values(["mine_id", "cal_yr", "cal_qtr"]).copy()
        gb = df.groupby("mine_id", sort=False)

        lag = gb["log_hours"].shift(cfg.lag_k)
        between = lag.groupby(df["mine_id"], sort=False).transform(
            lambda s: s.rolling(cfg.roll_k, min_periods=1).mean())
        within = lag - between
        hist_n = gb.cumcount() + 1
        valid = hist_n > cfg.min_hist

        def _z(x: pd.Series) -> pd.Series:
            mu, sd = x.mean(skipna=True), x.std(skipna=True)
            if not np.isfinite(sd) or sd == 0:
                return pd.Series(np.nan, index=x.index)
            return (x - mu) / sd

        df["hours_between"] = np.where(valid, _z(between), np.nan)
        df["hours_within"] = np.where(valid, _z(within), np.nan)
        return df.reset_index(drop=True)

    # -------------------------------------------------------------------------
    # I/O + run
    # -------------------------------------------------------------------------

    def save(self, df: pd.DataFrame, filename: str) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        fmt = self.config.output_format
        out_path = self.output_dir / f"{filename}.{fmt}"
        if fmt == "parquet":
            df.to_parquet(out_path, index=False, engine="pyarrow")
        elif fmt == "csv":
            df.to_csv(out_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {fmt}")
        logger.info(f"Saved {len(df):,} rows to {out_path}")
        return out_path

    def run(self, save: bool = True, include_yearly: bool = False) -> Dict[str, pd.DataFrame]:
        q = self.load_quarterly()
        subunit = self.build_subunit_quarter(q)
        mine_quarter = self.build_mine_quarter(q)
        self._subunit, self._mine_quarter = subunit, mine_quarter

        results = {"ops_subunit_quarter": subunit, "ops_mine_quarter": mine_quarter}
        if include_yearly:
            y = self.load_yearly()
            results["ops_subunit_yearly"] = y.reset_index(drop=True)

        if save:
            for name, df in results.items():
                self.save(df, name)
        logger.info("\n" + self.summary())
        return results

    def summary(self) -> str:
        mq = self._mine_quarter
        if mq is None:
            return "No data loaded."
        lines = ["MSHA Exposure (mine-quarter) Summary", "=" * 40,
                 f"Mine-quarters: {len(mq):,}",
                 f"Unique mines: {mq['mine_id'].nunique():,}",
                 f"Year range: {int(mq['cal_yr'].min())}-{int(mq['cal_yr'].max())}"]
        for c, sub in mq.groupby("commodity"):
            lines.append(f"  {c}: {len(sub):,} mine-quarters | "
                         f"median hours={sub['hours_worked'].median():,.0f}")
        lines.append(f"log_tons present (coal): {mq['log_tons'].notna().sum():,}")
        return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description="MSHA Exposure/Offset Pipeline")
    p.add_argument("--raw-dir", default="data/raw/msha")
    p.add_argument("--output", default="data/processed/msha")
    p.add_argument("--format", default="parquet", choices=["parquet", "csv"])
    p.add_argument("--include-yearly", action="store_true",
                   help="Also emit the annual rollup (pre-2000 backfill granularity)")
    p.add_argument("--no-download", action="store_true")
    p.add_argument("--summary", action="store_true")
    args = p.parse_args()

    cfg = MSHAOpsConfig(output_format=args.format)
    pipeline = MSHAOpsPipeline(raw_dir=args.raw_dir, output_dir=args.output,
                               config=cfg, auto_download=not args.no_download)
    pipeline.run(save=True, include_yearly=args.include_yearly)
    if args.summary:
        print("\n" + pipeline.summary())


if __name__ == "__main__":
    main()
