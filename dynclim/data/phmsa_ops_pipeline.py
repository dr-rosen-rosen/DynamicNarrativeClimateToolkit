"""
PHMSA Operator Annual Ops Pipeline (mileage denominator)
=========================================================

Reads PHMSA annual operator reports across all three modern pipeline types
(gas_distribution, gas_transmission, hazardous_liquid), extracts total
mileage per operator-year, computes the Mundlak between/within decomposition
of log(1 + mileage), and writes a single parquet:

    OPERATOR_ID, REPORT_YEAR, total_miles, pipeline_type,
    total_miles_between, total_miles_within

Replaces the R-side `phmsa/0_get_phmsa_ops.R` so the R analysis can read the
parquet directly without touching xlsx/csv.

Source files (per pipeline_type, all under
data/raw/phmsa/operator_data/{type}_2010_present):
    - 2017+: csv  (e.g. "GD AR 2017.csv")
    - 2010-2016: xlsx with two documentation rows before the header
                  (e.g. "annual_gas_distribution_2015.xlsx")

Schema mile column per type (consistent across formats):
    GD: MMILES_TOTAL
    GT: PARTDTOTALMILES   (Part D grand total — transmission + gathering)
    HL: PARTDTOTALMILES

Pre-2010 historical ops (GD 1970-2003, GT 1970-2009, HL 2004-2009) use
substantially different schemas (B1TOTAL_*, no REPORT_YEAR, etc.) and are NOT
loaded here. They could be added in a future Phase 2 alongside an Era 1-3
schema registry mirroring the events pipeline's HISTORICAL_FILES.

CLI:
    python phmsa_ops_pipeline.py \\
        --base-dir data/raw/phmsa/operator_data \\
        --output data/processed/phmsa \\
        --summary
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Source-file registry
# =============================================================================
# One entry per pipeline_type. Each year is resolved at runtime: try CSV first
# (post-2017 format), then xlsx (2010-2016). The mile column name is identical
# across both formats within a type.

OPS_FILES: Dict[str, Dict[str, Any]] = {
    "gas_distribution": {
        "subdir":       "annual_gas_distribution_2010_present 2",
        "miles_col":    "MMILES_TOTAL",
        "csv_template":  "GD AR {year}.csv",
        "xlsx_template": "annual_gas_distribution_{year}.xlsx",
    },
    "gas_transmission": {
        "subdir":       "annual_gas_transmission_gathering_2010_present",
        "miles_col":    "PARTDTOTALMILES",
        "csv_template":  "GT AR {year} Part A to D.csv",
        "xlsx_template": "annual_gas_transmission_gathering_{year}.xlsx",
    },
    "hazardous_liquid": {
        "subdir":       "annual_hazardous_liquid_2010_present",
        "miles_col":    "PARTDTOTALMILES",
        "csv_template":  "HL AR {year} Part A to E.csv",
        "xlsx_template": "annual_hazardous_liquid_{year}.xlsx",
    },
}


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class PHMSAOpsConfig:
    # Year range to scan. Anything missing on disk is silently skipped.
    min_year: int = 2010
    max_year: int = 2025

    # Mundlak between/within decomposition (annual data)
    roll_k:    int = 3   # rolling window for the between component (years)
    lag_k:     int = 1   # lag (years) — use prior-year report
    min_hist:  int = 2   # minimum years of history before features are valid

    output_format: str = "parquet"


# =============================================================================
# Pipeline
# =============================================================================

class PHMSAOpsPipeline:
    """Read PHMSA annual operator reports → operator-year mileage parquet."""

    def __init__(
        self,
        base_data_dir: Path | str,
        output_dir:    Path | str = "data/processed/phmsa",
        config:        Optional[PHMSAOpsConfig] = None,
    ):
        self.base_dir   = Path(base_data_dir)
        self.output_dir = Path(output_dir)
        self.config     = config or PHMSAOpsConfig()
        self._df: Optional[pd.DataFrame] = None

    # -------------------------------------------------------------------------
    # File resolution
    # -------------------------------------------------------------------------

    def _resolve(self, pipeline_type: str, year: int) -> Optional[Dict[str, Any]]:
        """Find the file for one (pipeline_type, year). Prefers csv, falls
        back to xlsx. Returns dict(path, format) or None if neither exists."""
        meta = OPS_FILES[pipeline_type]
        d = self.base_dir / meta["subdir"]
        csv_path  = d / meta["csv_template"].format(year=year)
        xlsx_path = d / meta["xlsx_template"].format(year=year)
        if csv_path.exists():
            return {"path": csv_path, "format": "csv",
                    "miles_col": meta["miles_col"]}
        if xlsx_path.exists():
            return {"path": xlsx_path, "format": "xlsx",
                    "miles_col": meta["miles_col"]}
        return None

    # -------------------------------------------------------------------------
    # Reading
    # -------------------------------------------------------------------------

    @staticmethod
    def _read_csv(path: Path, miles_col: str) -> Optional[pd.DataFrame]:
        # Latin-1 handles non-ASCII bytes that PHMSA emits in some operator
        # name / address fields (e.g. 0x96). UTF-8 is the default in pandas
        # but fails on those files. Latin-1 never errors on a byte stream.
        try:
            df = pd.read_csv(
                path,
                usecols=lambda c: c in ("OPERATOR_ID", "REPORT_YEAR", miles_col),
                low_memory=False,
                encoding="latin-1",
            )
        except Exception as e:
            logger.warning(f"Failed to read {path.name}: {e}")
            return None
        if miles_col not in df.columns:
            logger.warning(f"Column '{miles_col}' missing in {path.name}")
            return None
        df = df.rename(columns={miles_col: "total_miles"})
        df["OPERATOR_ID"] = pd.to_numeric(df["OPERATOR_ID"], errors="coerce")
        df["REPORT_YEAR"] = pd.to_numeric(df["REPORT_YEAR"], errors="coerce")
        df["total_miles"] = pd.to_numeric(df["total_miles"], errors="coerce")
        return df.dropna(subset=["OPERATOR_ID", "REPORT_YEAR"])

    @staticmethod
    def _read_xlsx(path: Path, miles_col: str) -> Optional[pd.DataFrame]:
        """Pre-2017 xlsx files have 2 documentation rows before the header."""
        try:
            df = pd.read_excel(path, skiprows=2)
        except Exception as e:
            logger.warning(f"Failed to read {path.name}: {e}")
            return None
        needed = {"OPERATOR_ID", "REPORT_YEAR", miles_col}
        missing = needed - set(df.columns)
        if missing:
            logger.warning(f"xlsx {path.name} missing columns: {sorted(missing)}")
            return None
        df = df[["OPERATOR_ID", "REPORT_YEAR", miles_col]].rename(
            columns={miles_col: "total_miles"}
        )
        df["OPERATOR_ID"] = pd.to_numeric(df["OPERATOR_ID"], errors="coerce")
        df["REPORT_YEAR"] = pd.to_numeric(df["REPORT_YEAR"], errors="coerce")
        df["total_miles"] = pd.to_numeric(df["total_miles"], errors="coerce")
        return df.dropna(subset=["OPERATOR_ID", "REPORT_YEAR"])

    def _read_one(self, pipeline_type: str, year: int) -> Optional[pd.DataFrame]:
        src = self._resolve(pipeline_type, year)
        if src is None:
            return None
        df = (self._read_csv(src["path"], src["miles_col"])
              if src["format"] == "csv"
              else self._read_xlsx(src["path"], src["miles_col"]))
        if df is None or df.empty:
            return None
        # Authoritative year override — some files have stale REPORT_YEAR cells
        df["REPORT_YEAR"]   = year
        df["pipeline_type"] = pipeline_type
        df["OPERATOR_ID"]   = df["OPERATOR_ID"].astype(int)
        return df

    # -------------------------------------------------------------------------
    # Loading + cleaning
    # -------------------------------------------------------------------------

    def load_ops(self) -> pd.DataFrame:
        cfg = self.config
        years = range(cfg.min_year, cfg.max_year + 1)

        frames: List[pd.DataFrame] = []
        for ptype in OPS_FILES:
            n_loaded = 0
            for yr in years:
                d = self._read_one(ptype, yr)
                if d is None:
                    logger.info(f"  Skipped {ptype} {yr} (no file)")
                    continue
                frames.append(d)
                n_loaded += len(d)
            logger.info(f"{ptype}: loaded {n_loaded:,} operator-year rows")

        if not frames:
            raise RuntimeError("No source files loaded. Check --base-dir.")

        df = pd.concat(frames, ignore_index=True)
        # Filter implausible negatives; keep zeros (legitimate per-year reports)
        df = df[df["total_miles"].notna() & (df["total_miles"] >= 0)]
        # Some operators submit multiple supplemental reports per year. Keep
        # the row with the largest mileage (matches R: slice_max(total_miles)).
        df = (df.sort_values(["pipeline_type", "OPERATOR_ID", "REPORT_YEAR",
                              "total_miles"],
                             ascending=[True, True, True, False])
                .drop_duplicates(["pipeline_type", "OPERATOR_ID", "REPORT_YEAR"],
                                 keep="first")
                .reset_index(drop=True))
        return df

    # -------------------------------------------------------------------------
    # Mundlak between/within decomposition (annual cadence)
    # -------------------------------------------------------------------------

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Per (pipeline_type, OPERATOR_ID), in chronological order:
            1. log1p(total_miles)
            2. lag by lag_k years
            3. rolling mean over roll_k years (the 'between' component, in
               natural-log space and on lagged values to enforce strict
               temporal ordering for downstream regression)
            4. deviation from rolling mean (the 'within' component)
            5. NA out the first min_hist years per operator
            6. standardize across the entire panel (matches the R script)
        """
        cfg = self.config
        df = df.sort_values(["pipeline_type", "OPERATOR_ID", "REPORT_YEAR"]).copy()

        df["miles_log"] = np.log1p(df["total_miles"])

        gb = df.groupby(["pipeline_type", "OPERATOR_ID"], sort=False)
        df["miles_lag"] = gb["miles_log"].shift(cfg.lag_k)

        df["miles_between_raw"] = (
            gb["miles_lag"]
              .transform(lambda s: s.rolling(cfg.roll_k, min_periods=1).mean())
        )
        df["miles_within_raw"] = df["miles_lag"] - df["miles_between_raw"]
        df["hist_n"] = gb.cumcount() + 1

        def _zscore(x: pd.Series) -> pd.Series:
            mu = x.mean(skipna=True)
            sd = x.std(skipna=True)
            if not np.isfinite(sd) or sd == 0:
                return pd.Series(np.nan, index=x.index)
            return (x - mu) / sd

        valid = df["hist_n"] > cfg.min_hist
        df["total_miles_between"] = np.where(
            valid, _zscore(df["miles_between_raw"]), np.nan
        )
        df["total_miles_within"] = np.where(
            valid, _zscore(df["miles_within_raw"]), np.nan
        )

        return df.drop(columns=["miles_log", "miles_lag",
                                "miles_between_raw", "miles_within_raw",
                                "hist_n"])

    # -------------------------------------------------------------------------
    # I/O
    # -------------------------------------------------------------------------

    def save(self, df: pd.DataFrame, filename: str = "operator_annual_ops") -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.output_dir / f"{filename}.{self.config.output_format}"
        if self.config.output_format == "parquet":
            df.to_parquet(out_path, index=False, engine="pyarrow")
        elif self.config.output_format == "csv":
            df.to_csv(out_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {self.config.output_format}")
        logger.info(f"Saved {len(df):,} operator-years to {out_path}")
        return out_path

    def run(self, save: bool = True) -> pd.DataFrame:
        raw = self.load_ops()
        logger.info(f"Combined: {len(raw):,} operator-year rows")
        out = self.add_features(raw)
        self._df = out
        if save:
            self.save(out)
        return out

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    def summary(self, df: Optional[pd.DataFrame] = None) -> str:
        df = df if df is not None else self._df
        if df is None:
            return "No data loaded."
        lines = ["PHMSA Operator Ops Summary", "=" * 30,
                 f"Total operator-years: {len(df):,}"]
        for ptype, sub in df.groupby("pipeline_type"):
            lines.append(
                f"  {ptype}: {len(sub):>6,} rows | "
                f"{sub['OPERATOR_ID'].nunique():>4} operators | "
                f"years {int(sub['REPORT_YEAR'].min())}-{int(sub['REPORT_YEAR'].max())} | "
                f"median miles={sub['total_miles'].median():.1f} | "
                f"pct_zero={100*(sub['total_miles']==0).mean():.1f}%"
            )
        return "\n".join(lines)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PHMSA Operator Annual Ops Pipeline (mileage denominator)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python phmsa_ops_pipeline.py \\
      --base-dir data/raw/phmsa/operator_data \\
      --output data/processed/phmsa --summary
"""
    )
    parser.add_argument("--base-dir", required=True,
                        help="Path to data/raw/phmsa/operator_data")
    parser.add_argument("--output",   default="data/processed/phmsa")
    parser.add_argument("--format",   default="parquet",
                        choices=["parquet", "csv"])
    parser.add_argument("--min-year", type=int, default=2010)
    parser.add_argument("--max-year", type=int, default=2025)
    parser.add_argument("--summary",  action="store_true")
    args = parser.parse_args()

    cfg = PHMSAOpsConfig(min_year=args.min_year, max_year=args.max_year,
                         output_format=args.format)
    pipeline = PHMSAOpsPipeline(base_data_dir=args.base_dir,
                                output_dir=args.output, config=cfg)
    df = pipeline.run(save=True)
    if args.summary:
        print("\n" + pipeline.summary(df))


if __name__ == "__main__":
    main()
