"""
MSHA Operational Covariates + Mines-Master Pipeline
===================================================

Builds the operational covariate tables that play the role of the NRC
``findings_count`` / ``action_matrix_col`` predictors, aggregated to the
mine-quarter join key (``MINE_ID`` + ``CAL_YR`` + ``CAL_QTR``), plus the Mines
master table used to define the universe and filter by commodity / status.

Sources (auto-downloaded via :mod:`dynclim.data.msha_sources`):
    Violations.zip   citation/order records (S&S flag, penalties) — quarter by
                     VIOLATION_ISSUE_DT (CAL_YR/CAL_QTR present in file)
    Inspections.zip  inspection events with on-site hours and inspector counts
    Mines.zip        master: commodity, type, status, controller/operator, geo

⚠️ Inspection intensity confounds violation counts (more on-site hours →
mechanically more findings). Both ``violation_count`` and inspection
``onsite_hours`` are exposed so the R side can model intensity explicitly
rather than treating violations as exogenous (build sheet §8.6).

Inspections file quirks: ``NBR_INSPECTOR`` (singular) and the hours columns are
literally named ``SUM(TOTAL_INSP_HOURS)`` / ``SUM(TOTAL_ON_SITE_HOURS)``.

Outputs:
    covariates_mine_quarter.parquet   merged violations+inspections per mine-quarter
    violations_mine_quarter.parquet   violations aggregate only
    inspections_mine_quarter.parquet  inspections aggregate only
    mines_master.parquet              cleaned master (one row per MINE_ID)

CLI:
    python -m dynclim.data.msha_covariates_pipeline --output data/processed/msha --summary
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from dynclim.data.msha_sources import read_dataset

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

COMMODITY_MAP: Dict[str, str] = {"C": "coal", "M": "mnm"}


@dataclass
class MSHACovariatesConfig:
    # Mines-master statuses considered "in scope" (producing). Used by
    # filter_universe(); the master itself is saved unfiltered.
    active_statuses: List[str] = None
    output_format: str = "parquet"

    def __post_init__(self):
        if self.active_statuses is None:
            self.active_statuses = ["Active", "Intermittent", "Temporarily Idled"]


def _mine_quarter(df: pd.DataFrame) -> pd.Series:
    return (df["mine_id"] + "_" + df["cal_yr"].astype(str)
            + "Q" + df["cal_qtr"].astype(str))


class MSHACovariatesPipeline:
    """Violations + Inspections → mine-quarter covariates; Mines → master."""

    def __init__(
        self,
        raw_dir: Union[str, Path] = "data/raw/msha",
        output_dir: Union[str, Path] = "data/processed/msha",
        config: Optional[MSHACovariatesConfig] = None,
        auto_download: bool = True,
    ):
        self.raw_dir = Path(raw_dir)
        self.output_dir = Path(output_dir)
        self.config = config or MSHACovariatesConfig()
        self.auto_download = auto_download

    # -------------------------------------------------------------------------
    # Violations
    # -------------------------------------------------------------------------

    def build_violations(self) -> pd.DataFrame:
        cols = ["MINE_ID", "CAL_YR", "CAL_QTR", "COAL_METAL_IND", "SIG_SUB",
                "NO_AFFECTED", "PROPOSED_PENALTY", "AMOUNT_PAID",
                "VIOLATOR_TYPE_CD"]
        df = read_dataset("violations", raw_dir=self.raw_dir,
                          usecols=cols,
                          dtype={"MINE_ID": str, "COAL_METAL_IND": str,
                                 "SIG_SUB": str, "VIOLATOR_TYPE_CD": str},
                          download_if_missing=self.auto_download)
        df = df.rename(columns=str.lower)
        df["mine_id"] = df["mine_id"].astype(str).str.strip()
        df["cal_yr"] = pd.to_numeric(df["cal_yr"], errors="coerce").astype("Int64")
        df["cal_qtr"] = pd.to_numeric(df["cal_qtr"], errors="coerce").astype("Int64")
        df = df[df["mine_id"].notna() & df["cal_yr"].notna() & df["cal_qtr"].notna()]

        df["is_ss"] = (df["sig_sub"].astype(str).str.strip().str.upper() == "Y").astype(int)
        df["proposed_penalty"] = pd.to_numeric(df["proposed_penalty"], errors="coerce").fillna(0)
        df["no_affected"] = pd.to_numeric(df["no_affected"], errors="coerce").fillna(0)

        agg = (df.groupby(["mine_id", "cal_yr", "cal_qtr"], dropna=False)
                 .agg(violation_count=("is_ss", "size"),
                      ss_count=("is_ss", "sum"),
                      proposed_penalty_sum=("proposed_penalty", "sum"),
                      no_affected_sum=("no_affected", "sum"),
                      commodity=("coal_metal_ind", "first"))
                 .reset_index())
        agg["ss_rate"] = agg["ss_count"] / agg["violation_count"].where(agg["violation_count"] > 0)
        agg["commodity"] = agg["commodity"].astype(str).str.strip().map(COMMODITY_MAP)
        agg["mine_quarter"] = _mine_quarter(agg)
        logger.info(f"Violations → {len(agg):,} mine-quarters")
        return agg

    # -------------------------------------------------------------------------
    # Inspections
    # -------------------------------------------------------------------------

    def build_inspections(self) -> pd.DataFrame:
        cols = ["MINE_ID", "CAL_YR", "CAL_QTR", "COAL_METAL_IND", "NBR_INSPECTOR",
                "SUM(TOTAL_ON_SITE_HOURS)", "SUM(TOTAL_INSP_HOURS)"]
        df = read_dataset("inspections", raw_dir=self.raw_dir,
                          usecols=cols,
                          dtype={"MINE_ID": str, "COAL_METAL_IND": str},
                          download_if_missing=self.auto_download)
        df = df.rename(columns={
            "MINE_ID": "mine_id", "CAL_YR": "cal_yr", "CAL_QTR": "cal_qtr",
            "COAL_METAL_IND": "coal_metal_ind", "NBR_INSPECTOR": "nbr_inspector",
            "SUM(TOTAL_ON_SITE_HOURS)": "onsite_hours",
            "SUM(TOTAL_INSP_HOURS)": "insp_hours",
        })
        df["mine_id"] = df["mine_id"].astype(str).str.strip()
        df["cal_yr"] = pd.to_numeric(df["cal_yr"], errors="coerce").astype("Int64")
        df["cal_qtr"] = pd.to_numeric(df["cal_qtr"], errors="coerce").astype("Int64")
        df = df[df["mine_id"].notna() & df["cal_yr"].notna() & df["cal_qtr"].notna()]

        for c in ("onsite_hours", "insp_hours", "nbr_inspector"):
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

        agg = (df.groupby(["mine_id", "cal_yr", "cal_qtr"], dropna=False)
                 .agg(inspection_count=("onsite_hours", "size"),
                      onsite_hours=("onsite_hours", "sum"),
                      insp_hours=("insp_hours", "sum"),
                      inspector_count=("nbr_inspector", "sum"),
                      commodity=("coal_metal_ind", "first"))
                 .reset_index())
        agg["commodity"] = agg["commodity"].astype(str).str.strip().map(COMMODITY_MAP)
        agg["mine_quarter"] = _mine_quarter(agg)
        logger.info(f"Inspections → {len(agg):,} mine-quarters")
        return agg

    # -------------------------------------------------------------------------
    # Mines master
    # -------------------------------------------------------------------------

    def build_mines(self) -> pd.DataFrame:
        cols = ["MINE_ID", "CURRENT_MINE_NAME", "COAL_METAL_IND",
                "CURRENT_MINE_TYPE", "CURRENT_MINE_STATUS", "CURRENT_STATUS_DT",
                "CURRENT_CONTROLLER_ID", "CURRENT_CONTROLLER_NAME",
                "CURRENT_OPERATOR_ID", "CURRENT_OPERATOR_NAME", "STATE",
                "PRIMARY_CANVASS", "PRIMARY_SIC", "LATITUDE", "LONGITUDE"]
        df = read_dataset("mines", raw_dir=self.raw_dir,
                          usecols=cols,
                          dtype={"MINE_ID": str, "COAL_METAL_IND": str,
                                 "CURRENT_CONTROLLER_ID": str,
                                 "CURRENT_OPERATOR_ID": str},
                          download_if_missing=self.auto_download)
        df = df.rename(columns=str.lower)
        df["mine_id"] = df["mine_id"].astype(str).str.strip()
        df["commodity"] = df["coal_metal_ind"].astype(str).str.strip().map(COMMODITY_MAP)
        df = df.drop_duplicates(subset=["mine_id"], keep="first").reset_index(drop=True)
        logger.info(f"Mines master → {len(df):,} mines")
        return df

    def filter_universe(self, mines: pd.DataFrame,
                        commodities: Optional[List[str]] = None) -> pd.DataFrame:
        """In-scope mines: active-ish status (+ optional commodity restriction)."""
        keep = mines[mines["current_mine_status"].isin(self.config.active_statuses)]
        if commodities:
            keep = keep[keep["commodity"].isin(set(commodities))]
        return keep.reset_index(drop=True)

    # -------------------------------------------------------------------------
    # I/O + run
    # -------------------------------------------------------------------------

    def save(self, df: pd.DataFrame, filename: str) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        fmt = self.config.output_format
        out_path = self.output_dir / f"{filename}.{fmt}"
        if fmt == "parquet":
            df = df.copy()
            for c in df.select_dtypes(include="object").columns:
                df[c] = df[c].astype(str)
            df.to_parquet(out_path, index=False, engine="pyarrow")
        elif fmt == "csv":
            df.to_csv(out_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {fmt}")
        logger.info(f"Saved {len(df):,} rows to {out_path}")
        return out_path

    def run(self, save: bool = True) -> Dict[str, pd.DataFrame]:
        violations = self.build_violations()
        inspections = self.build_inspections()
        mines = self.build_mines()

        covariates = pd.merge(
            violations.drop(columns=["commodity"]),
            inspections.drop(columns=["commodity"]),
            on=["mine_id", "cal_yr", "cal_qtr", "mine_quarter"], how="outer",
        )
        # Mine-quarters with inspections but no violations → 0 findings
        for c in ("violation_count", "ss_count", "proposed_penalty_sum",
                  "no_affected_sum"):
            covariates[c] = covariates[c].fillna(0)
        for c in ("inspection_count", "onsite_hours", "insp_hours",
                  "inspector_count"):
            covariates[c] = covariates[c].fillna(0)

        results = {
            "covariates_mine_quarter": covariates,
            "violations_mine_quarter": violations,
            "inspections_mine_quarter": inspections,
            "mines_master": mines,
        }
        if save:
            for name, df in results.items():
                self.save(df, name)
        logger.info("\n" + self.summary(results))
        return results

    def summary(self, results: Dict[str, pd.DataFrame]) -> str:
        cov, mines = results["covariates_mine_quarter"], results["mines_master"]
        lines = ["MSHA Covariates + Mines Summary", "=" * 40,
                 f"Covariate mine-quarters: {len(cov):,}",
                 f"  median violations/q: {cov['violation_count'].median():.0f}",
                 f"  median onsite hours/q: {cov['onsite_hours'].median():.0f}",
                 f"Mines in master: {len(mines):,}"]
        lines.append("Mine status counts:")
        for k, v in mines["current_mine_status"].value_counts().head(8).items():
            lines.append(f"  {k}: {v:,}")
        lines.append("Mines by commodity:")
        for k, v in mines["commodity"].value_counts(dropna=False).items():
            lines.append(f"  {k}: {v:,}")
        return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description="MSHA Operational Covariates + Mines Pipeline")
    p.add_argument("--raw-dir", default="data/raw/msha")
    p.add_argument("--output", default="data/processed/msha")
    p.add_argument("--format", default="parquet", choices=["parquet", "csv"])
    p.add_argument("--no-download", action="store_true")
    p.add_argument("--summary", action="store_true")
    args = p.parse_args()

    cfg = MSHACovariatesConfig(output_format=args.format)
    pipeline = MSHACovariatesPipeline(raw_dir=args.raw_dir, output_dir=args.output,
                                      config=cfg, auto_download=not args.no_download)
    results = pipeline.run(save=True)
    if args.summary:
        print("\n" + pipeline.summary(results))


if __name__ == "__main__":
    main()
