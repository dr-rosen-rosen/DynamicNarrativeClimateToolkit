"""
MSHA Mine Accident/Injury Data Pipeline (Form 7000-1)
=====================================================

Loads, cleans, and standardises the MSHA Open-Government **Accidents** dataset
for safety-climate feature extraction. The narrative (Item 9) is **inline** on
the accident record — no separate merge step.

Source (auto-downloaded via :mod:`dynclim.data.msha_sources`):
    Accidents.zip → Accidents.txt  (pipe-delimited, quoted, header row)

Join spine: ``MINE_ID`` + ``DOCUMENT_NO`` identify a record; the mine-quarter
key is ``MINE_ID`` + ``CAL_YR`` + ``CAL_QTR`` (+ ``SUBUNIT_CD``). These keys
join directly to the ops (exposure) and covariate (violations/inspections)
parquets, and to the Mines master.

Outcomes exposed for the R-side CV model hierarchy (see build sheet §5):
    fatality          binary   DEGREE_INJURY_CD == '01'      (emerg_binary analogue)
    days_lost         count    DAYS_LOST                     (power_loss hurdle analogue)
    days_restrict     count    DAYS_RESTRICT
    degree_ordinal    ordinal  severity spine from DEGREE_INJURY_CD (ordinal_scram analogue)
    immed_notify_flag binary   one of the 12 immediately-reportable event types
    accident_only     binary   DEGREE_INJURY_CD == '00'      (no-injury / near-miss stream)

Commodity is carried as a flag (``commodity`` ∈ {coal, mnm}); records are not
filtered by commodity (configurable). Contractor records carry
``is_contractor``; v1 defaults to operator-only (``include_contractors=False``).

Outputs:
    events.parquet      cleaned, narrative-bearing records (for NLP + modelling)
    events_all.parquet  optional: all records, no narrative filter (panel-rate counts)

Usage:
    from dynclim.data.msha_data_pipeline import MSHADataPipeline
    pipeline = MSHADataPipeline()           # auto-downloads Accidents.zip
    results = pipeline.run(save=True, save_unfiltered=True)
    report_data = pipeline.get_report_data()  # [(eid, narrative), ...]

CLI:
    python -m dynclim.data.msha_data_pipeline --output data/processed/msha --summary
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from dynclim.data.msha_sources import read_dataset

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Code maps
# =============================================================================

# Ordered injury-severity spine (build sheet §5). Codes off the spine
# (00 accident-only, 07 illness, 08 natural cause, 09 non-employee, 10 other,
# '?' unknown) map to NA — handle/drop on the R side.
#   06 no days away / no restriction  → 1 (lowest reportable)
#   05 days restricted only           → 2
#   03 days away only                 → 3
#   04 days away + restricted         → 4
#   02 permanent disability           → 5
#   01 fatality                       → 6 (highest)
DEGREE_ORDINAL: Dict[str, int] = {
    "06": 1, "05": 2, "03": 3, "04": 4, "02": 5, "01": 6,
}

COMMODITY_MAP: Dict[str, str] = {"C": "coal", "M": "mnm"}


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class MSHAConfig:
    """Configuration for the MSHA accident/injury pipeline."""

    # --- Source columns (real flat-file headers) ---
    mine_id_col:        str = "MINE_ID"
    document_no_col:    str = "DOCUMENT_NO"
    narrative_col:      str = "NARRATIVE"
    date_col:           str = "ACCIDENT_DT"
    commodity_col:      str = "COAL_METAL_IND"
    contractor_id_col:  str = "CONTRACTOR_ID"

    # --- ID columns to read as strings (preserve leading zeros) ---
    str_cols: List[str] = field(default_factory=lambda: [
        "MINE_ID", "CONTROLLER_ID", "OPERATOR_ID", "CONTRACTOR_ID",
        "DOCUMENT_NO", "CLOSED_DOC_NO", "SUBUNIT_CD", "DEGREE_INJURY_CD",
        "CLASSIFICATION_CD", "ACCIDENT_TYPE_CD", "IMMED_NOTIFY_CD",
        "OCCUPATION_CD", "ACTIVITY_CD", "UG_LOCATION_CD", "COAL_METAL_IND",
        "FIPS_STATE_CD",
    ])

    # --- Curated columns carried through to the output ---
    # (standardised names added by the pipeline are always kept; these are the
    #  source descriptive/covariate columns worth retaining for the R side)
    keep_cols: List[str] = field(default_factory=lambda: [
        "CONTROLLER_ID", "CONTROLLER_NAME", "OPERATOR_ID", "OPERATOR_NAME",
        "SUBUNIT_CD", "SUBUNIT", "DEGREE_INJURY", "CLASSIFICATION_CD",
        "CLASSIFICATION", "ACCIDENT_TYPE_CD", "ACCIDENT_TYPE", "NO_INJURIES",
        "TOT_EXPER", "MINE_EXPER", "JOB_EXPER", "OCCUPATION_CD", "OCCUPATION",
        "ACTIVITY_CD", "ACTIVITY", "UG_LOCATION_CD", "UG_LOCATION",
        "SCHEDULE_CHARGE", "IMMED_NOTIFY",
    ])

    # --- Cleaning thresholds (narratives are terse: keep low) ---
    min_narrative_words: int = 3
    min_narrative_chars: int = 10

    # --- Scope toggles ---
    include_contractors: bool = False          # v1 default: operator-only
    commodities: Optional[List[str]] = None    # None = keep both; e.g. ["coal"]

    # --- Output ---
    output_format: str = "parquet"
    domain: str = "msha"


# =============================================================================
# Helpers
# =============================================================================

_EID_NS = uuid.UUID("b7d2f4a1-3c8e-4d6b-9a1f-2e5c7d0b3a6f")


def _make_eid(mine_id: str, document_no: str, domain: str = "msha") -> str:
    return str(uuid.uuid5(_EID_NS, f"{domain}:{mine_id}:{document_no}"))


def _clean_text(series: pd.Series) -> pd.Series:
    return (
        series.fillna("").astype(str).str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.replace(r"^\s*(None|nan)\s*$", "", regex=True)
        .str.strip()
    )


def _is_valid_utf8(text: str) -> bool:
    if pd.isna(text):
        return False
    try:
        str(text).encode("utf-8")
        return True
    except (UnicodeEncodeError, UnicodeDecodeError):
        return False


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


# =============================================================================
# Pipeline
# =============================================================================

class MSHADataPipeline:
    """Pipeline for MSHA Accidents (Form 7000-1) injury/narrative records."""

    def __init__(
        self,
        raw_dir: Union[str, Path] = "data/raw/msha",
        output_dir: Union[str, Path] = "data/processed/msha",
        config: Optional[MSHAConfig] = None,
        auto_download: bool = True,
    ):
        self.raw_dir = Path(raw_dir)
        self.output_dir = Path(output_dir)
        self.config = config or MSHAConfig()
        self.auto_download = auto_download
        self._events_df: Optional[pd.DataFrame] = None

    # -------------------------------------------------------------------------
    # Load + standardise
    # -------------------------------------------------------------------------

    def load_events(self, clean: bool = True) -> pd.DataFrame:
        cfg = self.config
        dtype = {c: str for c in cfg.str_cols}
        raw = read_dataset(
            "accidents", raw_dir=self.raw_dir, dtype=dtype,
            download_if_missing=self.auto_download,
        )
        df = self._standardize(raw)
        df = self._apply_scope(df)
        if clean:
            df = self.clean_events(df)
        self._events_df = df
        return df

    def _standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        out = pd.DataFrame(index=df.index)

        # --- Keys / IDs ---
        out["mine_id"] = df[cfg.mine_id_col].astype(str).str.strip()
        out["document_no"] = df[cfg.document_no_col].astype(str).str.strip()
        out["eid"] = [
            _make_eid(m, d, cfg.domain)
            for m, d in zip(out["mine_id"], out["document_no"])
        ]

        # --- Time + mine-quarter key ---
        out["accident_date"] = pd.to_datetime(
            df[cfg.date_col], errors="coerce", format="%m/%d/%Y"
        )
        out["cal_yr"] = _num(df.get("CAL_YR")).astype("Int64")
        out["cal_qtr"] = _num(df.get("CAL_QTR")).astype("Int64")
        out["mine_quarter"] = (
            out["mine_id"] + "_" + out["cal_yr"].astype(str)
            + "Q" + out["cal_qtr"].astype(str)
        )

        # --- Commodity + contractor flags ---
        out["commodity"] = (
            df[cfg.commodity_col].astype(str).str.strip().map(COMMODITY_MAP)
        )
        contractor = df.get(cfg.contractor_id_col)
        out["is_contractor"] = (
            contractor.notna() & (contractor.astype(str).str.strip() != "")
            if contractor is not None else False
        )

        # --- Narrative ---
        out["narrative"] = (
            _clean_text(df[cfg.narrative_col])
            if cfg.narrative_col in df.columns else ""
        )

        # --- Outcomes ---
        degree = df.get("DEGREE_INJURY_CD").astype(str).str.strip()
        out["degree_injury_cd"] = degree
        out["fatality"] = (degree == "01").astype(int)
        out["accident_only"] = (degree == "00").astype(int)
        out["degree_ordinal"] = degree.map(DEGREE_ORDINAL).astype("Int64")

        out["days_lost"] = _num(df.get("DAYS_LOST"))
        out["days_restrict"] = _num(df.get("DAYS_RESTRICT"))

        notify = df.get("IMMED_NOTIFY_CD")
        if notify is not None:
            n = notify.astype(str).str.strip()
            valid = {f"{i:02d}" for i in range(1, 13)}   # 01..12 reportable
            out["immed_notify_flag"] = n.isin(valid).astype(int)
        else:
            out["immed_notify_flag"] = 0

        # --- Curated passthrough columns ---
        for col in cfg.keep_cols:
            if col in df.columns:
                out[col.lower()] = df[col]

        return out

    def _apply_scope(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        n0 = len(df)
        if not cfg.include_contractors:
            df = df[~df["is_contractor"]]
            logger.info(f"Operator-only scope: dropped "
                        f"{n0 - len(df):,} contractor records")
        if cfg.commodities:
            keep = set(cfg.commodities)
            df = df[df["commodity"].isin(keep)]
            logger.info(f"Commodity filter {sorted(keep)}: {len(df):,} rows kept")
        return df.reset_index(drop=True)

    # -------------------------------------------------------------------------
    # Clean
    # -------------------------------------------------------------------------

    def clean_events(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        initial = len(df)
        df = df.copy()

        df = df[df["narrative"].notna()]
        df = df[df["narrative"].astype(str).str.strip() != ""]
        df = df[df["narrative"].str.lower() != "none"]
        df["_wc"] = df["narrative"].str.split().str.len()
        df = df[df["_wc"] >= cfg.min_narrative_words]
        df = df[df["narrative"].str.len() >= cfg.min_narrative_chars]
        df = df.drop(columns=["_wc"])
        df = df[df["narrative"].apply(_is_valid_utf8)]

        df = df.drop_duplicates(subset=["eid"], keep="first")

        logger.info(f"Cleaning: {initial:,} → {len(df):,} events "
                    f"({initial - len(df):,} removed)")
        return df.reset_index(drop=True)

    # -------------------------------------------------------------------------
    # Accessors
    # -------------------------------------------------------------------------

    def get_report_data(self) -> List[Tuple[str, str]]:
        """[(eid, narrative), ...] for the linguistic feature extractor."""
        if self._events_df is None:
            raise RuntimeError("No events loaded. Call load_events() first.")
        return [
            (row["eid"], row["narrative"])
            for _, row in self._events_df.iterrows()
            if str(row.get("narrative", "")).strip()
        ]

    def get_events(self) -> pd.DataFrame:
        if self._events_df is None:
            raise RuntimeError("No events loaded. Call load_events() first.")
        return self._events_df.copy()

    # -------------------------------------------------------------------------
    # Save / summary / run
    # -------------------------------------------------------------------------

    def save(self, df: Optional[pd.DataFrame] = None, filename: str = "events") -> Path:
        df = df if df is not None else self._events_df
        if df is None:
            raise RuntimeError("No data to save.")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        fmt = self.config.output_format.lower()
        if fmt == "parquet":
            out_path = self.output_dir / f"{filename}.parquet"
            df = df.copy()
            for col in df.select_dtypes(include="object").columns:
                df[col] = df[col].astype(str)
            df.to_parquet(out_path, index=False, engine="pyarrow")
        elif fmt == "csv":
            out_path = self.output_dir / f"{filename}.csv"
            df.to_csv(out_path, index=False)
        else:
            raise ValueError(f"Unsupported output format: {fmt}")
        logger.info(f"Saved {len(df):,} events to {out_path}")
        return out_path

    def summary(self, df: Optional[pd.DataFrame] = None) -> str:
        df = df if df is not None else self._events_df
        if df is None:
            return "No data loaded."
        lines = ["MSHA Accident/Injury Data Summary", "=" * 40,
                 f"Total events: {len(df):,}"]
        if "commodity" in df:
            lines.append("By commodity:")
            for k, v in df["commodity"].value_counts(dropna=False).items():
                lines.append(f"  {k}: {v:,}")
        if "is_contractor" in df:
            lines.append(f"Contractor records: {int(df['is_contractor'].sum()):,}")
        if "accident_date" in df:
            valid = df["accident_date"].dropna()
            if len(valid):
                lines.append(f"Date range: {valid.min():%Y-%m-%d} to {valid.max():%Y-%m-%d}")
        if "mine_id" in df:
            lines.append(f"Unique mines: {df['mine_id'].nunique():,}")
        if "fatality" in df:
            lines.append(f"Fatalities: {int(df['fatality'].sum()):,}")
        if "immed_notify_flag" in df:
            lines.append(f"Immediately-reportable events: {int(df['immed_notify_flag'].sum()):,}")
        if "days_lost" in df:
            dl = df["days_lost"]
            lines.append(f"Days lost: events>0 = {(dl > 0).sum():,}, total = {int(dl.fillna(0).sum()):,}")
        if "degree_ordinal" in df:
            lines.append("Degree ordinal (severity spine):")
            for k, v in df["degree_ordinal"].value_counts(dropna=True).sort_index().items():
                lines.append(f"  level {k}: {v:,}")
        if "narrative" in df:
            wc = df["narrative"].str.split().str.len()
            lines.append(f"Narrative word count: mean={wc.mean():.0f}, "
                         f"median={wc.median():.0f}, min={wc.min()}, max={wc.max()}")
        return "\n".join(lines)

    def run(self, save: bool = True, save_unfiltered: bool = False) -> Dict[str, pd.DataFrame]:
        """Load → (scope) → clean → save. Returns {'events', 'events_all'}."""
        cfg = self.config
        dtype = {c: str for c in cfg.str_cols}
        raw = read_dataset("accidents", raw_dir=self.raw_dir, dtype=dtype,
                           download_if_missing=self.auto_download)
        std = self._standardize(raw)
        std = self._apply_scope(std)

        if save_unfiltered:
            self.save(std, "events_all")

        events = self.clean_events(std)
        self._events_df = events
        if save:
            self.save(events, "events")
        logger.info("\n" + self.summary(events))
        return {"events": events, "events_all": std}


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    p = argparse.ArgumentParser(description="MSHA Accident/Injury Data Pipeline")
    p.add_argument("--raw-dir", default="data/raw/msha")
    p.add_argument("--output", default="data/processed/msha")
    p.add_argument("--format", default="parquet", choices=["parquet", "csv"])
    p.add_argument("--include-contractors", action="store_true",
                   help="Include contractor injury records (default: operator-only)")
    p.add_argument("--commodity", action="append", choices=["coal", "mnm"],
                   help="Restrict to commodity (repeatable); default keeps both")
    p.add_argument("--min-words", type=int, default=3)
    p.add_argument("--min-chars", type=int, default=10)
    p.add_argument("--save-unfiltered", action="store_true")
    p.add_argument("--no-download", action="store_true",
                   help="Do not auto-download; require files already present")
    p.add_argument("--summary", action="store_true")
    args = p.parse_args()

    cfg = MSHAConfig(
        output_format=args.format,
        include_contractors=args.include_contractors,
        commodities=args.commodity,
        min_narrative_words=args.min_words,
        min_narrative_chars=args.min_chars,
    )
    pipeline = MSHADataPipeline(
        raw_dir=args.raw_dir, output_dir=args.output, config=cfg,
        auto_download=not args.no_download,
    )
    results = pipeline.run(save=True, save_unfiltered=args.save_unfiltered)
    if args.summary:
        print("\n" + pipeline.summary(results["events"]))


if __name__ == "__main__":
    main()
