"""
FAA AIDS (Accident and Incident Data System) Pipeline
======================================================

Reads the FAA AIDS A-file zips (one per year-range), parses the c-numbered
columns into a standardized schema, and writes a single
`aids_events.parquet` for downstream R panel analyses.

Source layout: data/aviation/aids/raw/
    a1975_79.zip ... a2020_26.zip   — A-files (main records)
    e1975_79.zip ... e2020_26.zip   — E-files (event narratives, optional)
    Apre1975.zip / Epre1975.zip     — pre-1975 archive
    airport.zip                      — FAA airport master (LOC ID lookup)
    acftser.zip                      — aircraft series codes (not used here)

Each A-file is tab-separated with `\\N` as null marker. Columns are c-numbered
(c1, c2, ..., c790). See `meta_docs/Afilelayout.txt` for full schema. Key
fields parsed here:

    c5    raw_event_id (15-char control number)        → eid (UUID5)
    c1    type code: 'a'=accident, 'i'=incident         → event_type
    c2    FAR part (e.g. "121", "135 COMMUTER")          → far_part
    c6/c7/c8/c9   year/month/day                        → event_date
    c13   state                                          → state
    c143  on-airport FAA LOC ID (4-char)                 → airport_id (preferred)
    c17   nearest off-airport FAA LOC ID                 → airport_id (fallback)
    c18   direction from off-airport ID                  → apt_direction
    c19   distance from off-airport ID (NM)              → apt_dist_nm
    c61   total persons onboard                          → n_persons_onboard
    c63   passengers killed                              → (sum component)
    c66   flight crew killed                             → (sum component)
    c70   cabin crew killed                              → (sum component)
    c72   ground crew killed                             → (sum component)
    c74   public killed                                  → (sum component)
    c76   total fatalities (from source)                 → total_fatalities
    c250  total injuries (from source)                   → total_injuries

Era 1 (pre-1975) and Era 2 (1975-2026) share the same schema. No per-era
standardization needed — single read path.

Output schema:
    eid (UUID5), raw_event_id, event_date (Date), event_year,
    event_type, far_part, state,
    airport_id, on_airport (bool), apt_dist_nm,
    n_persons_onboard, total_fatalities, total_injuries,
    plus optional joined airport metadata: airport_name, airport_city,
    airport_state_lookup, icao

CLI:
    python aids_data_pipeline.py \\
        --base-dir data/aviation/aids/raw \\
        --output   data/processed/aviation \\
        --summary
"""

from __future__ import annotations

import argparse
import io
import logging
import re
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Source-file registry
# =============================================================================
# A-files only (E-files are narrative remarks; can be added later if useful
# for feature extraction). Pre-1975 archive is included by default since the
# era restriction is applied downstream at panel-prep time.

AIDS_A_FILES: List[str] = [
    "Apre1975.zip",
    "a1975_79.zip", "a1980_84.zip", "a1985_89.zip", "a1990_94.zip",
    "a1995_99.zip", "a2000_04.zip", "a2005_09.zip", "a2010_14.zip",
    "a2015_19.zip", "a2020_26.zip",
]

AIRPORT_LOOKUP_ZIP: str = "airport.zip"

NULL_TOKEN: str = r"\N"


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class AIDSConfig:
    domain: str = "aids"

    # Casualty totals: prefer pre-summed columns (c76, c250) where present,
    # else compute from component fields. AIDS data quality varies by era;
    # both paths are populated so downstream code can sanity-check.
    fatality_components: List[str] = field(default_factory=lambda: [
        "c63",  # passengers killed
        "c66",  # flight crew killed
        "c70",  # cabin crew killed
        "c72",  # ground crew killed
        "c74",  # public killed
    ])
    injury_components: List[str] = field(default_factory=lambda: [
        "c64",  # passengers injured
        "c67",  # flight crew injured
        "c69",  # cabin crew injured
        "c71",  # ground crew injured
        "c73",  # public injured
    ])

    output_format: str = "parquet"


# =============================================================================
# Helpers
# =============================================================================

def _make_eid(raw_id: str, domain: str = "aids") -> str:
    """Deterministic UUID-5 scoped to (domain, raw_event_id)."""
    ns = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")  # standard DNS namespace
    return str(uuid.uuid5(ns, f"{domain}:{raw_id}"))


def _coerce_int(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _parse_aids_date(c9: pd.Series, c6: pd.Series, c7: pd.Series,
                     c8: pd.Series) -> pd.Series:
    """Parse AIDS date. Prefer c9 (YYYYMMDD), fall back to c6/c7/c8 trio."""
    out = pd.to_datetime(c9, format="%Y%m%d", errors="coerce")
    needs_fallback = out.isna() & c6.notna() & c7.notna() & c8.notna()
    if needs_fallback.any():
        ymd = (
            c6[needs_fallback].astype(str).str.zfill(4) +
            c7[needs_fallback].astype(str).str.zfill(2) +
            c8[needs_fallback].astype(str).str.zfill(2)
        )
        out.loc[needs_fallback] = pd.to_datetime(ymd, format="%Y%m%d",
                                                   errors="coerce")
    return out


def _classify_event_type(c1: pd.Series) -> pd.Series:
    """c1 codes: 'a'=accident, 'i'=incident, others = 'other'."""
    s = c1.astype(str).str.strip().str.lower()
    return s.where(s.isin(["a", "i"]), other="other") \
            .map({"a": "accident", "i": "incident"}) \
            .fillna("other")


# =============================================================================
# Pipeline
# =============================================================================

class AIDSDataPipeline:

    def __init__(self,
                 base_dir:    Path | str,
                 output_dir:  Path | str = "data/processed/aviation",
                 config:      Optional[AIDSConfig] = None):
        self.base_dir   = Path(base_dir)
        self.output_dir = Path(output_dir)
        self.config     = config or AIDSConfig()
        self._df: Optional[pd.DataFrame] = None
        self._airport_lookup: Optional[pd.DataFrame] = None

    # -------------------------------------------------------------------------
    # Loading
    # -------------------------------------------------------------------------

    @staticmethod
    def _read_a_zip(zip_path: Path) -> Optional[pd.DataFrame]:
        """Read an AIDS A-file zip; expects a single .txt member."""
        if not zip_path.exists():
            logger.warning(f"  Missing: {zip_path.name}")
            return None
        with zipfile.ZipFile(zip_path) as zf:
            members = [m for m in zf.namelist() if m.lower().endswith(".txt")]
            if not members:
                logger.warning(f"  No .txt member in {zip_path.name}")
                return None
            with zf.open(members[0]) as f:
                # Use python engine + on_bad_lines='warn' to tolerate the
                # occasional row with a stray tab in a free-text cell. Bad
                # rows are logged, not silently dropped.
                df = pd.read_csv(
                    f, sep="\t",
                    dtype=str,
                    encoding="latin-1",
                    na_values=[NULL_TOKEN, ""],
                    keep_default_na=False,
                    engine="python",
                    on_bad_lines="warn",
                )
        df["__source_zip"] = zip_path.name
        return df

    def _read_airport_lookup(self) -> Optional[pd.DataFrame]:
        path = self.base_dir / AIRPORT_LOOKUP_ZIP
        if not path.exists():
            logger.warning(f"  Airport lookup not found: {path.name}")
            return None
        with zipfile.ZipFile(path) as zf:
            members = [m for m in zf.namelist() if m.lower().endswith(".txt")]
            if not members:
                return None
            with zf.open(members[0]) as f:
                df = pd.read_csv(f, sep="\t", dtype=str, encoding="latin-1",
                                  na_values=[NULL_TOKEN, ""],
                                  keep_default_na=False, low_memory=False)
        # Strip + uppercase the airport_id for stable joins
        if "AIRPORT_ID" in df.columns:
            df["AIRPORT_ID"] = df["AIRPORT_ID"].astype(str).str.strip().str.upper()
        return df.rename(columns={
            "AIRPORT_ID":     "airport_id",
            "AIRPORT_NAME":   "airport_name",
            "AIRPORT_CITY":   "airport_city",
            "AIRPORT_STATE":  "airport_state_lookup",
            "ICAO":           "icao",
        })

    def load_events(self) -> pd.DataFrame:
        frames: List[pd.DataFrame] = []
        for fname in AIDS_A_FILES:
            path = self.base_dir / fname
            df = self._read_a_zip(path)
            if df is not None:
                logger.info(f"  Loaded {fname}: {len(df):,} rows")
                frames.append(df)
        if not frames:
            raise RuntimeError("No AIDS A-files loaded. Check --base-dir.")

        df = pd.concat(frames, ignore_index=True, sort=False)
        logger.info(f"Concatenated raw A-files: {len(df):,} rows")
        return df

    # -------------------------------------------------------------------------
    # Standardization
    # -------------------------------------------------------------------------

    def standardize(self, raw: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config

        # Filter to valid raw_event_id (c5)
        raw = raw[raw.get("c5", pd.Series(dtype=str)).notna()].copy()
        raw["raw_event_id"] = raw["c5"].astype(str).str.strip()
        raw = raw[raw["raw_event_id"] != ""]

        out = pd.DataFrame(index=raw.index)
        out["raw_event_id"] = raw["raw_event_id"]
        out["eid"] = out["raw_event_id"].apply(
            lambda s: _make_eid(s, cfg.domain)
        )

        # --- Event type ---
        out["event_type"] = _classify_event_type(raw.get("c1", pd.Series("")))

        # --- FAR part ---
        out["far_part"] = (
            raw.get("c2", pd.Series(dtype=str))
                .astype(str).str.strip().replace({"nan": ""})
        )

        # --- Date ---
        out["event_date"] = _parse_aids_date(
            raw.get("c9", pd.Series(dtype=str)),
            raw.get("c6", pd.Series(dtype=str)),
            raw.get("c7", pd.Series(dtype=str)),
            raw.get("c8", pd.Series(dtype=str)),
        )
        out["event_year"] = out["event_date"].dt.year

        # --- Location ---
        c143 = raw.get("c143", pd.Series(dtype=str))
        c17  = raw.get("c17",  pd.Series(dtype=str))
        c143_clean = c143.astype(str).str.strip().str.upper().replace({"": np.nan, "NAN": np.nan})
        c17_clean  = c17.astype(str).str.strip().str.upper().replace({"": np.nan, "NAN": np.nan})

        # Prefer on-airport ID (c143), fall back to nearest-airport ID (c17)
        out["airport_id"] = c143_clean.fillna(c17_clean)
        out["on_airport"] = c143_clean.notna()

        # Distance is meaningful only when off-airport (c17). On-airport rows
        # get NaN distance — caller treats these as "near airport" by default.
        c19 = raw.get("c19", pd.Series(dtype=str))
        out["apt_dist_nm"] = _coerce_int(c19).where(~out["on_airport"], np.nan)

        out["state"] = (
            raw.get("c13", pd.Series(dtype=str))
                .astype(str).str.strip().str.upper().replace({"": np.nan, "NAN": np.nan})
        )

        # --- Casualties ---
        out["n_persons_onboard"] = _coerce_int(raw.get("c61", pd.Series(dtype=str))).fillna(0).astype(int)

        # Total fatalities: prefer c76; fall back to component sum
        c76 = _coerce_int(raw.get("c76", pd.Series(dtype=str)))
        comp_fatal = sum(
            (_coerce_int(raw.get(c, pd.Series(dtype=str))).fillna(0)
             for c in cfg.fatality_components),
            pd.Series(0, index=raw.index)
        )
        out["total_fatalities"] = c76.fillna(comp_fatal).astype(int)

        # Total injuries: prefer c250; fall back to component sum
        c250 = _coerce_int(raw.get("c250", pd.Series(dtype=str)))
        comp_inj = sum(
            (_coerce_int(raw.get(c, pd.Series(dtype=str))).fillna(0)
             for c in cfg.injury_components),
            pd.Series(0, index=raw.index)
        )
        out["total_injuries"] = c250.fillna(comp_inj).astype(int)

        # --- Provenance ---
        out["source_zip"] = raw.get("__source_zip", pd.Series(dtype=str)).values

        return out.reset_index(drop=True)

    def join_airport_lookup(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._airport_lookup is None:
            self._airport_lookup = self._read_airport_lookup()
        if self._airport_lookup is None:
            return df
        keep = ["airport_id", "airport_name", "airport_city",
                "airport_state_lookup", "icao"]
        keep = [c for c in keep if c in self._airport_lookup.columns]
        # Some airports have multiple rows (re-coded over time); keep the
        # most recent record by LCHG_DATE if present, otherwise first.
        lk = self._airport_lookup
        if "LCHG_DATE" in lk.columns:
            lk = lk.assign(
                _lchg = pd.to_datetime(lk["LCHG_DATE"], errors="coerce")
            ).sort_values(["airport_id", "_lchg"]).drop_duplicates(
                subset=["airport_id"], keep="last"
            )
        return df.merge(lk[keep], on="airport_id", how="left")

    # -------------------------------------------------------------------------
    # Cleaning
    # -------------------------------------------------------------------------

    @staticmethod
    def clean(df: pd.DataFrame) -> pd.DataFrame:
        n0 = len(df)
        df = df[df["event_date"].notna()].copy()
        n1 = len(df)
        # Deduplicate on raw_event_id
        df = df.drop_duplicates(subset=["raw_event_id"], keep="first")
        n2 = len(df)
        logger.info(
            f"Cleaning: {n0:,} -> {n1:,} after date filter -> "
            f"{n2:,} after dedup ({n0 - n2:,} removed total)"
        )
        return df.reset_index(drop=True)

    # -------------------------------------------------------------------------
    # I/O
    # -------------------------------------------------------------------------

    def save(self, df: pd.DataFrame, filename: str = "aids_events") -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.output_dir / f"{filename}.{self.config.output_format}"
        if self.config.output_format == "parquet":
            df.to_parquet(out_path, index=False, engine="pyarrow")
        elif self.config.output_format == "csv":
            df.to_csv(out_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {self.config.output_format}")
        logger.info(f"Saved {len(df):,} events to {out_path}")
        return out_path

    def run(self, save: bool = True) -> pd.DataFrame:
        raw = self.load_events()
        df  = self.standardize(raw)
        df  = self.join_airport_lookup(df)
        df  = self.clean(df)
        self._df = df
        if save:
            self.save(df)
        return df

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    def summary(self, df: Optional[pd.DataFrame] = None) -> str:
        df = df if df is not None else self._df
        if df is None:
            return "No data loaded."
        lines = ["FAA AIDS Summary", "=" * 30,
                 f"Total events: {len(df):,}",
                 f"Date range: {df['event_date'].min().date()} -- "
                 f"{df['event_date'].max().date()}"]
        lines.append("Events by type:")
        for k, v in df["event_type"].value_counts().items():
            lines.append(f"  {k}: {v:,}")
        lines.append("Events by decade:")
        decades = (df["event_year"] // 10 * 10).astype("Int64")
        for d, n in decades.value_counts().sort_index().items():
            lines.append(f"  {int(d)}s: {n:,}")
        lines.append(f"On-airport: {df['on_airport'].mean()*100:.1f}%")
        lines.append(f"Unique airport_ids (events): "
                     f"{df['airport_id'].nunique(dropna=True):,}")
        lines.append(f"Total fatalities: {df['total_fatalities'].sum():,}")
        lines.append(f"Total injuries:   {df['total_injuries'].sum():,}")
        return "\n".join(lines)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="FAA AIDS Accident/Incident Data Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python aids_data_pipeline.py \\
      --base-dir data/aviation/aids/raw \\
      --output   data/processed/aviation --summary
"""
    )
    parser.add_argument("--base-dir", required=True,
                        help="Path to data/aviation/aids/raw")
    parser.add_argument("--output",   default="data/processed/aviation")
    parser.add_argument("--format",   default="parquet",
                        choices=["parquet", "csv"])
    parser.add_argument("--summary",  action="store_true")
    args = parser.parse_args()

    cfg = AIDSConfig(output_format=args.format)
    pipeline = AIDSDataPipeline(base_dir=args.base_dir,
                                 output_dir=args.output, config=cfg)
    df = pipeline.run(save=True)
    if args.summary:
        print("\n" + pipeline.summary(df))


if __name__ == "__main__":
    main()
