"""
Aviation Safety Reporting System (ASRS) Data Pipeline
======================================================

Single-file pipeline for loading, cleaning, and preparing ASRS event
reporting data for safety climate feature extraction.

Data Sources:
- Event reports: ASRS combined export (.xlsx) with narratives, operator info, dates
  (default filename: cmbd_ASRS.xlsx)

Outputs:
- events.parquet: Cleaned event reports ready for NLP feature extraction

Key Design Decisions:
- Column names are configurable via CLI args or ASRSConfig, with sensible defaults
  matching the cmbd_ASRS.xlsx export format
- Primary org: aircraft operator (Air Carrier, General Aviation, etc.)
- Secondary org: reporter function (Captain, First Officer, Controller, etc.)
- Single narrative field: cmbd_narrative (pre-combined by ASRS)
- Deterministic UUID-5 eid from ACN + domain namespace
- Follows same patterns as healthcare_data_pipeline.py and rail_data_pipeline.py

Usage:
    from aviation_data_pipeline import ASRSDataPipeline

    pipeline = ASRSDataPipeline(
        events_path="data/raw/aviation/cmbd_ASRS.xlsx",
        output_dir="data/processed/aviation"
    )

    events_df = pipeline.load_events()
    pipeline.run()
    report_data = pipeline.get_report_data()  # [(eid, narrative), ...]

CLI:
    python aviation_data_pipeline.py \\
        --events data/raw/aviation/cmbd_ASRS.xlsx \\
        --output data/processed/aviation \\
        --summary
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union, List, Dict, Tuple
from dataclasses import dataclass, field
import logging
import re
import uuid
import hashlib

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# ATC reporter-function sets
# =============================================================================
# Mirrors the aviation/config.R ATC_FUNCTIONS_* lists used by the downstream
# panel-rate analysis. Filtering at extraction time skips ~91% of the ASRS
# corpus that won't be used by the climate analysis (pilots, mechanics, etc.).

ATC_FUNCTIONS_LOCAL = [
    "Local",
    "Ground",
    "Flight Data / Clearance Delivery",
    "Handoff / Assist",
]
ATC_FUNCTIONS_TERMINAL = ["Approach", "Departure"]
ATC_FUNCTIONS_ENROUTE  = ["Enroute"]

ATC_FUNCTION_SETS: Dict[str, List[str]] = {
    "local":          list(ATC_FUNCTIONS_LOCAL),
    "local_terminal": list(ATC_FUNCTIONS_LOCAL) + list(ATC_FUNCTIONS_TERMINAL),
    "all_atc":        list(ATC_FUNCTIONS_LOCAL) + list(ATC_FUNCTIONS_TERMINAL)
                       + list(ATC_FUNCTIONS_ENROUTE),
}


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ASRSConfig:
    """Configuration for ASRS data pipeline.

    Column name defaults match the cmbd_ASRS.xlsx export format.
    Override via constructor or CLI args when column names differ.
    """

    # --- Column mappings (configurable) ---

    # ASRS Control Number — primary event identifier
    acn_col: str = "acn"
    acn_alts: List[str] = field(
        default_factory=lambda: ["acn", "ACN", "control_number", "report_number"]
    )

    # Date — already a parsed datetime in the combined export
    date_col: str = "date"
    date_alts: List[str] = field(
        default_factory=lambda: ["date", "time_date", "event_date", "Date"]
    )

    # Narrative — combined narrative field in the ASRS combined export
    narrative_col: str = "cmbd_narrative"
    narrative_alts: List[str] = field(
        default_factory=lambda: [
            "cmbd_narrative", "narrative", "Narrative", "report_text",
            "synopsis", "Synopsis",
        ]
    )

    # Primary org: aircraft operator (Air Carrier, General Aviation, Military, etc.)
    operator_col: str = "aircraft_1aircraft_operator"
    operator_alts: List[str] = field(
        default_factory=lambda: [
            "aircraft_1aircraft_operator", "aircraft_operator",
            "person_1reporter_organization", "reporter_organization",
            "operator",
        ]
    )

    # Airport — locale reference (IATA/ICAO code; analogous to facility/org in other pipelines)
    # Events can list multiple airports (semicolon-separated); primary_airport takes the first.
    airport_col: str = "place_locale_reference"
    airport_alts: List[str] = field(
        default_factory=lambda: [
            "place_locale_reference", "locale_reference", "airport",
            "airport_code", "Airport",
        ]
    )

    # Secondary org: reporter role/function (Captain, First Officer, Controller, etc.)
    reporter_function_col: str = "person_1function"
    reporter_function_alts: List[str] = field(
        default_factory=lambda: [
            "person_1function", "reporter_function", "function",
            "person_function", "role",
        ]
    )

    # --- Optional contextual columns preserved as-is ---
    # These are passed through without standardization but kept in output
    extra_cols: List[str] = field(
        default_factory=lambda: [
            "environment_flight_conditions",
            "aircraft_1flight_phase",
            "events_anomaly",
            "events_result",
            "assessments_primary_problem",
            "assessments_contributing_factors_situations",
            "person_1human_factors",
            "place_state_reference",
        ]
    )

    # --- Cleaning parameters ---
    min_narrative_words: int = 5
    min_narrative_chars: int = 20

    # --- ATC-only filter (optional) ---
    # When set to one of {"local", "local_terminal", "all_atc"}, drops reports
    # whose person_1function is outside the named ATC set BEFORE narrative
    # cleaning. Used to align the extraction corpus with the downstream
    # panel-rate analysis scope and dramatically reduce LLM/embedding compute
    # (~91% reduction at "all_atc"; ~95% at "local_terminal"). None = keep all.
    atc_filter: Optional[str] = None

    # --- Output ---
    output_format: str = "parquet"  # parquet, csv, or feather

    # --- Domain label (for cross-domain compatibility) ---
    domain: str = "asrs"


# =============================================================================
# Helpers
# =============================================================================

def _resolve_col(
    df: pd.DataFrame, primary: str, alts: List[str], label: str
) -> Optional[str]:
    """Find the actual column name in df, trying primary then alternatives."""
    if primary in df.columns:
        return primary
    for alt in alts:
        if alt in df.columns:
            logger.info(f"Column '{label}': using '{alt}' (primary '{primary}' not found)")
            return alt
    col_lower = {c.lower().strip(): c for c in df.columns}
    for candidate in [primary] + alts:
        if candidate.lower().strip() in col_lower:
            matched = col_lower[candidate.lower().strip()]
            logger.info(f"Column '{label}': case-insensitive match '{matched}'")
            return matched
    return None


def _make_eid(raw_id: str, domain: str = "asrs") -> str:
    """Generate a deterministic UUID-5 from raw event ID + domain namespace."""
    ns = uuid.UUID("a3c1e7d0-4b2f-4e8a-9c6d-1f0e2d3b4a5c")  # shared fixed namespace
    return str(uuid.uuid5(ns, f"{domain}:{raw_id}"))


def _make_org_id(name: str, level: str = "operator") -> str:
    """Generate a deterministic short hash for org identifiers."""
    h = hashlib.sha256(f"{level}:{name}".encode()).hexdigest()[:12]
    return f"{level[:3]}_{h}"


# =============================================================================
# Core Pipeline
# =============================================================================

class ASRSDataPipeline:
    """
    Pipeline for Aviation Safety Reporting System (ASRS) data.

    Handles:
    - Loading ASRS combined export files (.xlsx or .csv)
    - Resolving variable column names
    - Cleaning and standardizing
    - Operator + reporter-function org hierarchy with deterministic org_ids
    - Single combined narrative field (cmbd_narrative)
    - Deterministic UUID-5 eid from ACN
    """

    def __init__(
        self,
        events_path: Optional[Union[str, Path]] = None,
        output_dir: Union[str, Path] = "data/processed/aviation",
        config: Optional[ASRSConfig] = None,
    ):
        self.events_path = Path(events_path) if events_path else None
        self.output_dir = Path(output_dir)
        self.config = config or ASRSConfig()
        self._events_df: Optional[pd.DataFrame] = None
        self._resolved: Dict[str, Optional[str]] = {}

    # -------------------------------------------------------------------------
    # Column resolution
    # -------------------------------------------------------------------------

    def _resolve_columns(self, df: pd.DataFrame) -> Dict[str, Optional[str]]:
        """Resolve all configured column names against the actual DataFrame."""
        cfg = self.config
        resolved = {
            "acn": _resolve_col(df, cfg.acn_col, cfg.acn_alts, "acn"),
            "date": _resolve_col(df, cfg.date_col, cfg.date_alts, "date"),
            "narrative": _resolve_col(df, cfg.narrative_col, cfg.narrative_alts, "narrative"),
            "operator": _resolve_col(df, cfg.operator_col, cfg.operator_alts, "operator"),
            "airport": _resolve_col(df, cfg.airport_col, cfg.airport_alts, "airport"),
            "reporter_function": _resolve_col(
                df, cfg.reporter_function_col, cfg.reporter_function_alts, "reporter_function"
            ),
        }

        found = {k: v for k, v in resolved.items() if v is not None}
        missing = {k for k, v in resolved.items() if v is None}
        logger.info(f"Resolved columns: {found}")
        if missing:
            logger.warning(f"Could not resolve columns: {missing}")
            if "narrative" in missing:
                raise ValueError(
                    f"Cannot find narrative column. "
                    f"Tried: {[cfg.narrative_col] + cfg.narrative_alts}. "
                    f"Available columns: {list(df.columns)}"
                )

        self._resolved = resolved
        return resolved

    # -------------------------------------------------------------------------
    # Load Events
    # -------------------------------------------------------------------------

    def load_events(
        self,
        path: Optional[Union[str, Path]] = None,
        clean: bool = True,
    ) -> pd.DataFrame:
        """
        Load and process ASRS event data.

        Parameters
        ----------
        path : str or Path, optional
            Path to event file (.xlsx or .csv). Falls back to self.events_path.
        clean : bool
            Whether to clean narratives and filter invalid rows.

        Returns
        -------
        pd.DataFrame with standardized columns:
            eid, raw_event_id (acn), event_date, operator, operator_org_id,
            reporter_function, reporter_function_org_id, narrative,
            + all original columns preserved
        """
        path = Path(path) if path else self.events_path
        if path is None:
            raise ValueError("No events path specified")
        if not path.exists():
            raise FileNotFoundError(f"Events file not found: {path}")

        logger.info(f"Loading ASRS events from {path}")

        suffix = path.suffix.lower()
        if suffix in (".xlsx", ".xls"):
            df = pd.read_excel(path, engine="openpyxl" if suffix == ".xlsx" else None)
        elif suffix == ".csv":
            df = pd.read_csv(path, low_memory=False)
        elif suffix == ".tsv":
            df = pd.read_csv(path, sep="\t", low_memory=False)
        elif suffix == ".parquet":
            # Recommended path: feed the output of asrs_raw_pipeline.py here.
            df = pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

        logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

        resolved = self._resolve_columns(df)
        df = self._standardize(df, resolved)

        if clean:
            df = self.clean_events(df)

        self._events_df = df
        return df

    # -------------------------------------------------------------------------
    # Standardize
    # -------------------------------------------------------------------------

    def _standardize(self, df: pd.DataFrame, resolved: Dict[str, Optional[str]]) -> pd.DataFrame:
        """
        Add standardized columns while preserving all originals.

        Creates:
        - eid: deterministic UUID-5 from ACN
        - raw_event_id: original ACN string
        - event_date: parsed datetime
        - operator: cleaned aircraft operator name
        - operator_org_id: deterministic hash ID for operator
        - reporter_function: cleaned reporter role
        - reporter_function_org_id: deterministic hash ID scoped to operator+function
        - narrative: cleaned combined narrative text
        """
        df = df.copy()

        # --- Event ID (ACN) ---
        acn_col = resolved.get("acn")
        if acn_col:
            df["raw_event_id"] = df[acn_col].astype(str).str.strip()
        else:
            logger.warning("No ACN column found; generating sequential IDs")
            df["raw_event_id"] = [f"ASRS_{i:07d}" for i in range(len(df))]

        df["eid"] = df["raw_event_id"].apply(lambda x: _make_eid(x, self.config.domain))

        # --- Event Date ---
        date_col = resolved.get("date")
        if date_col:
            df["event_date"] = pd.to_datetime(df[date_col], errors="coerce")
            n_bad = df["event_date"].isna().sum()
            if n_bad > 0:
                logger.warning(f"{n_bad:,} events have unparseable dates")
        else:
            df["event_date"] = pd.NaT
            logger.warning("No date column found")

        # --- Operator (primary org) ---
        op_col = resolved.get("operator")
        if op_col:
            df["operator"] = df[op_col].astype(str).str.strip()
            df["operator"] = df["operator"].replace({"nan": "", "None": "", "NaN": ""})
            df["operator_org_id"] = df["operator"].apply(
                lambda x: _make_org_id(x, "operator") if x else ""
            )
        else:
            df["operator"] = ""
            df["operator_org_id"] = ""

        # --- Airport (primary location org — analogous to facility in other pipelines) ---
        airport_col = resolved.get("airport")
        if airport_col:
            raw_airport = df[airport_col].astype(str).str.strip()
            raw_airport = raw_airport.replace({"nan": "", "None": "", "NaN": ""})
            # Some events list multiple airports (semicolon-separated); take the first as primary
            df["airport_all"] = raw_airport
            df["airport"] = raw_airport.str.split(r"[;,]").str[0].str.strip()
            df["airport_org_id"] = df["airport"].apply(
                lambda x: _make_org_id(x.upper(), "airport") if x else ""
            )
        else:
            df["airport_all"] = ""
            df["airport"] = ""
            df["airport_org_id"] = ""

        # --- Reporter Function (secondary org) ---
        fn_col = resolved.get("reporter_function")
        if fn_col:
            df["reporter_function"] = df[fn_col].astype(str).str.strip()
            df["reporter_function"] = df["reporter_function"].replace(
                {"nan": "", "None": "", "NaN": ""}
            )
            # Scope function ID to operator for uniqueness across carriers
            df["reporter_function_org_id"] = df.apply(
                lambda row: _make_org_id(
                    f"{row['operator']}|{row['reporter_function']}", "function"
                )
                if row["reporter_function"]
                else "",
                axis=1,
            )
        else:
            df["reporter_function"] = ""
            df["reporter_function_org_id"] = ""

        # --- Narrative ---
        narr_col = resolved.get("narrative")
        if narr_col:
            df["narrative"] = self._clean_text_field(df[narr_col])
        else:
            df["narrative"] = ""

        return df

    # -------------------------------------------------------------------------
    # Text cleaning helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _clean_text_field(series: pd.Series) -> pd.Series:
        """Clean a text column: handle NaN, strip, normalize whitespace."""
        return (
            series.fillna("")
            .astype(str)
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
            .str.replace(r"^\s*None\s*$", "", regex=True)
            .str.replace(r"^\s*nan\s*$", "", regex=True)
            .str.strip()
        )

    @staticmethod
    def _is_valid_utf8(text: str) -> bool:
        """Check if text is valid UTF-8."""
        if pd.isna(text):
            return False
        try:
            str(text).encode("utf-8")
            return True
        except (UnicodeEncodeError, UnicodeDecodeError):
            return False

    # -------------------------------------------------------------------------
    # Clean events
    # -------------------------------------------------------------------------

    def clean_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean event data.

        Steps:
        0. (Optional) Filter to ATC-only reports via reporter_function
        1. Remove rows with missing/empty narratives
        2. Filter short narratives
        3. Validate UTF-8 encoding
        4. Deduplicate on raw_event_id (ACN)
        """
        initial_count = len(df)
        df = df.copy()

        # Step 0: ATC filter (applied first so all later steps work on a
        # smaller frame; speeds up regex-based narrative cleaning too).
        if self.config.atc_filter:
            valid_fns = ATC_FUNCTION_SETS.get(self.config.atc_filter)
            if valid_fns is None:
                raise ValueError(
                    f"Unknown atc_filter '{self.config.atc_filter}'. "
                    f"Must be one of: {sorted(ATC_FUNCTION_SETS)}."
                )
            if "reporter_function" not in df.columns:
                logger.warning(
                    "atc_filter set but 'reporter_function' column is absent; "
                    "skipping the filter."
                )
            else:
                n_before = len(df)
                # person_1function may contain semicolon-separated multi-roles
                # (e.g. "Captain; Pilot Flying"). Match any token equal to a
                # valid ATC function.
                fn = df["reporter_function"].astype(str)
                mask = fn.apply(
                    lambda s: any(tok.strip() in valid_fns
                                  for tok in s.split(";"))
                )
                df = df[mask]
                logger.info(
                    f"ATC filter ('{self.config.atc_filter}'): "
                    f"{n_before:,} -> {len(df):,} rows "
                    f"({100*(1 - len(df)/max(n_before,1)):.1f}% dropped)"
                )

        if "narrative" in df.columns:
            df = df[df["narrative"].notna()]
            df = df[df["narrative"].astype(str).str.strip() != ""]
            df = df[df["narrative"].str.lower() != "none"]

            df["_word_count"] = df["narrative"].str.split().str.len()
            df = df[df["_word_count"] >= self.config.min_narrative_words]
            df = df[df["narrative"].str.len() >= self.config.min_narrative_chars]
            df = df.drop(columns=["_word_count"])

            df = df[df["narrative"].apply(self._is_valid_utf8)]

        if "event_date" in df.columns:
            n_invalid = df["event_date"].isna().sum()
            if n_invalid > 0:
                logger.warning(f"{n_invalid:,} events have invalid/missing dates")

        # Deduplicate on ACN (keep first occurrence)
        if "raw_event_id" in df.columns:
            n_before = len(df)
            df = df.drop_duplicates(subset=["raw_event_id"], keep="first")
            n_dupes = n_before - len(df)
            if n_dupes > 0:
                logger.info(f"Removed {n_dupes:,} duplicate events (by ACN)")
        elif "eid" in df.columns:
            df = df.drop_duplicates(subset=["eid"], keep="first")

        final_count = len(df)
        logger.info(
            f"Cleaning: {initial_count:,} -> {final_count:,} events "
            f"({initial_count - final_count:,} removed)"
        )

        return df.reset_index(drop=True)

    # -------------------------------------------------------------------------
    # Accessors
    # -------------------------------------------------------------------------

    def get_report_data(self) -> List[Tuple[str, str]]:
        """Return list of (eid, narrative) tuples for NLP feature extraction."""
        if self._events_df is None:
            raise RuntimeError("No events loaded. Call load_events() first.")
        df = self._events_df
        return [
            (row["eid"], row["narrative"])
            for _, row in df.iterrows()
            if row.get("narrative", "").strip()
        ]

    def get_events(self) -> pd.DataFrame:
        """Return the processed events DataFrame."""
        if self._events_df is None:
            raise RuntimeError("No events loaded. Call load_events() first.")
        return self._events_df.copy()

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    def summary(self, df: Optional[pd.DataFrame] = None) -> str:
        """Return a human-readable summary of the loaded data."""
        df = df if df is not None else self._events_df
        if df is None:
            return "No data loaded."

        lines = [
            "ASRS Aviation Event Data Summary",
            "=" * 40,
            f"Total events: {len(df):,}",
        ]

        if "event_date" in df.columns:
            valid_dates = df["event_date"].dropna()
            if len(valid_dates) > 0:
                lines.append(
                    f"Date range: {valid_dates.min():%Y-%m-%d} to {valid_dates.max():%Y-%m-%d}"
                )
            lines.append(
                f"Events with valid dates: {len(valid_dates):,} / {len(df):,}"
            )

        if "operator" in df.columns:
            lines.append(f"Unique operators: {df['operator'].nunique():,}")
            top_ops = df["operator"].value_counts().head(5)
            lines.append("Top operators:")
            for op, cnt in top_ops.items():
                lines.append(f"  {op}: {cnt:,}")

        if "airport" in df.columns:
            n_airports = df["airport"].replace("", pd.NA).dropna().nunique()
            lines.append(f"Unique airports (primary): {n_airports:,}")
            top_airports = df["airport"].replace("", pd.NA).dropna().value_counts().head(5)
            lines.append("Top airports:")
            for ap, cnt in top_airports.items():
                lines.append(f"  {ap}: {cnt:,}")

        if "reporter_function" in df.columns:
            lines.append(f"Unique reporter functions: {df['reporter_function'].nunique():,}")

        if "narrative" in df.columns:
            word_counts = df["narrative"].str.split().str.len()
            lines.append(
                f"Narrative word count: "
                f"mean={word_counts.mean():.0f}, "
                f"median={word_counts.median():.0f}, "
                f"min={word_counts.min()}, "
                f"max={word_counts.max()}"
            )

        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # Save / Output
    # -------------------------------------------------------------------------

    def save(self, df: Optional[pd.DataFrame] = None, filename: str = "events") -> Path:
        """Save processed events to output directory."""
        df = df if df is not None else self._events_df
        if df is None:
            raise RuntimeError("No data to save. Call load_events() first.")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        fmt = self.config.output_format.lower()
        if fmt == "parquet":
            out_path = self.output_dir / f"{filename}.parquet"
            df = df.copy()
            for col in df.columns:
                if df[col].dtype == "object":
                    df[col] = df[col].astype(str)
            df.to_parquet(out_path, index=False, engine="pyarrow")
        elif fmt == "csv":
            out_path = self.output_dir / f"{filename}.csv"
            df.to_csv(out_path, index=False)
        elif fmt == "feather":
            out_path = self.output_dir / f"{filename}.feather"
            df.to_feather(out_path)
        else:
            raise ValueError(f"Unsupported output format: {fmt}")

        logger.info(f"Saved {len(df):,} events to {out_path}")
        return out_path

    # -------------------------------------------------------------------------
    # Full pipeline
    # -------------------------------------------------------------------------

    def run(self, save: bool = True) -> Dict[str, pd.DataFrame]:
        """Run the full pipeline: load -> clean -> save. Returns dict of DataFrames."""
        results = {}

        if self.events_path:
            events_df = self.load_events()
            results["events"] = events_df
            if save:
                self.save(events_df, "events")
            logger.info(self.summary(events_df))
        else:
            logger.warning("No events path specified; skipping events")

        return results


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Aviation Safety Reporting System (ASRS) Data Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with defaults
  python aviation_data_pipeline.py \\
      --events data/raw/aviation/cmbd_ASRS.xlsx \\
      --output data/processed/aviation

  # Override column names
  python aviation_data_pipeline.py \\
      --events cmbd_ASRS.xlsx \\
      --acn-col "acn" \\
      --narrative-col "cmbd_narrative" \\
      --operator-col "aircraft_1aircraft_operator" \\
      --summary

  # CSV output
  python aviation_data_pipeline.py \\
      --events cmbd_ASRS.xlsx \\
      --format csv \\
      --summary
        """,
    )

    parser.add_argument("--events", required=True, help="Path to ASRS event file (.xlsx or .csv)")
    parser.add_argument("--output", default="data/processed/aviation", help="Output directory")
    parser.add_argument("--format", default="parquet", choices=["parquet", "csv", "feather"])

    # Column name overrides
    parser.add_argument("--acn-col", default=None, help="ACN / event ID column name")
    parser.add_argument("--date-col", default=None, help="Date column name")
    parser.add_argument("--narrative-col", default=None, help="Narrative column name")
    parser.add_argument("--operator-col", default=None, help="Aircraft operator column name")
    parser.add_argument("--airport-col", default=None, help="Airport/locale reference column name")
    parser.add_argument("--reporter-function-col", default=None, help="Reporter function column name")

    # Cleaning
    parser.add_argument("--min-words", type=int, default=5, help="Min narrative word count")
    parser.add_argument("--min-chars", type=int, default=20, help="Min narrative char count")
    parser.add_argument("--no-clean", action="store_true", help="Skip cleaning step")
    parser.add_argument(
        "--atc-filter", default=None,
        choices=["local", "local_terminal", "all_atc"],
        help="Filter to ATC-only reports by person_1function. "
             "'local' (~2%% of corpus): tower controllers only. "
             "'local_terminal' (~5%%): adds Approach/Departure. "
             "'all_atc' (~9%%): adds Enroute. Skipping the filter keeps "
             "all ~205K reports (default).",
    )

    # Display
    parser.add_argument("--summary", action="store_true", help="Print summary after processing")
    parser.add_argument("--head", type=int, default=0, help="Print first N rows")

    args = parser.parse_args()

    config = ASRSConfig(output_format=args.format)
    config.min_narrative_words = args.min_words
    config.min_narrative_chars = args.min_chars
    config.atc_filter          = args.atc_filter

    if args.acn_col:
        config.acn_col = args.acn_col
    if args.date_col:
        config.date_col = args.date_col
    if args.narrative_col:
        config.narrative_col = args.narrative_col
    if args.operator_col:
        config.operator_col = args.operator_col
    if args.airport_col:
        config.airport_col = args.airport_col
    if args.reporter_function_col:
        config.reporter_function_col = args.reporter_function_col

    pipeline = ASRSDataPipeline(
        events_path=args.events,
        output_dir=args.output,
        config=config,
    )

    results = pipeline.run(save=True)

    if args.summary and "events" in results:
        print("\n" + pipeline.summary())

    if args.head > 0 and "events" in results:
        print(f"\nFirst {args.head} rows:")
        cols = ["eid", "raw_event_id", "event_date", "operator", "reporter_function", "narrative"]
        display_cols = [c for c in cols if c in results["events"].columns]
        print(results["events"][display_cols].head(args.head).to_string())


if __name__ == "__main__":
    main()
