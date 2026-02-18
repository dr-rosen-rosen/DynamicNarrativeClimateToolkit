"""
Healthcare Event Reporting (HERO) Data Pipeline
================================================

Single-file pipeline for loading, cleaning, and preparing healthcare event
reporting data for safety climate feature extraction.

Data Sources:
- Event reports: HERO event files (.xlsx) with narratives, org info, dates
- Operational data: (Placeholder for future integration)

Outputs:
- events.parquet: Cleaned event reports ready for NLP feature extraction

Key Design Decisions:
- Column names are configurable via CLI args or HeroConfig, with sensible defaults
- Two-level org hierarchy: facility (e.g., hospital) and unit (e.g., department)
- Both original narrative fields preserved + combined narrative for NLP
- UUID-based eid assigned alongside original event IDs
- Follows same patterns as rail_data_pipeline.py and nrc_data_pipeline.py

Usage:
    from healthcare_data_pipeline import HealthcareDataPipeline

    pipeline = HealthcareDataPipeline(
        events_path="data/raw/hero/events.xlsx",
        output_dir="data/processed/hero"
    )

    # Load and clean
    events_df = pipeline.load_events()

    # Or run full pipeline
    pipeline.run()

    # Get data for feature extraction
    report_data = pipeline.get_report_data()  # [(eid, narrative), ...]

CLI:
    python healthcare_data_pipeline.py \\
        --events hero_events.xlsx \\
        --output data/processed/hero \\
        --event-id-col "Event ID" \\
        --facility-col "Primary Organization" \\
        --unit-col "Primary Location" \\
        --event-date-col "Event Date" \\
        --description-col "Event Description" \\
        --suggestions-col "Suggestions" \\
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
# Configuration
# =============================================================================

@dataclass
class HeroConfig:
    """Configuration for Healthcare Event Reporting data pipeline.

    Column name defaults are set to the most common HERO export format.
    Override via constructor or CLI args when column names differ.
    """

    # --- Column mappings (configurable) ---

    # Event identifier in raw data
    event_id_col: str = "Event ID"
    event_id_alts: List[str] = field(
        default_factory=lambda: ["Event ID", "EventID", "Event_ID", "event_id", "ID", "Report ID"]
    )

    # Organization hierarchy
    facility_col: str = "Primary Organization"
    facility_alts: List[str] = field(
        default_factory=lambda: [
            "Primary Organization", "Organization", "Facility", "Hospital",
            "Primary_Organization", "primary_organization",
        ]
    )
    unit_col: str = "Primary Location"
    unit_alts: List[str] = field(
        default_factory=lambda: [
            "Primary Location", "Location", "Unit", "Department",
            "Primary_Location", "primary_location",
        ]
    )

    # Date
    event_date_col: str = "Event Date"
    event_date_alts: List[str] = field(
        default_factory=lambda: [
            "Event Date", "EventDate", "Event_Date", "event_date",
            "Date", "Date of Event", "Occurrence Date",
        ]
    )

    # Narrative fields (two parts that get concatenated)
    description_col: str = "Event Description"
    description_alts: List[str] = field(
        default_factory=lambda: [
            "Event Description", "EventDescription", "Event_Description",
            "event_description", "Description", "Narrative",
        ]
    )
    suggestions_col: str = "Suggestions"
    suggestions_alts: List[str] = field(
        default_factory=lambda: [
            "Suggestions", "Suggestion", "Recommendations",
            "suggestions", "Corrective Actions", "Follow Up",
        ]
    )

    # --- Cleaning parameters ---
    min_narrative_words: int = 5
    min_narrative_chars: int = 20

    # --- Output ---
    output_format: str = "parquet"  # parquet, csv, or feather

    # --- Domain label (for cross-domain compatibility) ---
    domain: str = "hero"


# =============================================================================
# Helper: resolve column name from alts
# =============================================================================

def _resolve_col(df: pd.DataFrame, primary: str, alts: List[str], label: str) -> Optional[str]:
    """Find the actual column name in df, trying primary then alternatives.

    Returns the matched column name or None if not found.
    """
    # Exact match first
    if primary in df.columns:
        return primary
    for alt in alts:
        if alt in df.columns:
            logger.info(f"Column '{label}': using '{alt}' (primary '{primary}' not found)")
            return alt
    # Case-insensitive fallback
    col_lower = {c.lower().strip(): c for c in df.columns}
    for candidate in [primary] + alts:
        if candidate.lower().strip() in col_lower:
            matched = col_lower[candidate.lower().strip()]
            logger.info(f"Column '{label}': case-insensitive match '{matched}'")
            return matched
    return None


# =============================================================================
# UUID generation (deterministic from raw event ID for reproducibility)
# =============================================================================

def _make_eid(raw_id: str, domain: str = "hero") -> str:
    """Generate a deterministic UUID-5 from raw event ID + domain namespace."""
    ns = uuid.UUID("a3c1e7d0-4b2f-4e8a-9c6d-1f0e2d3b4a5c")  # fixed namespace
    return str(uuid.uuid5(ns, f"{domain}:{raw_id}"))


def _make_org_id(name: str, level: str = "facility") -> str:
    """Generate a deterministic short hash for org identifiers."""
    h = hashlib.sha256(f"{level}:{name}".encode()).hexdigest()[:12]
    return f"{level[:3]}_{h}"


# =============================================================================
# Core Pipeline
# =============================================================================

class HealthcareDataPipeline:
    """
    Streamlined pipeline for Healthcare Event Reporting (HERO) data.

    Handles:
    - Loading event reports (.xlsx or .csv)
    - Resolving variable column names
    - Cleaning and standardizing
    - Two-level org hierarchy (facility + unit) with UUID org_ids
    - Dual narrative preservation + combined narrative
    - UUID eid generation
    """

    def __init__(
        self,
        events_path: Optional[Union[str, Path]] = None,
        output_dir: Union[str, Path] = "data/processed/hero",
        config: Optional[HeroConfig] = None,
    ):
        self.events_path = Path(events_path) if events_path else None
        self.output_dir = Path(output_dir)
        self.config = config or HeroConfig()
        self._events_df: Optional[pd.DataFrame] = None

        # Resolved column name cache (set during load)
        self._resolved: Dict[str, Optional[str]] = {}

    # -------------------------------------------------------------------------
    # Column resolution
    # -------------------------------------------------------------------------

    def _resolve_columns(self, df: pd.DataFrame) -> Dict[str, Optional[str]]:
        """Resolve all configured column names against the actual DataFrame."""
        cfg = self.config
        resolved = {
            "event_id": _resolve_col(df, cfg.event_id_col, cfg.event_id_alts, "event_id"),
            "facility": _resolve_col(df, cfg.facility_col, cfg.facility_alts, "facility"),
            "unit": _resolve_col(df, cfg.unit_col, cfg.unit_alts, "unit"),
            "event_date": _resolve_col(df, cfg.event_date_col, cfg.event_date_alts, "event_date"),
            "description": _resolve_col(df, cfg.description_col, cfg.description_alts, "description"),
            "suggestions": _resolve_col(df, cfg.suggestions_col, cfg.suggestions_alts, "suggestions"),
        }

        # Report what was found
        found = {k: v for k, v in resolved.items() if v is not None}
        missing = {k for k, v in resolved.items() if v is None}
        logger.info(f"Resolved columns: {found}")
        if missing:
            logger.warning(f"Could not resolve columns: {missing}")
            # description is required; suggestions is nice-to-have
            if "description" in missing:
                raise ValueError(
                    f"Cannot find narrative/description column. "
                    f"Tried: {[cfg.description_col] + cfg.description_alts}. "
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
        Load and process healthcare event data.

        Parameters
        ----------
        path : str or Path, optional
            Path to event file (.xlsx or .csv). Falls back to self.events_path.
        clean : bool
            Whether to clean narratives and filter invalid rows.

        Returns
        -------
        pd.DataFrame with standardized columns:
            eid, raw_event_id, event_date, facility, facility_org_id,
            unit, unit_org_id, narrative_description, narrative_suggestions,
            narrative, + all original columns preserved
        """
        path = Path(path) if path else self.events_path
        if path is None:
            raise ValueError("No events path specified")
        if not path.exists():
            raise FileNotFoundError(f"Events file not found: {path}")

        logger.info(f"Loading healthcare events from {path}")

        # --- Read file ---
        suffix = path.suffix.lower()
        if suffix in (".xlsx", ".xls"):
            df = pd.read_excel(path, engine="openpyxl" if suffix == ".xlsx" else None)
        elif suffix == ".csv":
            df = pd.read_csv(path, low_memory=False)
        elif suffix == ".tsv":
            df = pd.read_csv(path, sep="\t", low_memory=False)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

        logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

        # --- Resolve column names ---
        resolved = self._resolve_columns(df)

        # --- Standardize ---
        df = self._standardize(df, resolved)

        # --- Clean ---
        if clean:
            df = self.clean_events(df)

        self._events_df = df
        return df

    # -------------------------------------------------------------------------
    # Standardize columns
    # -------------------------------------------------------------------------

    def _standardize(self, df: pd.DataFrame, resolved: Dict[str, Optional[str]]) -> pd.DataFrame:
        """
        Add standardized columns while preserving all originals.

        Creates:
        - eid: UUID-5 from raw event ID
        - raw_event_id: original event ID
        - event_date: parsed datetime
        - facility / unit: cleaned org names
        - facility_org_id / unit_org_id: deterministic hash IDs
        - narrative_description: cleaned description text
        - narrative_suggestions: cleaned suggestions text
        - narrative: combined narrative (description + suggestions)
        """
        df = df.copy()

        # --- Event ID ---
        raw_id_col = resolved.get("event_id")
        if raw_id_col:
            df["raw_event_id"] = df[raw_id_col].astype(str).str.strip()
        else:
            # Generate sequential IDs if no event ID column
            logger.warning("No event ID column found; generating sequential IDs")
            df["raw_event_id"] = [f"HERO_{i:06d}" for i in range(len(df))]

        # UUID eid (deterministic from raw ID)
        df["eid"] = df["raw_event_id"].apply(lambda x: _make_eid(str(x), self.config.domain))

        # --- Event Date ---
        date_col = resolved.get("event_date")
        if date_col:
            df["event_date"] = pd.to_datetime(df[date_col], errors="coerce")
            n_bad = df["event_date"].isna().sum()
            if n_bad > 0:
                logger.warning(f"{n_bad:,} events have unparseable dates")
        else:
            df["event_date"] = pd.NaT
            logger.warning("No event date column found")

        # --- Facility (level 1 org) ---
        fac_col = resolved.get("facility")
        if fac_col:
            df["facility"] = df[fac_col].astype(str).str.strip()
            df["facility"] = df["facility"].replace({"nan": "", "None": "", "NaN": ""})
            df["facility_org_id"] = df["facility"].apply(
                lambda x: _make_org_id(x, "facility") if x else ""
            )
        else:
            df["facility"] = ""
            df["facility_org_id"] = ""

        # --- Unit (level 2 org) ---
        unit_col = resolved.get("unit")
        if unit_col:
            df["unit"] = df[unit_col].astype(str).str.strip()
            df["unit"] = df["unit"].replace({"nan": "", "None": "", "NaN": ""})
            # Unit org_id includes facility for uniqueness (same unit name at different hospitals)
            df["unit_org_id"] = df.apply(
                lambda row: _make_org_id(f"{row['facility']}|{row['unit']}", "unit")
                if row["unit"]
                else "",
                axis=1,
            )
        else:
            df["unit"] = ""
            df["unit_org_id"] = ""

        # --- Narratives ---
        desc_col = resolved.get("description")
        sugg_col = resolved.get("suggestions")

        # Description
        if desc_col:
            df["narrative_description"] = self._clean_text_field(df[desc_col])
        else:
            df["narrative_description"] = ""

        # Suggestions
        if sugg_col:
            df["narrative_suggestions"] = self._clean_text_field(df[sugg_col])
        else:
            df["narrative_suggestions"] = ""

        # Combined narrative
        df["narrative"] = df.apply(
            lambda row: self._combine_narratives(
                row["narrative_description"], row["narrative_suggestions"]
            ),
            axis=1,
        )

        return df

    # -------------------------------------------------------------------------
    # Text cleaning helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _clean_text_field(series: pd.Series) -> pd.Series:
        """Clean a text column: handle NaN, strip, normalize whitespace."""
        cleaned = (
            series.fillna("")
            .astype(str)
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)  # collapse whitespace
            .str.replace(r"NoneNone", "", regex=False)  # common artifact
            .str.replace(r"^\s*None\s*$", "", regex=True)
            .str.replace(r"^\s*nan\s*$", "", regex=True)
            .str.strip()
        )
        return cleaned

    @staticmethod
    def _combine_narratives(description: str, suggestions: str) -> str:
        """Concatenate description and suggestions with separator."""
        parts = []
        if description and description.lower() not in ("", "none", "nan"):
            parts.append(description.strip())
        if suggestions and suggestions.lower() not in ("", "none", "nan"):
            parts.append(suggestions.strip())
        return " ".join(parts) if parts else ""

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
        1. Remove rows with missing/empty narratives
        2. Clean narrative text artifacts
        3. Filter short narratives
        4. Validate dates
        5. Ensure valid UTF-8
        6. Drop duplicates on raw_event_id
        """
        initial_count = len(df)
        df = df.copy()

        # 1. Basic narrative filtering
        if "narrative" in df.columns:
            # Remove null/empty
            df = df[df["narrative"].notna()]
            df = df[df["narrative"].astype(str).str.strip() != ""]

            # Filter empty after cleaning
            df = df[df["narrative"] != ""]
            df = df[df["narrative"].str.lower() != "none"]

            # 2. Filter short narratives
            df["_word_count"] = df["narrative"].str.split().str.len()
            df = df[df["_word_count"] >= self.config.min_narrative_words]
            df = df[df["narrative"].str.len() >= self.config.min_narrative_chars]
            df = df.drop(columns=["_word_count"])

            # 3. UTF-8 validation
            df = df[df["narrative"].apply(self._is_valid_utf8)]

        # 4. Date validation
        if "event_date" in df.columns:
            n_invalid_dates = df["event_date"].isna().sum()
            if n_invalid_dates > 0:
                logger.warning(f"{n_invalid_dates:,} events have invalid/missing dates")

        # 5. Deduplicate on raw_event_id
        if "raw_event_id" in df.columns:
            n_before = len(df)
            df = df.drop_duplicates(subset=["raw_event_id"], keep="first")
            n_dupes = n_before - len(df)
            if n_dupes > 0:
                logger.info(f"Removed {n_dupes:,} duplicate events (by raw_event_id)")
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
        """
        Return list of (eid, narrative) tuples for feature extraction.

        Compatible with the format expected by the NLP feature extraction pipeline.
        """
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
        """Print a summary of the loaded data."""
        df = df if df is not None else self._events_df
        if df is None:
            return "No data loaded."

        lines = [
            f"Healthcare Event Data Summary",
            f"{'=' * 40}",
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

        if "facility" in df.columns:
            n_fac = df["facility"].nunique()
            lines.append(f"Unique facilities: {n_fac:,}")

        if "unit" in df.columns:
            n_units = df["unit"].nunique()
            lines.append(f"Unique units: {n_units:,}")

        if "narrative" in df.columns:
            word_counts = df["narrative"].str.split().str.len()
            lines.append(
                f"Narrative word count: "
                f"mean={word_counts.mean():.0f}, "
                f"median={word_counts.median():.0f}, "
                f"min={word_counts.min()}, "
                f"max={word_counts.max()}"
            )
            lines.append(
                f"Events with descriptions: "
                f"{(df['narrative_description'] != '').sum():,}"
            )
            lines.append(
                f"Events with suggestions: "
                f"{(df['narrative_suggestions'] != '').sum():,}"
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
            # puts all formats to string to avoid parquet errors
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
        """
        Run the full pipeline: load -> clean -> save.

        Returns dict of DataFrames produced.
        """
        results = {}

        # Events
        if self.events_path:
            events_df = self.load_events()
            results["events"] = events_df
            if save:
                self.save(events_df, "events")
            logger.info(self.summary(events_df))
        else:
            logger.warning("No events path specified; skipping events")

        # Operational data placeholder
        # (Future: load_operational() similar to rail pipeline)

        return results


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Healthcare Event Reporting (HERO) Data Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with defaults
  python healthcare_data_pipeline.py --events hero_events.xlsx --output data/processed/hero

  # Override column names
  python healthcare_data_pipeline.py \\
      --events hero_events.xlsx \\
      --event-id-col "Report Number" \\
      --facility-col "Hospital" \\
      --unit-col "Department" \\
      --event-date-col "Date" \\
      --description-col "Description" \\
      --suggestions-col "Recommendations"

  # CSV output with summary
  python healthcare_data_pipeline.py \\
      --events hero_events.xlsx \\
      --format csv \\
      --summary
        """,
    )

    # Required
    parser.add_argument("--events", required=True, help="Path to event data file (.xlsx or .csv)")

    # Output
    parser.add_argument("--output", default="data/processed/hero", help="Output directory")
    parser.add_argument("--format", default="parquet", choices=["parquet", "csv", "feather"])

    # Column name overrides
    parser.add_argument("--event-id-col", default=None, help="Event ID column name")
    parser.add_argument("--facility-col", default=None, help="Facility/org column name")
    parser.add_argument("--unit-col", default=None, help="Unit/location column name")
    parser.add_argument("--event-date-col", default=None, help="Event date column name")
    parser.add_argument("--description-col", default=None, help="Event description column name")
    parser.add_argument("--suggestions-col", default=None, help="Suggestions column name")

    # Cleaning
    parser.add_argument("--min-words", type=int, default=5, help="Min narrative word count")
    parser.add_argument("--min-chars", type=int, default=20, help="Min narrative char count")
    parser.add_argument("--no-clean", action="store_true", help="Skip cleaning step")

    # Display
    parser.add_argument("--summary", action="store_true", help="Print summary after processing")
    parser.add_argument("--head", type=int, default=0, help="Print first N rows")

    args = parser.parse_args()

    # Build config with overrides
    config = HeroConfig(output_format=args.format)
    config.min_narrative_words = args.min_words
    config.min_narrative_chars = args.min_chars

    if args.event_id_col:
        config.event_id_col = args.event_id_col
    if args.facility_col:
        config.facility_col = args.facility_col
    if args.unit_col:
        config.unit_col = args.unit_col
    if args.event_date_col:
        config.event_date_col = args.event_date_col
    if args.description_col:
        config.description_col = args.description_col
    if args.suggestions_col:
        config.suggestions_col = args.suggestions_col

    # Run
    pipeline = HealthcareDataPipeline(
        events_path=args.events,
        output_dir=args.output,
        config=config,
    )

    results = pipeline.run(save=True)

    if args.summary and "events" in results:
        print("\n" + pipeline.summary())

    if args.head > 0 and "events" in results:
        print(f"\nFirst {args.head} rows:")
        cols = ["eid", "raw_event_id", "event_date", "facility", "unit", "narrative"]
        display_cols = [c for c in cols if c in results["events"].columns]
        print(results["events"][display_cols].head(args.head).to_string())


if __name__ == "__main__":
    main()
