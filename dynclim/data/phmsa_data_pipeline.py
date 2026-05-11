"""
PHMSA Pipeline Incident/Accident Data Pipeline
===============================================

Single-file pipeline for loading, cleaning, and preparing PHMSA accident and
incident data for safety climate feature extraction.

Data Sources — Modern schema (Era 4, 2010+):
- accident_hazardous_liquid_jan2010_present.txt
- incident_gas_transmission_gathering_jan2010_present.txt
- incident_gas_distribution_jan2010_present.txt
- accident_gravity_reporting_regulated_jul2020_present.txt

Data Sources — Historical (Era 3, 2002–2009 and Era 2, 1984–2002):
- accident_hazardous_liquid_jan2002_dec2009.txt       (Era 3)
- incident_gas_transmission_gathering_2002_dec2009.txt (Era 3)
- incident_gas_distribution_mar2004_dec2009.txt        (Era 3)
- accident_hazardous_liquid_1986_jan2002.txt           (Era 2)
- incident_gas_distribution_mid1984_feb2004.txt        (Era 2)
- incident_gas_transmission_gathering_mid1984_2001.txt (Era 2)

All files are merged into a single dataset. Each record carries `era` (2, 3, or 4)
and `narrative_quality` flags to support downstream filtering.

Comparability notes (see project memory for full analysis):
- Era 4: rich NARRATIVE + detailed cause/cost breakdowns — most complete
- Era 3: NARRATIVE field directly comparable to Era 4; supplementary _TEXT fields
  appended when NARRATIVE is null; cost is property damage only (PRPTY)
- Era 2: NARRATIVE field present but shorter/less standardised; OPID operator ID
  uses a different numbering system from modern OPERATOR_ID; single FAT/INJ total
  only (no employee/contractor/GP breakdown); cost is property damage only
- Era 2 HL uses OPID not OPERATOR_ID — cross-era operator joins should use NAME

narrative_quality values:
  'modern'        — Era 4 free-text NARRATIVE
  'era3'          — Era 3 NARRATIVE field (comparable to modern)
  'era3_fallback' — Era 3, NARRATIVE was null; supplementary _TEXT fields used
  'era2'          — Era 2 NARRATIVE (shorter, less standardised)

Outputs:
- events.parquet: Cleaned, merged event records ready for NLP feature extraction

Usage:
    from phmsa_data_pipeline import PHMSADataPipeline

    # Modern only (default)
    pipeline = PHMSADataPipeline(
        base_data_dir="data/raw/phmsa/accident_incident_data"
    )

    # Include historical
    pipeline = PHMSADataPipeline(
        base_data_dir="data/raw/phmsa/accident_incident_data",
        include_historical=True,
    )

    events_df = pipeline.load_events()
    pipeline.run()
    report_data = pipeline.get_report_data()  # [(eid, narrative), ...]

CLI:
    # Auto-discover all files from base directory
    python phmsa_data_pipeline.py \\
        --base-dir data/raw/phmsa/accident_incident_data \\
        --include-historical \\
        --output data/processed/phmsa \\
        --summary
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union, List, Dict, Tuple, Any
from dataclasses import dataclass, field
import logging
import uuid
import hashlib

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# File registries
# =============================================================================

# Era 4 — modern schema (2010+). Key: label used in pipeline_type column.
MODERN_FILES: Dict[str, str] = {
    "hazardous_liquid":
        "accident_hazardous_liquid_jan2010_present/accident_hazardous_liquid_jan2010_present.txt",
    "gas_transmission":
        "incident_gas_transmission_gathering_jan2010_present/incident_gas_transmission_gathering_jan2010_present.txt",
    "gas_distribution":
        "incident_gas_distribution_jan2010_present 2/incident_gas_distribution_jan2010_present.txt",
    "gravity_regulated":
        "accident_gravity_reporting_regulated_jul2020_present/accident_gravity_reporting_regulated_jul2020_present.txt",
}

# Era 2–3 — historical schemas.
# Each entry describes the file's schema so _standardize_historical() can map
# columns without needing bespoke per-file logic.
#
# date_format:
#   'slash'    — M/D/YYYY string  (Era 3)
#   'yyyymmdd' — YYYYMMDD integer (Era 2)
#
# operator_id_system:
#   'operator_id' — same numbering as modern OPERATOR_ID (Era 3 + modern)
#   'opid'        — legacy OPID numbering (Era 2; do NOT join to modern operator data)
HISTORICAL_FILES: Dict[str, Dict[str, Any]] = {
    # ---- Era 3 ----------------------------------------------------------------
    "hazardous_liquid_2002_2009": {
        "rel_path":    "accident_hazardous_liquid_jan2002_dec2009/accident_hazardous_liquid_jan2002_dec2009.txt",
        "pipeline_type": "hazardous_liquid",
        "era":         3,
        "event_id_col":     "RPTID",
        "date_col":         "IDATE",
        "date_format":      "slash",
        "operator_id_col":  "OPERATOR_ID",
        "operator_id_system": "operator_id",
        "operator_name_col":"NAME",
        "operator_state_col":"OPSTATE",
        "fatality_col":     "FATAL",
        "injury_col":       "INJURE",
        "cost_col":         "PRPTY",
        # Only one supplementary text field in this file
        "supplementary_text": ["EVAC_REASON_TEXT"],
    },
    "gas_transmission_2002_2009": {
        "rel_path":    "incident_gas_transmission_gathering_2002_dec2009/incident_gas_transmission_gathering_2002_dec2009.txt",
        "pipeline_type": "gas_transmission",
        "era":         3,
        "event_id_col":     "RPTID",
        "date_col":         "IDATE",
        "date_format":      "slash",
        "operator_id_col":  "OPERATOR_ID",
        "operator_id_system": "operator_id",
        "operator_name_col":"NAME",
        "operator_state_col":"OPSTATE",
        "fatality_col":     "FATAL",
        "injury_col":       "INJURE",
        "cost_col":         "PRPTY",
        "supplementary_text": ["CAUSE_TEXT", "CAUSE_DETAILS_TEXT", "EVAC_REASON_TEXT"],
    },
    "gas_distribution_2004_2009": {
        "rel_path":    "incident_gas_distribution_mar2004_dec2009/incident_gas_distribution_mar2004_dec2009.txt",
        "pipeline_type": "gas_distribution",
        "era":         3,
        "event_id_col":     "RPTID",
        "date_col":         "IDATE",
        "date_format":      "slash",
        "operator_id_col":  "OPERATOR_ID",
        "operator_id_system": "operator_id",
        "operator_name_col":"NAME",
        "operator_state_col":"OPSTATE",
        "fatality_col":     "FATAL",
        "injury_col":       "INJURE",
        "cost_col":         "PRPTY",
        "supplementary_text": ["CAUSE_TEXT", "CAUSE_DETAILS_TEXT", "EVAC_REASON_TEXT"],
    },
    # ---- Era 2 ----------------------------------------------------------------
    "hazardous_liquid_1986_2002": {
        "rel_path":    "accident_hazardous_liquid_1986_jan2002/accident_hazardous_liquid_1986_jan2002.txt",
        "pipeline_type": "hazardous_liquid",
        "era":         2,
        "event_id_col":     "RPTID",
        "date_col":         "IDATE",
        "date_format":      "yyyymmdd",
        "operator_id_col":  "OPID",
        "operator_id_system": "opid",      # different numbering — flag for users
        "operator_name_col":"NAME",
        "operator_state_col":"ACSTATE",
        "fatality_col":     "TFAT",        # TFAT = total fatalities
        "injury_col":       "TINJ",        # TINJ = total injuries
        "cost_col":         "PRPTY",
        "supplementary_text": [],
    },
    "gas_distribution_1984_2004": {
        "rel_path":    "incident_gas_distribution_mid1984_feb2004/incident_gas_distribution_mid1984_feb2004.txt",
        "pipeline_type": "gas_distribution",
        "era":         2,
        "event_id_col":     "RPTID",
        "date_col":         "IDATE",
        "date_format":      "yyyymmdd",
        "operator_id_col":  "OPID",
        "operator_id_system": "opid",
        "operator_name_col":"NAME",
        "operator_state_col": None,        # no state field in this file
        "fatality_col":     "FAT",         # single total field
        "injury_col":       "INJ",
        "cost_col":         "PRPTY",
        "supplementary_text": [],
    },
    "gas_transmission_1984_2001": {
        "rel_path":    "incident_gas_transmission_gathering_mid1984_2001/incident_gas_transmission_gathering_mid1984_2001.txt",
        "pipeline_type": "gas_transmission",
        "era":         2,
        "event_id_col":     "RPTID",
        "date_col":         "IDATE",
        "date_format":      "yyyymmdd",
        "operator_id_col":  "OPID",
        "operator_id_system": "opid",
        "operator_name_col":"NAME",
        "operator_state_col":"ACSTATE",
        "fatality_col":     "FAT",
        "injury_col":       "INJ",
        "cost_col":         "PRPTY",
        "supplementary_text": [],
    },
    # ---- Era 1 ----------------------------------------------------------------
    # Pre-1986 / pre-1984 RSPA-era reports. Schema differs from Era 2-3:
    # casualties are split across multiple columns (employee vs non-employee
    # rather than a single TFAT/FAT). _standardize_historical() handles list-
    # typed fatality_col/injury_col/cost_col by summing the listed columns.
    # Operator IDs use the same legacy numbering as Era 2 (OPID-style; not
    # joinable to modern OPERATOR_ID), except hazardous_liquid pre-1986 which
    # already uses the modern OPERATOR_ID system.
    "gas_distribution_1970_1984": {
        "rel_path":    "incident_gas_distribution_1970_mid1984/incident_gas_distribution_1970_mid1984.txt",
        "pipeline_type": "gas_distribution",
        "era":         1,
        "event_id_col":     "REPORT_ID",
        "date_col":         "INCIDENT_DATE",
        "date_format":      "slash",
        "operator_id_col":  "OPERATOR_CODE",
        "operator_id_system": "opid",
        "operator_name_col":"OPERATOR_NAME",
        "operator_state_col":"INCIDENT_STATE",
        "fatality_col":     ["EMPLOYEE_FATALITIES", "NON_EMPLOYEE_FATALITIES"],
        "injury_col":       ["EMPLOYEE_INJURIES", "NON_EMPLOYEE_INJURIES"],
        "cost_col":         "OPERATORS_PROPERTY_DAMAGE",
        "supplementary_text": [],
    },
    "gas_transmission_1970_1984": {
        "rel_path":    "incident_gas_transmission_gathering_1970_mid1984/incident_gas_transmission_gathering_1970_mid1984.txt",
        "pipeline_type": "gas_transmission",
        "era":         1,
        "event_id_col":     "REPORT_ID",
        "date_col":         "INCIDENT_DATE",
        "date_format":      "slash",
        "operator_id_col":  "OPERATOR_CODE",
        "operator_id_system": "opid",
        "operator_name_col":"OPERATOR_NAME",
        "operator_state_col":"INCIDENT_STATE",
        # GT 1970-1984 only has EMPLOYEE_FATALITIES (no separate non-employee
        # fatality column); single-element list still passes through the
        # multi-col summation path uniformly.
        "fatality_col":     ["EMPLOYEE_FATALITIES"],
        "injury_col":       ["EMPLOYEE_INJURIES", "NON_EMPLOYEE_INJURIES"],
        "cost_col":         "OPERATORS_PROP_DAMAGE",
        "supplementary_text": [],
    },
    "hazardous_liquid_pre1986": {
        "rel_path":    "accident_hazardous_liquid_pre1986/accident_hazardous_liquid_pre1986.txt",
        "pipeline_type": "hazardous_liquid",
        "era":         1,
        "event_id_col":     "REPORT_ID",
        "date_col":         "ACCIDENT_DATE",
        "date_format":      "slash",
        # HL pre-1986 already uses the modern OPERATOR_ID numbering scheme
        "operator_id_col":  "OPERATOR_ID",
        "operator_id_system": "operator_id",
        "operator_name_col":"OPERATOR_NAME",
        "operator_state_col":"ACCIDENT_STATE",
        # Use the explicit TOTAL_* columns when present; pre-summed in source
        "fatality_col":     "TOTAL_FATALITIES",
        "injury_col":       "TOTAL_INJURIES",
        "cost_col":         "TOTAL_PRPTY_DAMAGE",
        "supplementary_text": ["PRIMARY_CAUSE_TEXT"],
    },
}


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class PHMSAConfig:
    """Configuration for PHMSA accident/incident data pipeline."""

    # --- Modern (Era 4) column names ---
    report_number_col:      str = "REPORT_NUMBER"
    supplemental_number_col: str = "SUPPLEMENTAL_NUMBER"
    narrative_col:          str = "NARRATIVE"
    date_col:               str = "LOCAL_DATETIME"
    date_alts: List[str] = field(
        default_factory=lambda: [
            "LOCAL_DATETIME", "REPORT_RECEIVED_DATE", "INCIDENT_IDENTIFIED_DATETIME"
        ]
    )
    operator_id_col:    str = "OPERATOR_ID"
    operator_name_col:  str = "NAME"
    operator_type_col:  str = "OPERATOR_TYPE"
    operator_state_col: str = "OPERATOR_STATE_ABBREVIATION"
    operator_city_col:  str = "OPERATOR_CITY_NAME"

    fatality_cols: List[str] = field(
        default_factory=lambda: [
            "NUM_EMP_FATALITIES", "NUM_CONTR_FATALITIES", "NUM_ER_FATALITIES",
            "NUM_WORKER_FATALITIES", "NUM_GP_FATALITIES",
        ]
    )
    injury_cols: List[str] = field(
        default_factory=lambda: [
            "NUM_EMP_INJURIES", "NUM_CONTR_INJURIES", "NUM_ER_INJURIES",
            "NUM_WORKER_INJURIES", "NUM_GP_INJURIES", "NUM_INJURED_TREATED_BY_EMT",
        ]
    )
    cost_cols_shared: List[str] = field(
        default_factory=lambda: [
            "EST_COST_OPER_PAID", "EST_COST_PROP_DAMAGE",
            "EST_COST_EMERGENCY",  "EST_COST_OTHER",
        ]
    )
    cost_cols_optional: List[str] = field(
        default_factory=lambda: [
            "EST_COST_GAS_RELEASED", "EST_COST_ENVIRONMENTAL",
            "EST_COST_UNINTENTIONAL_RELEASE", "EST_COST_INTENTIONAL_RELEASE",
        ]
    )
    commodity_col: str = "COMMODITY_RELEASED_TYPE"

    # --- Cleaning ---
    min_narrative_words: int = 5
    min_narrative_chars: int = 20

    # --- Output ---
    output_format: str = "parquet"

    # --- Domain ---
    domain: str = "phmsa"


# =============================================================================
# Module-level helpers
# =============================================================================

def _make_eid(pipeline_type: str, report_id: str, domain: str = "phmsa") -> str:
    """Deterministic UUID-5 scoped to (domain, pipeline_type, report_id)."""
    ns = uuid.UUID("a3c1e7d0-4b2f-4e8a-9c6d-1f0e2d3b4a5c")
    return str(uuid.uuid5(ns, f"{domain}:{pipeline_type}:{report_id}"))


def _make_org_id(raw_id: str, level: str = "operator") -> str:
    """Deterministic short hash for operator identifiers."""
    h = hashlib.sha256(f"{level}:{raw_id}".encode()).hexdigest()[:12]
    return f"{level[:3]}_{h}"


def _sum_numeric_cols(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    """Row-wise sum of cols that may or may not exist; missing → 0."""
    present = [c for c in cols if c in df.columns]
    if not present:
        return pd.Series(0, index=df.index)
    return df[present].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)


def _clean_text(series: pd.Series) -> pd.Series:
    """Strip, collapse whitespace, and remove nan/None artefacts."""
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.replace(r"^\s*None\s*$", "", regex=True)
        .str.replace(r"^\s*nan\s*$",  "", regex=True)
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


def _parse_historical_date(series: pd.Series, fmt: str) -> pd.Series:
    """
    Parse historical IDATE column.

    fmt='slash'    — M/D/YYYY strings  (Era 3)
    fmt='yyyymmdd' — YYYYMMDD integers (Era 2)
    """
    if fmt == "slash":
        return pd.to_datetime(series, errors="coerce")
    elif fmt == "yyyymmdd":
        # Coerce to string, zero-pad to 8 chars, then parse
        s = series.astype(str).str.strip().str.zfill(8)
        return pd.to_datetime(s, format="%Y%m%d", errors="coerce")
    else:
        return pd.to_datetime(series, errors="coerce")


# =============================================================================
# Core Pipeline
# =============================================================================

class PHMSADataPipeline:
    """
    Pipeline for PHMSA pipeline accident/incident data.

    By default loads only the four modern (Era 4, 2010+) files.
    Set include_historical=True to also merge Era 3 (2002–2009) and
    Era 2 (1984–2002) records, each tagged with `era` and `narrative_quality`.

    All eras produce the same standardised output columns:
        eid, raw_event_id, pipeline_type, era, narrative_quality,
        event_date, operator, operator_raw_id, operator_org_id,
        operator_id_system, operator_state, narrative,
        total_fatalities, total_injuries, total_cost
    """

    def __init__(
        self,
        # --- Modern file paths (explicit overrides) ---
        hazardous_liquid_path:  Optional[Union[str, Path]] = None,
        gas_transmission_path:  Optional[Union[str, Path]] = None,
        gas_distribution_path:  Optional[Union[str, Path]] = None,
        gravity_regulated_path: Optional[Union[str, Path]] = None,
        # --- Historical file paths (explicit overrides, keyed by HISTORICAL_FILES) ---
        historical_paths: Optional[Dict[str, Union[str, Path]]] = None,
        # --- Convenience: auto-discover from base directory ---
        base_data_dir: Optional[Union[str, Path]] = None,
        # --- Controls ---
        include_historical: bool = False,
        min_era: int = 2,          # 1, 2, or 3; ignored when include_historical=False
        output_dir: Union[str, Path] = "data/processed/phmsa",
        config: Optional[PHMSAConfig] = None,
    ):
        self.config = config or PHMSAConfig()
        self.output_dir = Path(output_dir)
        self.include_historical = include_historical
        self.min_era = min_era
        self._events_df: Optional[pd.DataFrame] = None

        # --- Resolve modern paths ---
        self._modern_paths: Dict[str, Optional[Path]] = {
            "hazardous_liquid":  Path(hazardous_liquid_path)  if hazardous_liquid_path  else None,
            "gas_transmission":  Path(gas_transmission_path)  if gas_transmission_path  else None,
            "gas_distribution":  Path(gas_distribution_path)  if gas_distribution_path  else None,
            "gravity_regulated": Path(gravity_regulated_path) if gravity_regulated_path else None,
        }

        # --- Resolve historical paths ---
        self._historical_paths: Dict[str, Optional[Path]] = {
            key: Path(historical_paths[key]) if historical_paths and key in historical_paths else None
            for key in HISTORICAL_FILES
        }

        # Auto-discover from base dir
        if base_data_dir:
            base = Path(base_data_dir)
            for ptype, rel in MODERN_FILES.items():
                if self._modern_paths[ptype] is None:
                    candidate = base / rel
                    if candidate.exists():
                        self._modern_paths[ptype] = candidate

            for key, meta in HISTORICAL_FILES.items():
                if self._historical_paths[key] is None:
                    candidate = base / meta["rel_path"]
                    if candidate.exists():
                        self._historical_paths[key] = candidate

    # =========================================================================
    # Loading
    # =========================================================================

    def _load_raw(self, path: Path, label: str) -> pd.DataFrame:
        """Read a single TSV file."""
        logger.info(f"Loading {label} from {path.name}")
        df = pd.read_csv(path, sep="\t", low_memory=False, encoding="latin-1")
        logger.info(f"  {len(df):,} rows, {len(df.columns)} columns")
        return df

    def load_events(self, clean: bool = True) -> pd.DataFrame:
        """
        Load, merge, standardise, and optionally clean all configured files.

        Returns a DataFrame with standardised columns across all eras.
        Original columns are preserved for modern (Era 4) records; historical
        records carry only standardised columns to avoid a sparse mega-table.
        """
        frames: List[pd.DataFrame] = []

        # --- Modern files (Era 4) ---
        for ptype, path in self._modern_paths.items():
            if path is None:
                logger.warning(f"No path for modern/{ptype}; skipping")
                continue
            if not path.exists():
                logger.warning(f"File not found: {path}; skipping")
                continue
            raw = self._load_raw(path, f"modern/{ptype}")
            raw["pipeline_type"] = ptype
            standardised = self._standardize_modern(raw)
            frames.append(standardised)

        # --- Historical files (Era 2–3) ---
        if self.include_historical:
            for key, meta in HISTORICAL_FILES.items():
                if meta["era"] < self.min_era:
                    continue
                path = self._historical_paths.get(key)
                if path is None:
                    logger.warning(f"No path for historical/{key}; skipping")
                    continue
                if not path.exists():
                    logger.warning(f"File not found: {path}; skipping")
                    continue
                raw = self._load_raw(path, f"era{meta['era']}/{key}")
                standardised = self._standardize_historical(raw, meta)
                frames.append(standardised)

        if not frames:
            raise RuntimeError("No source files loaded. Check paths.")

        df = pd.concat(frames, ignore_index=True, sort=False)
        logger.info(f"Merged {len(frames)} files → {len(df):,} total rows")

        if clean:
            df = self.clean_events(df)

        self._events_df = df
        return df

    # =========================================================================
    # Standardisation — Era 4 (modern)
    # =========================================================================

    def _standardize_modern(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardise a modern (Era 4) file.
        Preserves all original columns and adds standardised ones.
        """
        cfg = self.config
        df = df.copy()

        # era / narrative_quality metadata
        df["era"] = 4
        df["narrative_quality"] = "modern"

        # Event ID
        if cfg.report_number_col in df.columns:
            df["raw_event_id"] = df[cfg.report_number_col].astype(str).str.strip()
        else:
            logger.warning(f"'{cfg.report_number_col}' not found; generating IDs")
            df["raw_event_id"] = [f"PHMSA_{i:07d}" for i in range(len(df))]

        df["eid"] = df.apply(
            lambda r: _make_eid(r.get("pipeline_type", "unknown"),
                                r["raw_event_id"], cfg.domain),
            axis=1,
        )

        # Date
        date_col = next(
            (c for c in [cfg.date_col] + cfg.date_alts if c in df.columns), None
        )
        if date_col:
            df["event_date"] = pd.to_datetime(df[date_col], errors="coerce")
        else:
            df["event_date"] = pd.NaT
            logger.warning("No date column found in modern file")

        # Operator
        df["operator"] = (
            _clean_text(df[cfg.operator_name_col]) if cfg.operator_name_col in df.columns
            else ""
        )
        if cfg.operator_id_col in df.columns:
            df["operator_raw_id"] = df[cfg.operator_id_col].astype(str).str.strip()
        else:
            df["operator_raw_id"] = ""
        df["operator_org_id"] = df["operator_raw_id"].apply(
            lambda x: _make_org_id(x, "operator") if x and x not in ("nan", "") else ""
        )
        df["operator_id_system"] = "operator_id"
        df["operator_state"] = (
            _clean_text(df[cfg.operator_state_col]) if cfg.operator_state_col in df.columns
            else ""
        )

        # Narrative
        df["narrative"] = (
            _clean_text(df[cfg.narrative_col]) if cfg.narrative_col in df.columns
            else ""
        )

        # Aggregate casualties and costs
        df["total_fatalities"] = _sum_numeric_cols(df, cfg.fatality_cols)
        df["total_injuries"]   = _sum_numeric_cols(df, cfg.injury_cols)
        df["total_cost"]       = _sum_numeric_cols(
            df, cfg.cost_cols_shared + cfg.cost_cols_optional
        )

        return df

    # =========================================================================
    # Standardisation — Era 2–3 (historical)
    # =========================================================================

    def _standardize_historical(
        self, df: pd.DataFrame, meta: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Standardise a historical (Era 2 or 3) file using per-file schema metadata.

        Returns only the standardised columns (no raw column preservation) to
        avoid creating a massive sparse merged table.
        """
        era          = meta["era"]
        ptype        = meta["pipeline_type"]
        date_fmt     = meta["date_format"]
        op_id_col    = meta["operator_id_col"]
        op_id_system = meta["operator_id_system"]
        op_name_col  = meta["operator_name_col"]
        op_state_col = meta.get("operator_state_col")
        fat_col      = meta["fatality_col"]
        inj_col      = meta["injury_col"]
        cost_col     = meta["cost_col"]
        narr_col     = "NARRATIVE"
        supp_cols    = meta.get("supplementary_text", [])

        out = pd.DataFrame(index=df.index)

        # --- Metadata ---
        out["pipeline_type"] = ptype
        out["era"] = era

        # --- Event ID ---
        if meta["event_id_col"] in df.columns:
            out["raw_event_id"] = df[meta["event_id_col"]].astype(str).str.strip()
        else:
            out["raw_event_id"] = [f"HIST_{era}_{i:07d}" for i in range(len(df))]

        out["eid"] = out.apply(
            lambda r: _make_eid(
                f"{ptype}_era{era}", r["raw_event_id"], self.config.domain
            ),
            axis=1,
        )

        # --- Date ---
        if meta["date_col"] in df.columns:
            out["event_date"] = _parse_historical_date(df[meta["date_col"]], date_fmt)
            n_bad = out["event_date"].isna().sum()
            if n_bad > 0:
                logger.warning(
                    f"  {n_bad:,} unparseable dates in {ptype} era{era}"
                )
        else:
            out["event_date"] = pd.NaT

        # --- Operator ---
        out["operator"] = (
            _clean_text(df[op_name_col]) if op_name_col in df.columns else ""
        )
        out["operator_raw_id"] = (
            df[op_id_col].astype(str).str.strip()
            if op_id_col in df.columns
            else ""
        )
        out["operator_org_id"] = out["operator_raw_id"].apply(
            lambda x: _make_org_id(x, "operator") if x and x not in ("nan", "") else ""
        )
        out["operator_id_system"] = op_id_system   # 'operator_id' or 'opid'
        out["operator_state"] = (
            _clean_text(df[op_state_col])
            if op_state_col and op_state_col in df.columns
            else ""
        )

        # --- Narrative + narrative_quality ---
        narr_series = (
            _clean_text(df[narr_col]) if narr_col in df.columns
            else pd.Series("", index=df.index)
        )

        # Build supplementary text (concatenate non-empty fields in order)
        supp_present = [c for c in supp_cols if c in df.columns]
        if supp_present:
            supp_series = (
                df[supp_present]
                .fillna("")
                .astype(str)
                .apply(
                    lambda row: " ".join(
                        p.strip() for p in row if p.strip()
                        and p.strip().lower() not in ("nan", "none", "")
                    ),
                    axis=1,
                )
            )
        else:
            supp_series = pd.Series("", index=df.index)

        # Combine: use NARRATIVE as primary; fall back to supplementary text
        era_label = f"era{era}"

        def _resolve_narrative(narr: str, supp: str) -> Tuple[str, str]:
            """Return (narrative_text, quality_flag)."""
            if narr and len(narr.split()) >= 3:
                return narr, era_label
            if supp and len(supp.split()) >= 3:
                # Supplementary fallback flagged with era + _fallback (Era 3
                # historical preserves the original "era3_fallback" label).
                return supp, f"{era_label}_fallback"
            return narr, era_label

        resolved = [
            _resolve_narrative(n, s)
            for n, s in zip(narr_series, supp_series)
        ]
        out["narrative"]        = [r[0] for r in resolved]
        out["narrative_quality"] = [r[1] for r in resolved]

        # --- Casualties ---
        # Era 2-3 use a single total field (TFAT/FAT). Era 1 splits across
        # employee vs non-employee columns; meta passes a list of columns to
        # sum in that case. _coerce_sum() unifies both paths.
        def _coerce_sum(col_spec) -> pd.Series:
            if isinstance(col_spec, (list, tuple)):
                cols = [c for c in col_spec if c in df.columns]
                if not cols:
                    return pd.Series(0, index=df.index)
                return sum(pd.to_numeric(df[c], errors="coerce").fillna(0)
                           for c in cols)
            else:
                if col_spec in df.columns:
                    return pd.to_numeric(df[col_spec], errors="coerce").fillna(0)
                return pd.Series(0, index=df.index)

        out["total_fatalities"] = _coerce_sum(fat_col)
        out["total_injuries"]   = _coerce_sum(inj_col)

        # --- Cost ---
        # Historical cost = property damage only (PRPTY); not comparable to
        # modern EST_COST_* breakdown, but at least recorded.
        out["total_cost"] = _coerce_sum(cost_col)

        # --- Incident year (useful for time-series filtering) ---
        out["iyear"] = (
            pd.to_numeric(df["IYEAR"], errors="coerce")
            if "IYEAR" in df.columns
            else pd.NA
        )

        return out

    # =========================================================================
    # Cleaning
    # =========================================================================

    def clean_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean merged event data.

        Steps:
        1. Remove rows with missing/empty narratives
        2. Filter short narratives (word count + char count thresholds)
        3. Validate UTF-8 encoding
        4. Deduplicate on (pipeline_type, era, raw_event_id)
        """
        initial = len(df)
        df = df.copy()

        if "narrative" in df.columns:
            df = df[df["narrative"].notna()]
            df = df[df["narrative"].astype(str).str.strip() != ""]
            df = df[df["narrative"].str.lower() != "none"]

            df["_wc"] = df["narrative"].str.split().str.len()
            df = df[df["_wc"] >= self.config.min_narrative_words]
            df = df[df["narrative"].str.len() >= self.config.min_narrative_chars]
            df = df.drop(columns=["_wc"])

            df = df[df["narrative"].apply(_is_valid_utf8)]

        # Deduplicate: within each (pipeline_type, era) pair
        dedup_cols = [c for c in ["pipeline_type", "era", "raw_event_id"] if c in df.columns]
        if dedup_cols:
            n_before = len(df)
            df = df.drop_duplicates(subset=dedup_cols, keep="first")
            n_dupes = n_before - len(df)
            if n_dupes:
                logger.info(f"Removed {n_dupes:,} duplicate events")

        logger.info(
            f"Cleaning: {initial:,} → {len(df):,} events "
            f"({initial - len(df):,} removed)"
        )
        return df.reset_index(drop=True)

    # =========================================================================
    # Accessors
    # =========================================================================

    def get_report_data(self) -> List[Tuple[str, str]]:
        """Return [(eid, narrative), ...] for NLP feature extraction."""
        if self._events_df is None:
            raise RuntimeError("No events loaded. Call load_events() first.")
        return [
            (row["eid"], row["narrative"])
            for _, row in self._events_df.iterrows()
            if row.get("narrative", "").strip()
        ]

    def get_events(self) -> pd.DataFrame:
        if self._events_df is None:
            raise RuntimeError("No events loaded. Call load_events() first.")
        return self._events_df.copy()

    # =========================================================================
    # Summary
    # =========================================================================

    def summary(self, df: Optional[pd.DataFrame] = None) -> str:
        df = df if df is not None else self._events_df
        if df is None:
            return "No data loaded."

        lines = [
            "PHMSA Accident/Incident Data Summary",
            "=" * 40,
            f"Total events: {len(df):,}",
        ]

        if "era" in df.columns:
            lines.append("Events by era:")
            for era, cnt in df["era"].value_counts().sort_index().items():
                lines.append(f"  Era {era}: {cnt:,}")

        if "narrative_quality" in df.columns:
            lines.append("Narrative quality breakdown:")
            for q, cnt in df["narrative_quality"].value_counts().items():
                lines.append(f"  {q}: {cnt:,}")

        if "pipeline_type" in df.columns:
            lines.append("Events by pipeline type:")
            for ptype, cnt in df["pipeline_type"].value_counts().items():
                lines.append(f"  {ptype}: {cnt:,}")

        if "event_date" in df.columns:
            valid = df["event_date"].dropna()
            if len(valid):
                lines.append(
                    f"Date range: {valid.min():%Y-%m-%d} to {valid.max():%Y-%m-%d}"
                )

        if "operator" in df.columns:
            lines.append(
                f"Unique operators: "
                f"{df['operator'].replace('', pd.NA).dropna().nunique():,}"
            )

        if "operator_state" in df.columns:
            lines.append(
                f"Unique states: "
                f"{df['operator_state'].replace('', pd.NA).dropna().nunique():,}"
            )

        if "total_fatalities" in df.columns:
            lines.append(
                f"Total fatalities: {int(df['total_fatalities'].sum()):,} "
                f"(events with fatalities: {(df['total_fatalities'] > 0).sum():,})"
            )

        if "total_injuries" in df.columns:
            lines.append(
                f"Total injuries: {int(df['total_injuries'].sum()):,} "
                f"(events with injuries: {(df['total_injuries'] > 0).sum():,})"
            )

        if "total_cost" in df.columns:
            lines.append(f"Total reported cost: ${df['total_cost'].sum():,.0f}")

        if "narrative" in df.columns:
            wc = df["narrative"].str.split().str.len()
            lines.append(
                f"Narrative word count: mean={wc.mean():.0f}, "
                f"median={wc.median():.0f}, min={wc.min()}, max={wc.max()}"
            )

        return "\n".join(lines)

    # =========================================================================
    # Save
    # =========================================================================

    def save(self, df: Optional[pd.DataFrame] = None, filename: str = "events") -> Path:
        df = df if df is not None else self._events_df
        if df is None:
            raise RuntimeError("No data to save. Call load_events() first.")

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
        elif fmt == "feather":
            out_path = self.output_dir / f"{filename}.feather"
            df.to_feather(out_path)
        else:
            raise ValueError(f"Unsupported output format: {fmt}")

        logger.info(f"Saved {len(df):,} events to {out_path}")
        return out_path

    # =========================================================================
    # Full pipeline
    # =========================================================================

    def run(self, save: bool = True, save_unfiltered: bool = False) -> Dict[str, pd.DataFrame]:
        """Load → clean → save. Returns {'events': df}.

        When save_unfiltered=True, also saves an `events_all.parquet` containing
        merged-but-unfiltered events (no narrative filter, no dedup). Era 1
        records typically lack free-text narratives and would otherwise be
        excluded by clean_events; events_all is meant for downstream tabular
        / panel-rate analyses that need raw event counts per
        (operator, date, pipeline_type) regardless of narrative content.
        """
        # Load merged-but-unfiltered first if needed; clean independently
        # (clean_events doesn't mutate the input — it returns a new df)
        events_raw = self.load_events(clean=False)

        if save_unfiltered:
            self.save(events_raw, "events_all")

        events_df = self.clean_events(events_raw)
        self._events_df = events_df

        results = {"events": events_df, "events_all": events_raw}
        if save:
            self.save(events_df, "events")
        logger.info(self.summary(events_df))
        return results


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="PHMSA Pipeline Accident/Incident Data Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Modern only — auto-discover files from base dir
  python phmsa_data_pipeline.py \\
      --base-dir data/raw/phmsa/accident_incident_data \\
      --output data/processed/phmsa --summary

  # Include Era 2 and 3 historical files
  python phmsa_data_pipeline.py \\
      --base-dir data/raw/phmsa/accident_incident_data \\
      --include-historical --output data/processed/phmsa --summary

  # Include Era 3 only (skip Era 2)
  python phmsa_data_pipeline.py \\
      --base-dir data/raw/phmsa/accident_incident_data \\
      --include-historical --min-era 3 --output data/processed/phmsa --summary
        """,
    )

    # Data source
    parser.add_argument("--base-dir", default=None,
                        help="Base directory containing PHMSA accident/incident subdirectories")

    # Modern file overrides
    parser.add_argument("--hazardous-liquid",  default=None)
    parser.add_argument("--gas-transmission",  default=None)
    parser.add_argument("--gas-distribution",  default=None)
    parser.add_argument("--gravity-regulated", default=None)

    # Historical controls
    parser.add_argument("--include-historical", action="store_true",
                        help="Load Era 2 and/or Era 3 historical files")
    parser.add_argument("--min-era", type=int, default=2, choices=[1, 2, 3],
                        help="Minimum era to include when --include-historical is set (default 2; "
                             "set to 1 to also load Era 1 pre-1986/pre-1984 RSPA-era reports)")

    # Output
    parser.add_argument("--output", default="data/processed/phmsa")
    parser.add_argument("--format", default="parquet", choices=["parquet", "csv", "feather"])

    # Cleaning
    parser.add_argument("--min-words", type=int, default=5)
    parser.add_argument("--min-chars", type=int, default=20)
    parser.add_argument("--save-unfiltered", action="store_true",
                        help="Also save events_all.parquet with all merged events "
                             "(no narrative filter; useful for tabular/panel analyses "
                             "that need raw event counts, especially Era 1 reports "
                             "which typically lack free-text narratives).")

    # Display
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--head", type=int, default=0)

    args = parser.parse_args()

    config = PHMSAConfig(output_format=args.format)
    config.min_narrative_words = args.min_words
    config.min_narrative_chars = args.min_chars

    pipeline = PHMSADataPipeline(
        hazardous_liquid_path=args.hazardous_liquid,
        gas_transmission_path=args.gas_transmission,
        gas_distribution_path=args.gas_distribution,
        gravity_regulated_path=args.gravity_regulated,
        base_data_dir=args.base_dir,
        include_historical=args.include_historical,
        min_era=args.min_era,
        output_dir=args.output,
        config=config,
    )

    results = pipeline.run(save=True, save_unfiltered=args.save_unfiltered)

    if args.summary and "events" in results:
        print("\n" + pipeline.summary())

    if args.head > 0 and "events" in results:
        cols = ["eid", "raw_event_id", "pipeline_type", "era", "narrative_quality",
                "event_date", "operator", "operator_state", "total_fatalities",
                "total_injuries", "total_cost", "narrative"]
        display_cols = [c for c in cols if c in results["events"].columns]
        print(f"\nFirst {args.head} rows:")
        print(results["events"][display_cols].head(args.head).to_string())


if __name__ == "__main__":
    main()
