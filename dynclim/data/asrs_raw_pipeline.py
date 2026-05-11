"""
ASRS Raw Pipeline
==================

Reads the per-year ASRS DBOnline download files from
`data/raw/aviation/asrs_raw/`, combines them into a single canonical parquet,
and emits diagnostics on duplicates and date-range coverage.

ASRS DBOnline downloads are tab-delimited text files (with .xls extension)
having a two-row header — first row is the field category (Time, Place,
Aircraft 1, Person 1, etc.), second row is the sub-column name (Date,
Locale Reference, ATC / Advisory, etc.). The third row is empty before
data begins.

File-naming convention from the ASRS download UI: one or two files per year,
with optional `-1` / `-2` suffixes when the per-query record cap forces
splitting a year. Some years also have an unsuffixed file alongside the
splits — assume duplicates and dedupe by ACN.

Output (default `data/processed/aviation/asrs_combined.parquet`):
    snake_case columns matching the legacy `cmbd_ASRS.xlsx` naming where
    possible (e.g. `acn`, `time_date`, `place_locale_reference`,
    `person_1function`, `aircraft_1aircraft_operator`), plus a derived
    `cmbd_narrative` column that concatenates Report 1 + Report 2 narratives
    and Report 1 synopsis (the same combined-narrative field the downstream
    `aviation_data_pipeline.py` expects).

Diagnostics:
    - Duplicate ACNs (within and across files)
    - Missing year-months in the [min, max] range
    - Per-month report volume (flag thin months — possible incomplete pull)

CLI:
    python asrs_raw_pipeline.py \\
        --raw-dir data/raw/aviation/asrs_raw \\
        --output  data/processed/aviation/asrs_combined.parquet \\
        --summary
"""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Column-name normalization
# =============================================================================
# The downstream `aviation_data_pipeline.py` looks up canonical columns via
# `_resolve_col()` with multiple alternatives (e.g. acn / ACN, time_date /
# Date, etc.). We match the legacy `cmbd_ASRS.xlsx` snake_case convention so
# all alternates resolve cleanly. Examples from cmbd_ASRS:
#
#     "Aircraft 1" / "ATC / Advisory"  →  aircraft_1atc_advisory
#     "Time"       / "Date"            →  time_date
#     "Place"      / "Locale Reference"→  place_locale_reference
#     ""           / "ACN"             →  acn
#     "Person 1"   / "Function"        →  person_1function
#
# Rule (matches cmbd output):
#   1. Lowercase both category + sub-column
#   2. Replace any non-alphanumeric run with "_", strip leading/trailing "_"
#   3. Concatenate without an extra separator: cat + "_" + sub, then collapse
#      multiple "_" into single. Note that for cat ending in alnum + space +
#      digit (e.g. "aircraft 1" → "aircraft_1"), the trailing "_1" already
#      separates the boundary, producing "aircraft_1atc_advisory" rather
#      than "aircraft_1_atc_advisory" — we explicitly squash the extra "_"
#      between cat and sub when cat ends in a digit-or-alpha block.

def _slugify(s: str) -> str:
    s = str(s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def _flatten_header(category: str, subcol: str) -> str:
    cat = _slugify(category)
    sub = _slugify(subcol)
    if not cat and not sub:
        return ""
    if not cat:
        return sub
    if not sub:
        return cat
    # Match cmbd_ASRS quirk: when category ends in `_<digit>`, drop the
    # separator before the sub-column (so "aircraft_1" + "atc_advisory" =>
    # "aircraft_1atc_advisory"). Otherwise insert "_" between.
    if re.search(r"_\d+$", cat):
        return cat + sub
    return cat + "_" + sub


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ASRSRawConfig:
    # Read parameters
    n_header_rows: int = 2          # categories + sub-columns
    skip_rows_after_header: int = 1  # blank row before data

    # Narrative columns to combine into `cmbd_narrative`. Order matters —
    # we concatenate in this order and skip blanks. Names match the
    # post-flatten convention.
    narrative_components: List[str] = field(default_factory=lambda: [
        "report_1narrative",
        "report_2narrative",
        "report_1synopsis",
    ])

    # Date column once flattened (e.g. "time_date") — used for diagnostics
    # and for the year-month coverage check.
    date_col: str = "time_date"

    # Output format
    output_format: str = "parquet"


# =============================================================================
# Pipeline
# =============================================================================

class ASRSRawPipeline:

    def __init__(self,
                 raw_dir: Path | str,
                 output:  Path | str,
                 config:  Optional[ASRSRawConfig] = None):
        self.raw_dir = Path(raw_dir)
        self.output  = Path(output)
        self.config  = config or ASRSRawConfig()
        self._df: Optional[pd.DataFrame] = None
        self._diagnostics: Dict[str, object] = {}

    # -------------------------------------------------------------------------
    # File reading
    # -------------------------------------------------------------------------

    def _read_one(self, path: Path) -> Optional[pd.DataFrame]:
        cfg = self.config
        try:
            # Read the first n_header_rows + 1 lines to extract headers
            with open(path, "r", encoding="latin-1", errors="replace") as f:
                header_lines = [f.readline().rstrip("\n").split("\t")
                                for _ in range(cfg.n_header_rows)]
            # The TSV may have a trailing empty column from the line ending —
            # use the longest header-row width as the canonical column count.
            ncols = max(len(r) for r in header_lines)
            header_lines = [r + [""] * (ncols - len(r)) for r in header_lines]

            cats = header_lines[0]
            subs = header_lines[1]
            colnames = [_flatten_header(c, s) for c, s in zip(cats, subs)]

            # De-duplicate column names if the ASRS export ever produces
            # collisions (rare, but defensive).
            seen: Dict[str, int] = {}
            deduped: List[str] = []
            for c in colnames:
                if c == "":
                    c = "_unnamed"
                seen[c] = seen.get(c, 0)
                if seen[c] == 0:
                    deduped.append(c)
                else:
                    deduped.append(f"{c}_{seen[c]}")
                seen[c] += 1
            colnames = deduped

            df = pd.read_csv(
                path,
                sep="\t",
                skiprows=cfg.n_header_rows + cfg.skip_rows_after_header,
                names=colnames,
                dtype=str,
                encoding="latin-1",
                na_values=[""],
                keep_default_na=False,
                engine="python",
                on_bad_lines="warn",
            )
        except Exception as e:
            logger.warning(f"  Failed to read {path.name}: {e}")
            return None

        # Drop fully-empty rows (sometimes appear at end of file)
        df = df.dropna(how="all")

        df["__source_file"] = path.name
        return df

    def load_all(self) -> pd.DataFrame:
        files = sorted(self.raw_dir.glob("*.xls"))
        if not files:
            raise RuntimeError(f"No .xls files in {self.raw_dir}")
        logger.info(f"Found {len(files)} ASRS download files")

        frames: List[pd.DataFrame] = []
        per_file_counts: List[Tuple[str, int]] = []
        for f in files:
            df = self._read_one(f)
            if df is None or df.empty:
                continue
            per_file_counts.append((f.name, len(df)))
            logger.info(f"  {f.name}: {len(df):,} rows")
            frames.append(df)

        self._diagnostics["per_file_counts"] = per_file_counts

        if not frames:
            raise RuntimeError("No usable ASRS files loaded.")

        # Reconcile schema across files — different vintages occasionally have
        # extra/missing columns. Take the union; missing values become NaN.
        all_cols: List[str] = []
        for df in frames:
            for c in df.columns:
                if c not in all_cols:
                    all_cols.append(c)
        for i, df in enumerate(frames):
            for c in all_cols:
                if c not in df.columns:
                    df[c] = pd.NA
            frames[i] = df[all_cols]

        df = pd.concat(frames, ignore_index=True, sort=False)
        logger.info(f"Concatenated raw rows: {len(df):,}")
        return df

    # -------------------------------------------------------------------------
    # Cleaning + derivations
    # -------------------------------------------------------------------------

    def clean_and_derive(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config

        # --- ACN: must be present ---
        if "acn" not in df.columns:
            raise RuntimeError("Expected 'acn' column not found after flattening")
        df = df[df["acn"].notna() & (df["acn"].astype(str).str.strip() != "")]
        df["acn"] = df["acn"].astype(str).str.strip()

        # --- Combined narrative ---
        # Concatenate Report 1 narrative + Report 2 narrative + Report 1 synopsis,
        # skipping blanks. Column names already flattened to snake_case.
        narr_parts: List[pd.Series] = []
        for col in cfg.narrative_components:
            if col in df.columns:
                narr_parts.append(df[col].fillna("").astype(str).str.strip())
            else:
                logger.info(f"  Narrative component '{col}' not present in any file")
        if narr_parts:
            cmbd = narr_parts[0].copy()
            for s in narr_parts[1:]:
                cmbd = (cmbd.where(cmbd != "", "") + " " + s).str.strip()
            df["cmbd_narrative"] = cmbd.replace("", pd.NA)
        else:
            df["cmbd_narrative"] = pd.NA

        # --- Deduplicate by ACN ---
        # ASRS download splits sometimes overlap. Keep the row with the
        # most non-null fields (richer record) per ACN.
        n_before = len(df)
        df["_n_nonnull"] = df.notna().sum(axis=1)
        df = (df.sort_values(["acn", "_n_nonnull"], ascending=[True, False])
                .drop_duplicates(subset=["acn"], keep="first")
                .drop(columns=["_n_nonnull"])
                .reset_index(drop=True))
        n_dupes = n_before - len(df)
        self._diagnostics["dupes_removed"] = n_dupes
        logger.info(f"Dedup by ACN: {n_before:,} -> {len(df):,} "
                    f"({n_dupes:,} duplicates removed)")

        # --- Parse year/month from time_date ---
        # ASRS raw time_date is YYYYMM (e.g. "202401"). We rewrite the
        # `time_date` column to ISO "YYYY-MM-01" form so downstream
        # `pd.to_datetime(time_date)` (in aviation_data_pipeline.py) parses
        # unambiguously — pandas' heuristic infers wrong dates from bare
        # 6-digit strings (e.g. "201203" → 2003-12-20) and outright fails
        # on others (e.g. "198812"). Keeping ISO form avoids both.
        if cfg.date_col in df.columns:
            td = df[cfg.date_col].astype(str).str.strip().str.replace(r"[^0-9]", "", regex=True)
            # Only treat as YYYYMM if exactly 6 digits AND year sane (1980-2030).
            valid = td.str.len() == 6
            yr  = pd.to_numeric(td.where(valid).str.slice(0, 4), errors="coerce")
            mo  = pd.to_numeric(td.where(valid).str.slice(4, 6), errors="coerce")
            year_ok = yr.between(1980, 2030)
            month_ok = mo.between(1, 12)
            ok = year_ok & month_ok
            df["__year"]  = yr.where(ok)
            df["__month"] = mo.where(ok)
            n_bad = int((~ok).sum())
            if n_bad > 0:
                logger.info(f"  {n_bad:,} rows had unparseable time_date "
                            f"(treated as missing for coverage check)")
            # Rewrite time_date to ISO "YYYY-MM-01" where parseable; leave
            # original string where not (downstream code that does to_datetime
            # with errors='coerce' will simply NaT those — same behavior).
            iso = (
                yr.where(ok).astype("Int64").astype(str) + "-" +
                mo.where(ok).astype("Int64").astype(str).str.zfill(2) + "-01"
            )
            df[cfg.date_col] = iso.where(ok, df[cfg.date_col])
        else:
            df["__year"] = np.nan
            df["__month"] = np.nan

        return df

    # -------------------------------------------------------------------------
    # Coverage diagnostics
    # -------------------------------------------------------------------------

    def coverage_report(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build a per-year-month report-volume table; flag missing or
        suspiciously thin months."""
        if df["__year"].isna().all():
            logger.warning("No parsable dates; skipping coverage report")
            return pd.DataFrame()

        valid = df[df["__year"].notna() & df["__month"].notna()].copy()
        valid["__year"]  = valid["__year"].astype(int)
        valid["__month"] = valid["__month"].astype(int)
        valid = valid[(valid["__month"] >= 1) & (valid["__month"] <= 12)]

        ym_counts = (valid.groupby(["__year", "__month"]).size()
                          .rename("n_reports").reset_index())

        if ym_counts.empty:
            return ym_counts

        # Build complete year-month grid spanning observed range
        y_min, y_max = ym_counts["__year"].min(), ym_counts["__year"].max()
        full = pd.DataFrame([(y, m) for y in range(y_min, y_max + 1)
                                    for m in range(1, 13)],
                             columns=["__year", "__month"])
        report = full.merge(ym_counts, on=["__year", "__month"], how="left") \
                     .fillna({"n_reports": 0})
        report["n_reports"] = report["n_reports"].astype(int)

        # Clip the trailing edge: we don't expect months past the latest
        # actual report.
        latest = ym_counts.sort_values(["__year", "__month"]).tail(1).iloc[0]
        report = report[
            (report["__year"] < latest["__year"]) |
            ((report["__year"] == latest["__year"]) &
             (report["__month"] <= latest["__month"]))
        ].reset_index(drop=True)

        # Flag missing months (zero reports) and thin months (less than half
        # the local median over a ±6-month neighborhood, with an absolute
        # floor — many ASRS query downloads cap at ~5000 records, so a month
        # below half the local median AND below 100 reports likely indicates
        # an incomplete download for that period).
        report["missing"] = report["n_reports"] == 0
        med = report["n_reports"].rolling(13, center=True, min_periods=3).median()
        report["thin"] = (~report["missing"]) & \
                         (report["n_reports"] < 0.5 * med.fillna(0)) & \
                         (report["n_reports"] < 100)

        self._diagnostics["coverage_report"] = report
        n_missing = int(report["missing"].sum())
        n_thin    = int(report["thin"].sum())
        logger.info(f"Coverage: {len(report)} months in [{y_min}-{y_max}]: "
                    f"{n_missing} missing, {n_thin} thin")
        if n_missing > 0:
            miss_examples = report[report["missing"]].head(10)
            for _, r in miss_examples.iterrows():
                logger.warning(f"  MISSING month: {int(r['__year'])}-{int(r['__month']):02d}")
            if n_missing > 10:
                logger.warning(f"  ... ({n_missing - 10} more missing months)")
        if n_thin > 0:
            thin_examples = report[report["thin"]].head(10)
            for _, r in thin_examples.iterrows():
                logger.warning(
                    f"  THIN month: {int(r['__year'])}-{int(r['__month']):02d} "
                    f"({int(r['n_reports'])} reports — possible incomplete download)"
                )

        return report

    # -------------------------------------------------------------------------
    # I/O
    # -------------------------------------------------------------------------

    def save(self, df: pd.DataFrame, path: Optional[Path | str] = None) -> Path:
        path = Path(path) if path else self.output
        path.parent.mkdir(parents=True, exist_ok=True)

        # Drop scratch columns before writing
        cols_drop = [c for c in df.columns if c.startswith("__")]
        out = df.drop(columns=cols_drop)

        if self.config.output_format == "parquet":
            out.to_parquet(path, index=False, engine="pyarrow")
        elif self.config.output_format == "csv":
            out.to_csv(path, index=False)
        else:
            raise ValueError(f"Unsupported format: {self.config.output_format}")
        logger.info(f"Saved {len(out):,} rows to {path}")
        return path

    def run(self, save: bool = True) -> pd.DataFrame:
        raw = self.load_all()
        df  = self.clean_and_derive(raw)
        cov = self.coverage_report(df)
        self._df = df
        if save:
            self.save(df)
            # Also drop the per-month coverage CSV alongside the parquet for
            # easy inspection in spreadsheet tools.
            if not cov.empty:
                cov_path = self.output.with_suffix("").parent / \
                            (self.output.stem + "_coverage.csv")
                cov.to_csv(cov_path, index=False)
                logger.info(f"Saved coverage report to {cov_path}")
        return df

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    def summary(self) -> str:
        df = self._df
        if df is None:
            return "No data loaded."
        diag = self._diagnostics

        lines = ["ASRS Raw Pipeline Summary",
                 "=" * 40,
                 f"Total reports (post-dedup): {len(df):,}",
                 f"Duplicate ACNs removed:     {diag.get('dupes_removed', 0):,}"]

        if "report_1narrative" in df.columns:
            n_with_narr = df["cmbd_narrative"].notna().sum()
            lines.append(f"Reports with combined narrative: {n_with_narr:,} "
                         f"({100*n_with_narr/max(len(df),1):.1f}%)")

        if "__year" in df.columns and df["__year"].notna().any():
            yr_counts = df["__year"].dropna().astype(int).value_counts().sort_index()
            lines.append(f"Year span: {int(yr_counts.index.min())} – "
                         f"{int(yr_counts.index.max())}")
            lines.append("Reports per decade:")
            for d, n in (yr_counts.groupby(yr_counts.index // 10 * 10).sum()
                                  .items()):
                lines.append(f"  {int(d)}s: {int(n):,}")

        cov = diag.get("coverage_report")
        if isinstance(cov, pd.DataFrame) and not cov.empty:
            lines.append(f"Months in range: {len(cov)}")
            lines.append(f"  missing:       {int(cov['missing'].sum())}")
            lines.append(f"  thin (likely incomplete): {int(cov['thin'].sum())}")

        if "person_1function" in df.columns:
            atc_set = ["Local", "Ground", "Flight Data / Clearance Delivery",
                       "Handoff / Assist", "Approach", "Departure", "Enroute"]
            fn = df["person_1function"].astype(str)
            atc_mask = fn.apply(
                lambda s: any(t.strip() in atc_set for t in s.split(";"))
            )
            lines.append(f"ATC reporters (any tier): {int(atc_mask.sum()):,} "
                         f"({100*atc_mask.mean():.1f}%)")

        return "\n".join(lines)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ASRS DBOnline raw → combined parquet pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--raw-dir", required=True,
                        help="Directory containing per-year ASRS_DBOnline_*.xls files")
    parser.add_argument("--output",  required=True,
                        help="Output parquet path (e.g. data/processed/aviation/asrs_combined.parquet)")
    parser.add_argument("--format",  default="parquet",
                        choices=["parquet", "csv"])
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    cfg = ASRSRawConfig(output_format=args.format)
    pipe = ASRSRawPipeline(raw_dir=args.raw_dir, output=args.output, config=cfg)
    pipe.run(save=True)
    if args.summary:
        print("\n" + pipe.summary())


if __name__ == "__main__":
    main()
