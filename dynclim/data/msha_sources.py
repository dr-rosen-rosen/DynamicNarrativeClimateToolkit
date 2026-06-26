"""
MSHA Open Government data — fetch / unzip / parse helper
========================================================

Shared download layer for the three MSHA pipelines (events, ops, covariates).

All Open-Gov datasets are served as zipped, pipe-delimited (`|`) text with a
header row, double-quoted text fields, and a single inner `.txt` file named
after the zip (e.g. ``Accidents.zip`` → ``Accidents.txt``). MSHA refreshes the
files every Friday afternoon, so :func:`download` re-fetches a cached zip once
it is older than ``REFRESH_DAYS`` unless told otherwise.

IMPORTANT — column names: the flat-file headers do NOT always match the
``*_Definition_File.txt`` field dictionaries. Some files use abbreviated
headers (e.g. MinesProdYearly uses ``ANNUAL_HRS`` / ``CALENDAR_YR`` / ``C_M_IND``
rather than the documented ``ANNUAL_HOURS`` / ``CAL_YR`` / ``COAL_METAL_IND``).
Each pipeline maps against the real header, not the definition file. Use
:func:`peek_header` to inspect a file's actual columns.

Usage:
    from dynclim.data.msha_sources import read_dataset, download

    # Auto-download (if missing/stale) and parse the Mines master table
    mines = read_dataset("mines", raw_dir="data/raw/msha",
                         dtype={"MINE_ID": str})

    # Just ensure a file is present, get its extracted path
    path = download("accidents", raw_dir="data/raw/msha")
"""

from __future__ import annotations

import logging
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)

BASE_URL = "https://arlweb.msha.gov/opengovernmentdata/DataSets"
REFRESH_DAYS = 7          # Open-Gov files refresh every Friday
DEFAULT_RAW_DIR = "data/raw/msha"

# Logical name -> zip basename. Inner text file is "{basename}.txt".
DATASETS: Dict[str, str] = {
    "accidents":                    "Accidents",
    "mines":                        "Mines",
    "prod_quarterly":               "MinesProdQuarterly",
    "prod_yearly":                  "MinesProdYearly",
    "violations":                   "Violations",
    "inspections":                  "Inspections",
    # Optional / contractor + org-history extensions (not used in v1 by default)
    "contractor_prod_quarterly":    "ContractorProdQuarterly",
    "contractor_prod_yearly":       "ContractorProdYearly",
    "controller_operator_history":  "ControllerOperatorHistory",
}

# A browser-ish UA; arlweb serves scripted requests fine, but be polite.
_UA = "Mozilla/5.0 (dynclim MSHA pipeline; +https://github.com/)"


def dataset_url(name: str) -> str:
    """Direct .zip download URL for a logical dataset name."""
    base = DATASETS[name]
    return f"{BASE_URL}/{base}.zip"


def _zip_path(name: str, raw_dir: Path) -> Path:
    return raw_dir / f"{DATASETS[name]}.zip"


def _txt_path(name: str, raw_dir: Path) -> Path:
    return raw_dir / f"{DATASETS[name]}.txt"


def _is_stale(path: Path, max_age_days: int) -> bool:
    if not path.exists():
        return True
    age_days = (time.time() - path.stat().st_mtime) / 86400.0
    return age_days > max_age_days


def download(
    name: str,
    raw_dir: Union[str, Path] = DEFAULT_RAW_DIR,
    force: bool = False,
    max_age_days: int = REFRESH_DAYS,
) -> Path:
    """
    Ensure the dataset's extracted ``.txt`` is present and reasonably fresh.

    Downloads the zip (if missing, stale, or ``force``), extracts the inner
    ``.txt`` next to it, and returns the path to that ``.txt``.
    """
    if name not in DATASETS:
        raise KeyError(f"Unknown MSHA dataset '{name}'. "
                       f"Known: {sorted(DATASETS)}")

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = _zip_path(name, raw_dir)
    txt_path = _txt_path(name, raw_dir)

    need_download = force or _is_stale(zip_path, max_age_days)
    if need_download:
        url = dataset_url(name)
        logger.info(f"Downloading MSHA/{name} ← {url}")
        req = urllib.request.Request(url, headers={"User-Agent": _UA})
        tmp = zip_path.with_suffix(".zip.part")
        with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as fh:
            # Stream in chunks; some files are 100+ MB.
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
        tmp.replace(zip_path)
        logger.info(f"  saved {zip_path.stat().st_size:,} bytes")
        # Force re-extract when we just downloaded a fresh zip
        if txt_path.exists():
            txt_path.unlink()
    else:
        logger.info(f"Using cached MSHA/{name} ({zip_path.name})")

    if not txt_path.exists():
        with zipfile.ZipFile(zip_path) as zf:
            inner = f"{DATASETS[name]}.txt"
            members = zf.namelist()
            member = inner if inner in members else members[0]
            logger.info(f"  extracting {member}")
            with zf.open(member) as src, open(txt_path, "wb") as dst:
                while True:
                    chunk = src.read(1 << 20)
                    if not chunk:
                        break
                    dst.write(chunk)

    return txt_path


def peek_header(
    name: str,
    raw_dir: Union[str, Path] = DEFAULT_RAW_DIR,
    download_if_missing: bool = True,
) -> List[str]:
    """Return the real flat-file column names (the header row, split on '|')."""
    raw_dir = Path(raw_dir)
    txt_path = _txt_path(name, raw_dir)
    if download_if_missing and not txt_path.exists():
        txt_path = download(name, raw_dir)
    with open(txt_path, "r", encoding="latin-1") as fh:
        header = fh.readline().rstrip("\n").rstrip("\r")
    return [c.strip().strip('"') for c in header.split("|")]


def read_dataset(
    name: str,
    raw_dir: Union[str, Path] = DEFAULT_RAW_DIR,
    usecols: Optional[List[str]] = None,
    dtype: Optional[Dict[str, type]] = None,
    nrows: Optional[int] = None,
    download_if_missing: bool = True,
    force_download: bool = False,
    max_age_days: int = REFRESH_DAYS,
) -> pd.DataFrame:
    """
    Parse an MSHA Open-Gov flat file into a DataFrame.

    Pipe-delimited, double-quoted text fields, header row. Latin-1 decoding
    handles the non-ASCII bytes MSHA emits in some name/address fields.

    ID-style columns (MINE_ID, CONTROLLER_ID, etc.) should be read as ``str``
    via ``dtype`` to preserve leading zeros — pass e.g. ``dtype={"MINE_ID": str}``.
    Empty strings are treated as NaN.
    """
    raw_dir = Path(raw_dir)
    if force_download:
        txt_path = download(name, raw_dir, force=True, max_age_days=max_age_days)
    else:
        txt_path = _txt_path(name, raw_dir)
        if not txt_path.exists():
            if download_if_missing:
                txt_path = download(name, raw_dir, max_age_days=max_age_days)
            else:
                raise FileNotFoundError(
                    f"{txt_path} not found and download_if_missing=False"
                )

    df = pd.read_csv(
        txt_path,
        sep="|",
        header=0,
        usecols=usecols,
        dtype=dtype,
        nrows=nrows,
        encoding="latin-1",
        low_memory=False,
        na_values=[""],
        keep_default_na=True,
    )
    logger.info(f"Read MSHA/{name}: {len(df):,} rows × {len(df.columns)} cols")
    return df
