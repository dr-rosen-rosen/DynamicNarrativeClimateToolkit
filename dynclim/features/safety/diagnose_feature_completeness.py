"""
diagnose_feature_completeness.py
---------------------------------
Assess quality / completeness of generated features across multiverse
analysis configurations. Reads a _cfg directory containing per-config
results.parquet files and a config_registry.csv, then returns a
diagnostic DataFrame linking completeness metrics to configuration factors.

Usage
-----
    # As a script:
    python diagnose_feature_completeness.py /path/to/_cfg

    # As an importable function:
    from diagnose_feature_completeness import diagnose_feature_completeness
    diag_df = diagnose_feature_completeness("/path/to/_cfg")
    summary = summarize_failures(diag_df, factor="th__method")
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core diagnostic: per-config completeness metrics
# ---------------------------------------------------------------------------

def _assess_parquet(
    parquet_path: Path,
    domain_pattern: str = "_domain_score",
    sparsity_threshold: float = 0.90,
) -> dict:
    """
    Read a single results.parquet and compute completeness metrics
    for all domain-score columns.

    Returns a dict with:
      - n_reports: total rows
      - n_domain_cols: number of *_domain_score columns found
      - domain_cols: list of column names
      - Per domain column:
          {col}_n_valid, {col}_n_na, {col}_pct_na,
          {col}_variance, {col}_zero_var (bool),
          {col}_pct_zero, {col}_sparse (bool)
      - Aggregate flags:
          any_na, any_zero_var, any_sparse,
          all_na, all_zero_var,
          total_na_pct (across all domain cols)
    """
    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        return {
            "file_exists": True,
            "read_error": str(e),
            "n_reports": 0,
            "n_domain_cols": 0,
        }

    domain_cols = [c for c in df.columns if domain_pattern in c]

    out = {
        "file_exists": True,
        "read_error": None,
        "n_reports": len(df),
        "n_domain_cols": len(domain_cols),
        "domain_cols_found": "|".join(domain_cols),
    }

    if len(domain_cols) == 0 or len(df) == 0:
        out.update({
            "any_na": len(domain_cols) == 0,
            "any_zero_var": False,
            "any_sparse": False,
            "all_na": len(domain_cols) == 0,
            "all_zero_var": False,
            "total_na_pct": 1.0 if len(domain_cols) == 0 else 0.0,
            "total_zero_var_cols": 0,
            "total_sparse_cols": 0,
        })
        return out

    # Per-column metrics
    na_flags = []
    zero_var_flags = []
    sparse_flags = []
    na_counts = []

    for col in domain_cols:
        vals = df[col]
        n_na = int(vals.isna().sum())
        n_valid = int(vals.notna().sum())
        pct_na = n_na / len(df) if len(df) > 0 else 0.0

        # Variance on non-NA values
        valid_vals = vals.dropna()
        if len(valid_vals) < 2:
            var = 0.0
            zero_var = True
        else:
            var = float(valid_vals.var())
            zero_var = var == 0.0 or np.isnan(var)

        # Sparsity: fraction of zeros (among non-NA values)
        if len(valid_vals) > 0:
            pct_zero = float((valid_vals == 0).sum()) / len(valid_vals)
        else:
            pct_zero = 0.0
        sparse = pct_zero >= sparsity_threshold

        out[f"{col}__n_valid"] = n_valid
        out[f"{col}__n_na"] = n_na
        out[f"{col}__pct_na"] = round(pct_na, 4)
        out[f"{col}__variance"] = round(var, 6)
        out[f"{col}__zero_var"] = zero_var
        out[f"{col}__pct_zero"] = round(pct_zero, 4)
        out[f"{col}__sparse"] = sparse

        na_flags.append(n_na > 0)
        zero_var_flags.append(zero_var)
        sparse_flags.append(sparse)
        na_counts.append(n_na)

    # Aggregate flags
    total_cells = len(df) * len(domain_cols)
    total_na = sum(na_counts)

    out.update({
        "any_na": any(na_flags),
        "any_zero_var": any(zero_var_flags),
        "any_sparse": any(sparse_flags),
        "all_na": all(na_flags),
        "all_zero_var": all(zero_var_flags),
        "total_na_pct": round(total_na / total_cells, 4) if total_cells > 0 else 0.0,
        "total_zero_var_cols": sum(zero_var_flags),
        "total_sparse_cols": sum(sparse_flags),
    })

    # --- Segment count statistics ---
    # Look for *_n_segments columns (per-item segment counts)
    seg_cols = [c for c in df.columns if c.endswith("_n_segments")]

    if seg_cols:
        # Aggregate across all segment count columns per report
        seg_matrix = df[seg_cols]
        # Mean segment count per report (averaged across items)
        report_mean_segs = seg_matrix.mean(axis=1)

        out["seg__n_cols_found"] = len(seg_cols)
        out["seg__global_mean"] = round(float(report_mean_segs.mean()), 2)
        out["seg__global_median"] = round(float(report_mean_segs.median()), 2)
        out["seg__global_min"] = round(float(report_mean_segs.min()), 2)
        out["seg__global_max"] = round(float(report_mean_segs.max()), 2)

        # Reports with critically low segment counts
        out["seg__pct_reports_le1"] = round(
            float((report_mean_segs <= 1).mean()), 4
        )
        out["seg__pct_reports_eq0"] = round(
            float((report_mean_segs == 0).mean()), 4
        )

        # Per-item segment count detail (which items starve?)
        for col in seg_cols:
            vals = df[col].dropna()
            if len(vals) > 0:
                out[f"{col}__mean"] = round(float(vals.mean()), 2)
                out[f"{col}__min"] = int(vals.min())
                out[f"{col}__pct_le1"] = round(float((vals <= 1).mean()), 4)
                out[f"{col}__pct_eq0"] = round(float((vals == 0).mean()), 4)

        # Flag: segment starvation (>10% of reports have <=1 segment on avg)
        out["seg__starvation"] = out["seg__pct_reports_le1"] > 0.10
    else:
        out["seg__n_cols_found"] = 0
        out["seg__starvation"] = None

    return out


def diagnose_feature_completeness(
    cfg_dir: str | Path,
    registry_filename: str = "config_registry.csv",
    domain_pattern: str = "_domain_score",
    sparsity_threshold: float = 0.90,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Scan all config subdirectories under `cfg_dir`, assess each
    results.parquet for feature completeness, and join to the
    config registry.

    Parameters
    ----------
    cfg_dir : path
        Root config directory containing config_registry.csv and
        one subdirectory per config_id, each with results.parquet.
    registry_filename : str
        Name of the registry CSV file.
    domain_pattern : str
        Substring to match domain-score columns (default: '_domain_score').
    sparsity_threshold : float
        Fraction of zeros above which a column is flagged as sparse.
    verbose : bool
        Print progress and summary.

    Returns
    -------
    pd.DataFrame
        One row per configuration, with registry columns + completeness metrics.
        Includes a composite 'failure_flag' column (any_na | any_zero_var | any_sparse).
    """
    cfg_dir = Path(cfg_dir)

    # Load registry — check _cfg dir first, then parent
    reg_path = cfg_dir / registry_filename
    if not reg_path.exists():
        reg_path = cfg_dir.parent / registry_filename
    if not reg_path.exists():
        raise FileNotFoundError(
            f"Registry not found in {cfg_dir} or {cfg_dir.parent}"
        )

    registry = pd.read_csv(reg_path)
    if "config_id" not in registry.columns:
        raise ValueError("config_registry.csv must have a 'config_id' column")

    # Ensure config_id is string for path construction
    registry["config_id"] = registry["config_id"].astype(str)

    if verbose:
        print(f"Loaded registry: {len(registry)} configurations")

    # Scan config subdirectories
    diagnostics = []
    missing = []
    for _, row in registry.iterrows():
        config_id = row["config_id"]
        pq_path = cfg_dir / config_id / "results.parquet"

        if not pq_path.exists():
            diag = {
                "config_id": config_id,
                "file_exists": False,
                "read_error": "results.parquet not found",
                "n_reports": 0,
                "n_domain_cols": 0,
                "any_na": True,
                "any_zero_var": False,
                "any_sparse": False,
                "all_na": True,
                "all_zero_var": False,
                "total_na_pct": 1.0,
                "total_zero_var_cols": 0,
                "total_sparse_cols": 0,
                "seg__n_cols_found": 0,
                "seg__starvation": None,
            }
            missing.append(config_id)
        else:
            diag = _assess_parquet(
                pq_path,
                domain_pattern=domain_pattern,
                sparsity_threshold=sparsity_threshold,
            )
            diag["config_id"] = config_id

        diagnostics.append(diag)

    diag_df = pd.DataFrame(diagnostics)

    # Join to registry
    result = registry.merge(diag_df, on="config_id", how="left")

    # Composite failure flag
    result["failure_flag"] = (
        result.get("any_na", False).fillna(True).astype(bool)
        | result.get("any_zero_var", False).fillna(False).astype(bool)
        | result.get("any_sparse", False).fillna(False).astype(bool)
    )

    # Segment starvation flag (separate — may not have _n_segments cols)
    if "seg__starvation" in result.columns:
        result["seg__starvation"] = result["seg__starvation"].fillna(False).astype(bool)
        result["failure_flag_with_starvation"] = (
            result["failure_flag"] | result["seg__starvation"]
        )

    if verbose:
        n_total = len(result)
        n_missing = len(missing)
        n_failed = result["failure_flag"].sum()
        n_na = result["any_na"].sum() if "any_na" in result.columns else 0
        n_zv = result["any_zero_var"].sum() if "any_zero_var" in result.columns else 0
        n_sp = result["any_sparse"].sum() if "any_sparse" in result.columns else 0

        print(f"\n{'='*60}")
        print(f"FEATURE COMPLETENESS DIAGNOSTIC SUMMARY")
        print(f"{'='*60}")
        print(f"  Total configurations:    {n_total}")
        print(f"  Missing results files:   {n_missing}")
        print(f"  Configs with any NA:     {n_na} ({n_na/n_total:.1%})")
        print(f"  Configs with zero var:   {n_zv} ({n_zv/n_total:.1%})")
        print(f"  Configs with sparsity:   {n_sp} ({n_sp/n_total:.1%})")
        print(f"  Total failure-flagged:   {n_failed} ({n_failed/n_total:.1%})")

        if "seg__starvation" in result.columns:
            n_starved = result["seg__starvation"].sum()
            has_segs = (result["seg__n_cols_found"] > 0).sum()
            print(f"  ---")
            print(f"  Configs with seg data:   {has_segs}")
            print(f"  Segment starvation:      {n_starved} ({n_starved/n_total:.1%})")
            if "seg__global_mean" in result.columns:
                starved = result[result["seg__starvation"] == True]
                healthy = result[result["seg__starvation"] == False]
                if len(starved) > 0 and len(healthy) > 0:
                    print(f"  Mean segs (starved):     {starved['seg__global_mean'].mean():.1f}")
                    print(f"  Mean segs (healthy):     {healthy['seg__global_mean'].mean():.1f}")
                    print(f"  Pct reports ≤1 seg (starved): "
                          f"{starved['seg__pct_reports_le1'].mean():.1%}")

        print(f"{'='*60}\n")

    return result


# ---------------------------------------------------------------------------
# Summary helpers: failure rates by config factor
# ---------------------------------------------------------------------------

def summarize_failures(
    diag_df: pd.DataFrame,
    factor: str | Sequence[str],
    failure_col: str = "failure_flag",
) -> pd.DataFrame:
    """
    Compute failure rate grouped by one or more config factors.

    Parameters
    ----------
    diag_df : DataFrame from diagnose_feature_completeness()
    factor : str or list of str
        Column name(s) from the config registry to group by.
    failure_col : str
        Column indicating failure (default: 'failure_flag').

    Returns
    -------
    DataFrame with columns: factor(s), n_configs, n_failures,
        failure_rate, na_rate, zero_var_rate, sparse_rate.
    """
    if isinstance(factor, str):
        factor = [factor]

    for f in factor:
        if f not in diag_df.columns:
            raise ValueError(f"Factor '{f}' not found in diagnostic DataFrame. "
                             f"Available: {sorted(diag_df.columns.tolist())}")

    agg = diag_df.groupby(factor, dropna=False).agg(
        n_configs=(failure_col, "count"),
        n_failures=(failure_col, "sum"),
        na_rate=("any_na", "mean"),
        zero_var_rate=("any_zero_var", "mean"),
        sparse_rate=("any_sparse", "mean"),
        mean_total_na_pct=("total_na_pct", "mean"),
    ).reset_index()

    agg["failure_rate"] = agg["n_failures"] / agg["n_configs"]

    # Sort by failure rate descending
    agg = agg.sort_values("failure_rate", ascending=False).reset_index(drop=True)

    return agg


def summarize_all_factors(
    diag_df: pd.DataFrame,
    exclude_cols: Optional[Sequence[str]] = None,
    failure_col: str = "failure_flag",
) -> dict[str, pd.DataFrame]:
    """
    Run summarize_failures for every config factor column (auto-detected).
    
    Returns dict mapping factor name -> summary DataFrame.
    Only includes columns that vary across configs and are not diagnostic columns.
    """
    # Identify registry columns (not diagnostic outputs)
    # Exact-match diagnostic columns
    diag_exact = {
        "file_exists", "read_error", "n_reports", "n_domain_cols",
        "domain_cols_found", "any_na", "any_zero_var", "any_sparse",
        "all_na", "all_zero_var", "total_na_pct", "total_zero_var_cols",
        "total_sparse_cols", "failure_flag",
    }
    # Per-domain detail columns contain domain_pattern + "__"
    # e.g. "LSC_domain_score__n_na", "LSC_domain_score__pct_na"
    diag_markers = ["_domain_score__"]
    exclude = set(exclude_cols or []) | {"config_id", "signature"}

    factor_cols = []
    for col in diag_df.columns:
        if col in exclude or col in diag_exact:
            continue
        if any(m in col for m in diag_markers):
            continue
        # Must have >1 unique value to be interesting
        if diag_df[col].nunique(dropna=False) > 1:
            factor_cols.append(col)

    results = {}
    for f in factor_cols:
        try:
            results[f] = summarize_failures(diag_df, factor=f, failure_col=failure_col)
        except Exception:
            pass

    return results


# ---------------------------------------------------------------------------
# Per-domain detail: which specific domains are problematic
# ---------------------------------------------------------------------------

def domain_failure_detail(
    diag_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Pivot the per-domain metrics into a tidy format showing which
    domains have the most NA / zero-variance / sparsity issues.

    Returns DataFrame with columns: domain, n_configs_with_na,
        pct_configs_na, n_configs_zero_var, n_configs_sparse, mean_pct_na.
    """
    # Find per-domain detail columns
    na_cols = [c for c in diag_df.columns if c.endswith("__n_na")]
    domains = [c.replace("__n_na", "") for c in na_cols]

    rows = []
    for domain in domains:
        na_col = f"{domain}__n_na"
        zv_col = f"{domain}__zero_var"
        sp_col = f"{domain}__sparse"
        pna_col = f"{domain}__pct_na"

        row = {"domain": domain}
        if na_col in diag_df.columns:
            has_na = (diag_df[na_col] > 0).sum()
            row["n_configs_with_na"] = has_na
            row["pct_configs_na"] = round(has_na / len(diag_df), 4)
        if pna_col in diag_df.columns:
            row["mean_pct_na"] = round(diag_df[pna_col].mean(), 4)
        if zv_col in diag_df.columns:
            row["n_configs_zero_var"] = diag_df[zv_col].sum()
        if sp_col in diag_df.columns:
            row["n_configs_sparse"] = diag_df[sp_col].sum()

        rows.append(row)

    detail = pd.DataFrame(rows)
    if "n_configs_with_na" in detail.columns:
        detail = detail.sort_values("n_configs_with_na", ascending=False)

    return detail.reset_index(drop=True)


def segment_starvation_detail(
    diag_df: pd.DataFrame,
    factor: str | Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Analyze segment count statistics, optionally grouped by config factors.

    If factor is None, returns one row per config with segment stats.
    If factor is provided, returns aggregated segment stats by factor level.

    Useful for confirming whether failures correlate with segment starvation
    from aggressive thresholding.
    """
    seg_cols = ["seg__global_mean", "seg__global_median", "seg__global_min",
                "seg__global_max", "seg__pct_reports_le1", "seg__pct_reports_eq0",
                "seg__starvation"]
    available = [c for c in seg_cols if c in diag_df.columns]

    if not available:
        print("No segment count columns found in diagnostic DataFrame. "
              "Results may not contain *_n_segments columns.")
        return pd.DataFrame()

    if factor is None:
        # Per-config detail
        id_cols = ["config_id"]
        # Also include failure flags for easy comparison
        extra = [c for c in ["failure_flag", "any_na", "any_zero_var", "any_sparse"]
                 if c in diag_df.columns]
        return diag_df[id_cols + extra + available].copy()

    if isinstance(factor, str):
        factor = [factor]

    # Aggregate by factor
    agg_dict = {}
    for col in available:
        if col == "seg__starvation":
            agg_dict[col] = ("seg__starvation", "mean")  # rate
        elif "pct" in col:
            agg_dict[col + "_mean"] = (col, "mean")
        else:
            agg_dict[col + "_mean"] = (col, "mean")
            agg_dict[col + "_min"] = (col, "min")

    # Add failure rate for cross-reference
    if "failure_flag" in diag_df.columns:
        agg_dict["failure_rate"] = ("failure_flag", "mean")

    agg_dict["n_configs"] = ("seg__global_mean", "count")

    result = diag_df.groupby(factor, dropna=False).agg(**agg_dict).reset_index()

    # Rename starvation column for clarity
    if "seg__starvation" in result.columns:
        result = result.rename(columns={"seg__starvation": "starvation_rate"})

    return result.sort_values(
        "starvation_rate" if "starvation_rate" in result.columns else factor,
        ascending=False,
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Diagnose feature completeness across multiverse configs"
    )
    parser.add_argument(
        "cfg_dir",
        help="Path to _cfg directory with config_registry.csv and per-config subdirs",
    )
    parser.add_argument(
        "--registry", default="config_registry.csv",
        help="Registry filename (default: config_registry.csv)",
    )
    parser.add_argument(
        "--pattern", default="_domain_score",
        help="Column name pattern for domain scores (default: _domain_score)",
    )
    parser.add_argument(
        "--sparsity", type=float, default=0.90,
        help="Sparsity threshold (default: 0.90)",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Save diagnostic DataFrame to CSV (optional)",
    )
    parser.add_argument(
        "--summary-output", default=None,
        help="Save per-factor summaries to CSV (optional)",
    )

    args = parser.parse_args()

    # Run diagnostics
    diag_df = diagnose_feature_completeness(
        args.cfg_dir,
        registry_filename=args.registry,
        domain_pattern=args.pattern,
        sparsity_threshold=args.sparsity,
        verbose=True,
    )

    # Print per-factor summaries
    factor_summaries = summarize_all_factors(diag_df)
    if factor_summaries:
        print("\nFAILURE RATES BY CONFIGURATION FACTOR")
        print("-" * 60)
        for factor, summary in factor_summaries.items():
            print(f"\n  {factor}:")
            for _, row in summary.iterrows():
                label = row[factor] if pd.notna(row[factor]) else "NA"
                print(f"    {label:>30s}  "
                      f"fail={row['failure_rate']:.1%}  "
                      f"(na={row['na_rate']:.1%}, "
                      f"zv={row['zero_var_rate']:.1%}, "
                      f"sp={row['sparse_rate']:.1%})  "
                      f"n={int(row['n_configs'])}")

    # Print domain detail
    domain_detail = domain_failure_detail(diag_df)
    if len(domain_detail) > 0:
        print(f"\n\nPER-DOMAIN FAILURE DETAIL")
        print("-" * 60)
        print(domain_detail.to_string(index=False))

    # Print segment starvation detail by factor
    if "seg__starvation" in diag_df.columns and diag_df["seg__starvation"].any():
        print(f"\n\nSEGMENT STARVATION BY CONFIGURATION FACTOR")
        print("-" * 60)
        for factor in factor_summaries.keys():
            seg_detail = segment_starvation_detail(diag_df, factor=factor)
            if len(seg_detail) > 0:
                print(f"\n  {factor}:")
                display_cols = [c for c in seg_detail.columns
                                if c != factor and not c.endswith("_min")]
                print(seg_detail[[factor] + display_cols].to_string(index=False))

    # Save outputs
    if args.output:
        diag_df.to_csv(args.output, index=False)
        print(f"\nDiagnostic DataFrame saved to: {args.output}")

    if args.summary_output and factor_summaries:
        # Stack all factor summaries with a 'factor' column
        frames = []
        for factor, summary in factor_summaries.items():
            s = summary.copy()
            s.insert(0, "factor_name", factor)
            s = s.rename(columns={factor: "factor_level"})
            s["factor_level"] = s["factor_level"].astype(str)
            frames.append(s)
        all_summaries = pd.concat(frames, ignore_index=True)
        all_summaries.to_csv(args.summary_output, index=False)
        print(f"Factor summaries saved to: {args.summary_output}")

    return diag_df


if __name__ == "__main__":
    main()
