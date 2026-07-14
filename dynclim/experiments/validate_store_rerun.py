"""
Old-vs-new store validation for the paper-1 revision rerun
==========================================================

Compares a historical ExperimentRunner store against a vectorized-driver
rerun of the same corpus. Configs are matched by registry SIGNATURE (id
assignment is shared, but signature matching stays correct even if the
grids differ). Per matched config, results are joined on report_id and
every numeric scoring column is compared (item *_final/_sent/_strength/
_n_segments, domain *_domain_score, overall_*).

Acceptance criteria (handoff_paper1_revision_rerun.md §4, revised
2026-07-14 after the msha validation):
  - weighted_average / attention_weight arms must reproduce EXCEPT for a
    bounded fraction of rows (--max-diff-frac, default 0) affected by the
    old per-report path's case-collision cache contamination: segment
    cache keys were case-folded (unified_processor._normalize_segment),
    so a report whose sentence had an earlier-seen case variant elsewhere
    in the corpus inherited that variant's cached sentiment/embedding.
    VADER and cased transformer sentiment models are case-sensitive
    (ALL-CAPS boosts VADER; siebert flips ±1), so affected rows differ.
    msha: 0.96%% of reports. Verified by direct recomputation: the NEW
    store matches vader(raw text); the old store does not.
  - power_attention arms are EXPECTED to differ (the old stores' power
    arms additionally carry the torch-2.0.1 MPS nanquantile corruption);
    their shift is quantified, not asserted — these numbers feed the
    revised Methods.

Outputs (written into the NEW store dir):
  validation_vs_<old>__configs.csv   per-config comparison table
  validation_vs_<old>.md             headline summary

Run (by file path):
  .dynClimVenv/bin/python dynclim/experiments/validate_store_rerun.py \\
      --old dynclim/notebooks/checkpoints/msha_06-01-2026 \\
      --new dynclim/notebooks/checkpoints/msha_2026-07-13
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SKIP_COLS = {"config_id", "processing_time"}


def load_registry(store: Path) -> pd.DataFrame:
    reg = pd.read_csv(store / "config_registry.csv")
    # Zero-variation columns are dropped to the sidecar at build time;
    # backfill the ones this script reports on.
    const_path = store / "registry_constants.json"
    constants = {}
    if const_path.exists():
        import json
        constants = json.loads(const_path.read_text())
    for col in ("comp__method", "embedding_model", "sent_method"):
        if col not in reg.columns:
            reg[col] = constants.get(col, "")
    missing = {"config_id", "signature"} - set(reg.columns)
    if missing:
        raise SystemExit(f"{store}: registry missing columns {missing}")
    return reg


def case_collision_eligible_ids(new_store: Path) -> set:
    """Report ids whose segments have a case/whitespace-variant twin
    somewhere in the corpus — the population the legacy path's normalized
    cache keys could contaminate. Row i of a vectorized store's results
    corresponds to corpus row i, which links report_id to segments."""
    cdir = new_store / "_corpus"
    if not cdir.exists():
        raise SystemExit(f"{new_store}: no _corpus/ — "
                         f"--explain-case-collisions needs a vectorized store")
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from dynclim.experiments.run_vectorized_experiment import (
        load_corpus_segments)
    lens, idx, uniques = load_corpus_segments(cdir)
    norm = {}
    for i, s in enumerate(uniques):
        norm.setdefault(" ".join(s.lower().strip().split()), []).append(i)
    prone = np.zeros(len(uniques), dtype=bool)
    for ids in norm.values():
        if len(ids) > 1:
            prone[ids] = True
    offsets = np.concatenate([[0], np.cumsum(lens)])
    eligible = np.array([prone[idx[offsets[r]:offsets[r + 1]]].any()
                         for r in range(len(lens))])

    first_cfg = sorted((new_store / "_cfg").iterdir(),
                       key=lambda p: int(p.name))[0]
    rids = pd.read_parquet(first_cfg / "results.parquet",
                           columns=["report_id"])["report_id"].astype(str)
    if len(rids) != len(lens):
        raise SystemExit(f"{new_store}: results rows ({len(rids)}) != corpus "
                         f"rows ({len(lens)}); cannot map eligibility")
    ids = set(rids[eligible])
    print(f"case-collision eligibility: {prone.sum()} of {len(uniques)} "
          f"unique segments have a variant twin; {len(ids)} of {len(lens)} "
          f"reports eligible ({100 * len(ids) / len(lens):.2f}%)")
    return ids


def compare_config(old_df: pd.DataFrame, new_df: pd.DataFrame, tol: float,
                   eligible_ids: set = None):
    m = old_df.merge(new_df, on="report_id", how="outer",
                     suffixes=("_o", "_n"), indicator=True)
    joined = int((m["_merge"] == "both").sum())
    old_only = int((m["_merge"] == "left_only").sum())
    new_only = int((m["_merge"] == "right_only").sum())
    both = m[m["_merge"] == "both"]

    num_cols = [c for c in old_df.columns
                if c not in SKIP_COLS and c != "report_id"
                and pd.api.types.is_numeric_dtype(old_df[c])
                and f"{c}_n" in both.columns]
    max_abs, max_col = 0.0, ""
    deltas_overall = None
    row_diff = np.zeros(len(both), dtype=bool)
    for c in num_cols:
        o = both[f"{c}_o"].to_numpy(dtype=np.float64)
        n = both[f"{c}_n"].to_numpy(dtype=np.float64)
        d = np.abs(o - n)
        d[np.isnan(o) & np.isnan(n)] = 0.0     # NaN==NaN is agreement
        row_diff |= np.nan_to_num(d) > tol
        dmax = float(np.nanmax(d)) if len(d) else 0.0
        if dmax > max_abs:
            max_abs, max_col = dmax, c
        if c == "overall_final_score":
            deltas_overall = (o, n, d)

    o, n, d = deltas_overall
    ok = ~(np.isnan(o) | np.isnan(n))
    corr = float(np.corrcoef(o[ok], n[ok])[0, 1]) if ok.sum() > 2 else np.nan
    diff_explained = np.nan
    if eligible_ids is not None and row_diff.any():
        diff_ids = both["report_id"].to_numpy()[row_diff]
        diff_explained = float(np.mean([r in eligible_ids for r in diff_ids]))
    return {
        "n_joined": joined, "n_old_only": old_only, "n_new_only": new_only,
        "max_abs_delta": max_abs, "max_delta_col": max_col,
        "frac_rows_gt_tol": float(row_diff.mean()) if len(both) else 0.0,
        "frac_diff_explained": diff_explained,
        "overall_max_abs_delta": float(np.nanmax(d)) if len(d) else 0.0,
        "overall_mean_abs_delta": float(np.nanmean(d)) if len(d) else 0.0,
        "overall_corr": corr,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--max-diff-frac", type=float, default=0.0,
                    help="strict arms PASS if the fraction of rows with any "
                         "column delta > tol is at most this (use for old "
                         "stores with known case-collision cache "
                         "contamination; see unified_processor."
                         "normalize_segments)")
    ap.add_argument("--note", default="",
                    help="free-text context appended to the summary md")
    ap.add_argument("--explain-case-collisions", action="store_true",
                    help="compute the case-collision-eligible report set "
                         "from the new store's _corpus; strict arms PASS if "
                         ">=99.5%% of their differing rows fall in it")
    args = ap.parse_args()

    old_store, new_store = Path(args.old), Path(args.new)
    old_reg, new_reg = load_registry(old_store), load_registry(new_store)
    pairs = old_reg[["config_id", "signature", "comp__method",
                     "embedding_model", "sent_method"]].merge(
        new_reg[["config_id", "signature"]], on="signature",
        suffixes=("_old", "_new"))
    unmatched_old = len(old_reg) - len(pairs)
    unmatched_new = len(new_reg) - len(pairs)
    print(f"{len(pairs)} configs matched by signature "
          f"({unmatched_old} old / {unmatched_new} new unmatched)")

    eligible_ids = (case_collision_eligible_ids(new_store)
                    if args.explain_case_collisions else None)

    rows = []
    for i, r in enumerate(pairs.itertuples(), 1):
        old_p = old_store / "_cfg" / str(r.config_id_old) / "results.parquet"
        new_p = new_store / "_cfg" / str(r.config_id_new) / "results.parquet"
        if not old_p.exists() or not new_p.exists():
            rows.append({"config_id": r.config_id_old,
                         "comp_method": r.comp__method,
                         "status": "MISSING_RESULTS"})
            continue
        old_df, new_df = pd.read_parquet(old_p), pd.read_parquet(new_p)
        old_df["report_id"] = old_df["report_id"].astype(str)
        new_df["report_id"] = new_df["report_id"].astype(str)
        stats = compare_config(old_df, new_df, args.tol, eligible_ids)
        strict = r.comp__method in ("weighted_average", "attention_weight")
        explained = (eligible_ids is not None
                     and stats["frac_diff_explained"] >= 0.995)
        status = ("PASS" if stats["frac_rows_gt_tol"] <= args.max_diff_frac
                  or explained else "FAIL") if strict else "POWER"
        rows.append({"config_id": r.config_id_old,
                     "embedding_model": r.embedding_model,
                     "sent_method": r.sent_method,
                     "comp_method": r.comp__method,
                     "arm": "strict" if strict else "power",
                     "status": status, **stats})
        expl = ("" if np.isnan(stats.get("frac_diff_explained", np.nan))
                else f" explained: {100*stats['frac_diff_explained']:.2f}%")
        print(f"[{i}/{len(pairs)}] cfg {r.config_id_old} "
              f"{r.comp__method:>18} {status:>6} "
              f"max|Δ|={stats['max_abs_delta']:.2e} ({stats['max_delta_col']}) "
              f"rows>{args.tol:g}: {100*stats['frac_rows_gt_tol']:.2f}%{expl}")

    out = pd.DataFrame(rows)
    tag = old_store.name
    out.to_csv(new_store / f"validation_vs_{tag}__configs.csv", index=False)

    strict = out[out["arm"] == "strict"]
    power = out[out["arm"] == "power"]
    missing = out[out["status"] == "MISSING_RESULTS"]
    n_fail = int((strict["status"] == "FAIL").sum())

    lines = [
        f"# Store validation: {new_store.name} vs {tag}",
        "",
        f"- configs matched by signature: {len(pairs)} "
        f"(old-only {unmatched_old}, new-only {unmatched_new}, "
        f"missing results {len(missing)})",
        f"- report coverage (first config): "
        f"{out.iloc[0].get('n_joined', 'n/a')} joined, "
        f"{out.iloc[0].get('n_old_only', 'n/a')} old-only, "
        f"{out.iloc[0].get('n_new_only', 'n/a')} new-only",
        "",
        f"## weighted_average / attention_weight arms "
        f"({len(strict)} configs) — must reproduce",
        f"- max |Δ| across all scoring columns: "
        f"{strict['max_abs_delta'].max():.3e} (tol {args.tol:g}) — "
        f"{'ALL PASS' if n_fail == 0 else f'{n_fail} FAIL'} "
        f"(allowed diff-row fraction {args.max_diff_frac:g})",
        f"- rows with any |Δ| > tol: median "
        f"{100*strict['frac_rows_gt_tol'].median():.3f}%, max "
        f"{100*strict['frac_rows_gt_tol'].max():.3f}%"
        + (f"; differing rows case-collision-explained: min "
           f"{100*strict['frac_diff_explained'].min():.2f}%"
           if args.explain_case_collisions else ""),
        f"- overall_final_score mean |Δ|: median "
        f"{strict['overall_mean_abs_delta'].median():.3e}, max "
        f"{strict['overall_mean_abs_delta'].max():.3e}",
        "",
        f"## power_attention arms ({len(power)} configs) — expected shift "
        f"(MPS nanquantile corruption in the old store)",
        f"- overall_final_score max |Δ|: median "
        f"{power['overall_max_abs_delta'].median():.3e}, "
        f"max {power['overall_max_abs_delta'].max():.3e}",
        f"- overall_final_score mean |Δ|: median "
        f"{power['overall_mean_abs_delta'].median():.3e}, "
        f"max {power['overall_mean_abs_delta'].max():.3e}",
        f"- old-vs-new overall_final_score correlation: median "
        f"{power['overall_corr'].median():.4f}, "
        f"min {power['overall_corr'].min():.4f}",
        "",
    ]
    if args.note:
        lines += ["## Notes", args.note, ""]
    md = "\n".join(lines)
    (new_store / f"validation_vs_{tag}.md").write_text(md, encoding="utf-8")
    print("\n" + md)

    if n_fail or len(missing):
        print(f"VALIDATION FAILED: {n_fail} strict-arm failures, "
              f"{len(missing)} configs missing results")
        sys.exit(1)
    print("VALIDATION OK")


if __name__ == "__main__":
    main()
