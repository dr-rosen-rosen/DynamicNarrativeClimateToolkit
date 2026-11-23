#!/usr/bin/env python3
"""
analyze_validity_stage2.py

Stage-2 analysis of the content-validity rating study (protocol section 5):

  1. Inter-rater reliability: CJ agreement (Cohen kappa) and, on narratives
     both raters rated, quadratic weighted kappa / ICC / Spearman / exact and
     within-one agreement. Pooled and by industry.
  2. Validity: Spearman correlation between the mean of the two ratings and
     the indicator (consensus score = primary; each main-text champion
     configuration = secondary). WITHIN industry, with bootstrap CIs; pooled
     estimate uses within-industry ranks.
  3. Severity-conditioned validity for mining and rail: partial rank
     correlation controlling the structured severity stratum (dummies), and
     severity + log narrative length.
  4. CJ prevalence per industry; CJ treated as missing.

Inputs (one stage-2 directory built by sample_validity_ratings.py --stage2):
  rating_key_consolidated.csv         key (strata, scores, screen outcomes)
  <rater A sheet>, <rater B sheet>    csv or xlsx with rating_id,
                                      rating_1to5, cant_judge, notes

Outputs (written into the same directory):
  stage2_ratings_merged.csv           key + both raters' calls
  stage2_cj_agreement.csv
  stage2_reliability.csv
  stage2_confusion.csv
  stage2_validity.csv
  stage2_for_consensus.csv            CJ-discordant, >=2-point gaps, blanks
                                      (text + calls + notes; NO indicator
                                      scores, so the consensus pass stays blind)

Usage:
  python -m dynclim.experiments.analyze_validity_stage2 <stage2_dir> \
      --rater-a "Completed.rating_sheet_raterA_consolidated(Sheet1).csv" \
      --rater-b rating_sheet_raterB_consolidated.xlsx
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score, roc_auc_score

SEED = 20261105
N_BOOT = 5000
INDUSTRIES = ["aviation", "msha", "nrc", "rail"]
LABEL = {"aviation": "Aviation", "msha": "Mining", "nrc": "Nuclear", "rail": "Rail"}

# main-text champion configurations (analysis repo, build_champion_manuscript_table)
CHAMPION_CELLS = {
    ("rail", 127): "accidents", ("rail", 30): "fatalities", ("rail", 39): "injuries",
    ("nrc", 119): "emergency declarations", ("nrc", 124): "LERs",
    ("nrc", 120): "pct power loss", ("nrc", 29): "scrams",
    ("aviation", 59): "AIDS no-harm", ("aviation", 26): "AIDS prop-damage",
    ("aviation", 62): "NTSB serious/fatal",
    ("msha", 89): "days lost", ("msha", 40): "t0 no-injury", ("msha", 131): "t1 severe",
    ("msha", 46): "t2 lost-time / t3 minor",
    # champions of cells that are no longer in the main text (carried in the key)
    ("nrc", 102): "inspection findings (not main text)",
    ("aviation", 8): "NTSB fatal (not main text)",
    ("aviation", 10): "NTSB serious (not main text)",
    ("msha", 47): "fatal (not main text)",
}


# ----------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------
def read_sheet(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in (".xlsx", ".xls"):
        d = pd.read_excel(path)
    else:
        d = pd.read_csv(path, encoding_errors="replace")
    need = {"rating_id", "rating_1to5", "cant_judge"}
    missing = need - set(d.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
    if "notes" not in d.columns:
        d["notes"] = np.nan
    r = pd.to_numeric(d["rating_1to5"], errors="coerce")
    bad = r.notna() & ~r.isin([1, 2, 3, 4, 5])
    if bad.any():
        raise ValueError(f"{path.name}: ratings outside 1-5 at {d.loc[bad, 'rating_id'].tolist()}")
    cj = d["cant_judge"].astype(str).str.strip().str.lower().isin(["cj", "y", "yes", "x", "true", "1"])
    both = cj & r.notna()
    if both.any():
        print(f"  NOTE {path.name}: {both.sum()} rows have BOTH a rating and CJ -> treated as CJ: "
              f"{d.loc[both, 'rating_id'].tolist()}")
        r = r.where(~cj)
    out = pd.DataFrame({"rating_id": d["rating_id"].astype(str), "r": r, "cj": cj,
                        "note": d["notes"]})
    out["state"] = np.select([out["cj"], out["r"].notna()], ["cj", "rated"], default="blank")
    return out


def load(d: Path, fa: str, fb: str) -> pd.DataFrame:
    key = pd.read_csv(d / "rating_key_consolidated.csv")
    key["rating_id"] = key["rating_id"].astype(str)
    key = key.rename(columns={"cj_A": "screen_cj_A", "cj_B": "screen_cj_B"})
    a, b = read_sheet(d / fa), read_sheet(d / fb)
    for nm, s in (("A", a), ("B", b)):
        if set(s["rating_id"]) != set(key["rating_id"]) or s["rating_id"].duplicated().any():
            raise ValueError(f"rater {nm}: rating ids do not match the key")
    m = key.merge(a.add_suffix("_A").rename(columns={"rating_id_A": "rating_id"}), on="rating_id") \
           .merge(b.add_suffix("_B").rename(columns={"rating_id_B": "rating_id"}), on="rating_id")
    assert len(m) == len(key)
    m["both_rated"] = (m["state_A"] == "rated") & (m["state_B"] == "rated")
    m["mean_AB"] = np.where(m["both_rated"], (m["r_A"] + m["r_B"]) / 2, np.nan)
    m["mean_any"] = m[["r_A", "r_B"]].mean(axis=1, skipna=True)
    return m


# ----------------------------------------------------------------------------
# reliability
# ----------------------------------------------------------------------------
def icc2(x: np.ndarray, y: np.ndarray) -> dict:
    """Two-way ICCs for two raters (McGraw & Wong): A,1 / A,k / C,1."""
    X = np.column_stack([x, y]).astype(float)
    n, k = X.shape
    if n < 3:
        return {"icc_a1": np.nan, "icc_ak": np.nan, "icc_c1": np.nan}
    gm = X.mean()
    msr = k * ((X.mean(1) - gm) ** 2).sum() / (n - 1)
    msc = n * ((X.mean(0) - gm) ** 2).sum() / (k - 1)
    sse = ((X - X.mean(1, keepdims=True) - X.mean(0, keepdims=True) + gm) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        a1 = (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)
        ak = (msr - mse) / (msr + (msc - mse) / n)
        c1 = (msr - mse) / (msr + (k - 1) * mse)
    return {"icc_a1": a1, "icc_ak": ak, "icc_c1": c1}


def qwk(x, y) -> float:
    if len(x) < 3 or (len(set(x)) == 1 and len(set(y)) == 1):
        return np.nan
    return cohen_kappa_score(x.astype(int), y.astype(int), weights="quadratic", labels=[1, 2, 3, 4, 5])


def boot_ci(fn, n: int, rng, reps: int = N_BOOT):
    vals = []
    for _ in range(reps):
        idx = rng.integers(0, n, n)
        try:
            v = fn(idx)
        except Exception:
            v = np.nan
        vals.append(v)
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    if len(vals) < reps * 0.5:
        return np.nan, np.nan
    return tuple(np.percentile(vals, [2.5, 97.5]))


def cj_table(m: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def one(lab, sub, d):
        d = d[(d["state_A"] != "blank") & (d["state_B"] != "blank")]
        a, b = d["cj_A"].to_numpy(), d["cj_B"].to_numpy()
        n = len(d)
        kap = cohen_kappa_score(a, b) if n and (a.any() or b.any()) and not (a.all() and b.all()) else np.nan
        rows.append({"industry": lab, "subset": sub, "n": n,
                     "cj_rate_A": a.mean() if n else np.nan, "cj_rate_B": b.mean() if n else np.nan,
                     "both_cj": int((a & b).sum()), "A_cj_only": int((a & ~b).sum()),
                     "B_cj_only": int((~a & b).sum()), "both_rated": int((~a & ~b).sum()),
                     "agreement": (a == b).mean() if n else np.nan, "kappa": kap})

    for lab, d in [("pooled", m)] + [(i, m[m["industry"] == i]) for i in INDUSTRIES]:
        one(lab, "all", d)
        one(lab, "screen concordant", d[~d["screen_discordant"]])
        one(lab, "screen discordant", d[d["screen_discordant"]])
    return pd.DataFrame(rows)


def reliability_table(m: pd.DataFrame, rng) -> pd.DataFrame:
    rows = []
    for lab, d in [("pooled", m)] + [(i, m[m["industry"] == i]) for i in INDUSTRIES]:
        d = d[d["both_rated"]]
        x, y = d["r_A"].to_numpy(float), d["r_B"].to_numpy(float)
        n = len(d)
        row = {"industry": lab, "n_both_rated": n}
        if n >= 5:
            row.update({
                "exact": (x == y).mean(), "within_1": (np.abs(x - y) <= 1).mean(),
                "gap_ge2": int((np.abs(x - y) >= 2).sum()),
                "mean_A": x.mean(), "mean_B": y.mean(), "mean_diff_A_minus_B": (x - y).mean(),
                "sd_A": x.std(ddof=1), "sd_B": y.std(ddof=1),
                "qwk": qwk(x, y), "spearman": stats.spearmanr(x, y)[0], **icc2(x, y),
            })
            lo, hi = boot_ci(lambda i: qwk(x[i], y[i]), n, rng, 2000)
            row["qwk_lo"], row["qwk_hi"] = lo, hi
            lo, hi = boot_ci(lambda i: icc2(x[i], y[i])["icc_a1"], n, rng, 2000)
            row["icc_a1_lo"], row["icc_a1_hi"] = lo, hi
            # valence only: unfavourable (1-2) / mixed (3) / favourable (4-5)
            v = lambda z: np.select([z <= 2, z >= 4], [0, 2], 1)
            vx, vy = v(x), v(y)
            row["valence3_agree"] = (vx == vy).mean()
            row["valence3_kappa"] = cohen_kappa_score(vx, vy) if len(set(vx) | set(vy)) > 1 else np.nan
            row["opposite_valence"] = int(((vx == 0) & (vy == 2)).sum() + ((vx == 2) & (vy == 0)).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def confusion(m: pd.DataFrame) -> pd.DataFrame:
    lab = lambda r, s: np.where(s == "cj", "CJ", np.where(s == "blank", "blank",
                                                          r.fillna(0).astype(int).astype(str)))
    d = m.assign(A=lab(m["r_A"], m["state_A"]), B=lab(m["r_B"], m["state_B"]))
    order = ["1", "2", "3", "4", "5", "CJ", "blank"]
    out = []
    for ind, g in [("pooled", d)] + [(i, d[d["industry"] == i]) for i in INDUSTRIES]:
        t = pd.crosstab(g["A"], g["B"]).reindex(index=order, columns=order, fill_value=0)
        t.insert(0, "industry", ind)
        t.insert(1, "rater_A", t.index)
        out.append(t.reset_index(drop=True))
    return pd.concat(out, ignore_index=True)


# ----------------------------------------------------------------------------
# validity
# ----------------------------------------------------------------------------
def partial_rank(x, y, Z) -> tuple[float, float, int]:
    """Partial Spearman: rank x and y, residualise both on Z (with intercept)."""
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    Zc = np.column_stack([np.ones(len(x)), Z])
    ex = rx - Zc @ np.linalg.lstsq(Zc, rx, rcond=None)[0]
    ey = ry - Zc @ np.linalg.lstsq(Zc, ry, rcond=None)[0]
    if ex.std() == 0 or ey.std() == 0:
        return np.nan, np.nan, len(x)
    r = np.corrcoef(ex, ey)[0, 1]
    df = len(x) - 2 - np.linalg.matrix_rank(Zc) + 1
    if df <= 0 or abs(r) >= 1:
        return r, np.nan, len(x)
    t = r * np.sqrt(df / (1 - r ** 2))
    return r, 2 * stats.t.sf(abs(t), df), len(x)


def weighted_rank_corr(x, y, w) -> float:
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    w = w / w.sum()
    mx, my = (w * rx).sum(), (w * ry).sum()
    cov = (w * (rx - mx) * (ry - my)).sum()
    return cov / np.sqrt((w * (rx - mx) ** 2).sum() * (w * (ry - my) ** 2).sum())


def within_rank(v: pd.Series, g: pd.Series) -> pd.Series:
    """Rank within group, rescaled to (0, 1)."""
    return v.groupby(g).transform(lambda s: (s.rank() - 0.5) / s.notna().sum())


def validity_table(m: pd.DataFrame, rng) -> pd.DataFrame:
    rows = []
    targets = {"mean_AB": "mean of both raters (both rated) [PRIMARY]",
               "r_A": "rater A alone", "r_B": "rater B alone",
               "mean_any": "mean of available ratings (rated by >=1)"}
    score_cols = [c for c in m.columns if c.startswith("score_")]

    def one(ind, d, sc, tgt, score_label, role):
        d = d[d[tgt].notna() & d[sc].notna()]
        n = len(d)
        row = {"industry": ind, "score": sc, "score_label": score_label, "role": role,
               "target": tgt, "n": n}
        if n < 8:
            rows.append(row)
            return
        x, y = d[sc].to_numpy(float), d[tgt].to_numpy(float)
        rho, p = stats.spearmanr(x, y)
        lo, hi = boot_ci(lambda i: stats.spearmanr(x[i], y[i])[0], n, rng)
        row.update({"spearman": rho, "ci_lo": lo, "ci_hi": hi, "p": p})
        fav, unf = y > 3, y < 3
        if fav.sum() >= 3 and unf.sum() >= 3:
            keep = fav | unf
            row.update({"auc_fav_vs_unfav": roc_auc_score(fav[keep], x[keep]),
                        "n_fav": int(fav.sum()), "n_unfav": int(unf.sum())})
        if ind in ("msha", "rail"):
            S = pd.get_dummies(d["severity"], drop_first=True).to_numpy(float)
            row["partial_sev"], row["partial_sev_p"], _ = partial_rank(x, y, S)
            L = np.column_stack([S, np.log(d["n_words"].to_numpy(float))])
            row["partial_sev_len"], row["partial_sev_len_p"], _ = partial_rank(x, y, L)
            row["weighted_spearman"] = weighted_rank_corr(x, y, d["incl_weight"].to_numpy(float))
        else:
            L = np.log(d["n_words"].to_numpy(float))[:, None]
            row["partial_len"], row["partial_len_p"], _ = partial_rank(x, y, L)
        rows.append(row)

    for ind in INDUSTRIES:
        d = m[m["industry"] == ind]
        for sc in score_cols:
            if d[sc].notna().sum() == 0:
                continue
            if sc == "score_consensus":
                lab, role = "consensus (81 configs)", "primary"
            else:
                cid = int(sc.replace("score_cfg", ""))
                lab, role = f"cfg {cid}: {CHAMPION_CELLS.get((ind, cid), '?')}", "champion"
            for tgt in targets:
                one(ind, d, sc, tgt, lab, role)

    # pooled, consensus only: raw, and with both variables ranked within industry
    for tgt in targets:
        d = m[m[tgt].notna()].copy()
        x, y = d["score_consensus"].to_numpy(float), d[tgt].to_numpy(float)
        rho, p = stats.spearmanr(x, y)
        lo, hi = boot_ci(lambda i: stats.spearmanr(x[i], y[i])[0], len(d), rng)
        rows.append({"industry": "pooled (raw)", "score": "score_consensus",
                     "score_label": "consensus (81 configs)", "role": "primary", "target": tgt,
                     "n": len(d), "spearman": rho, "ci_lo": lo, "ci_hi": hi, "p": p})
        xr = within_rank(d["score_consensus"], d["industry"]).to_numpy()
        yr = within_rank(d[tgt], d["industry"]).to_numpy()
        r, p = stats.pearsonr(xr, yr)
        lo, hi = boot_ci(lambda i: np.corrcoef(xr[i], yr[i])[0, 1], len(d), rng)
        rows.append({"industry": "pooled (within-industry ranks)", "score": "score_consensus",
                     "score_label": "consensus (81 configs)", "role": "primary", "target": tgt,
                     "n": len(d), "spearman": r, "ci_lo": lo, "ci_hi": hi, "p": p})
    out = pd.DataFrame(rows)
    out["target_label"] = out["target"].map(targets)
    return out


def tercile_table(m: pd.DataFrame) -> pd.DataFrame:
    d = m[m["both_rated"]]
    t = d.groupby(["industry", "tercile"])["mean_AB"].agg(["count", "mean"]).unstack("tercile")
    t = t.reindex(columns=["low", "mid", "high"], level=1)
    return t


# ----------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage2_dir")
    ap.add_argument("--rater-a", default="Completed.rating_sheet_raterA_consolidated(Sheet1).csv")
    ap.add_argument("--rater-b", default="rating_sheet_raterB_consolidated.xlsx")
    args = ap.parse_args()
    d = Path(args.stage2_dir)
    rng = np.random.default_rng(SEED)
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.float_format", "{:.3f}".format)

    m = load(d, args.rater_a, args.rater_b)
    sheet = pd.read_csv(d / "rating_sheet_raterA_consolidated.csv", usecols=["rating_id", "text"])
    sheet["rating_id"] = sheet["rating_id"].astype(str)
    m.to_csv(d / "stage2_ratings_merged.csv", index=False)

    print(f"\n== {len(m)} narratives; states ==")
    print(pd.crosstab(m["state_A"], m["state_B"], margins=True))
    for r in "AB":
        bl = m.loc[m[f"state_{r}"] == "blank", "rating_id"].tolist()
        if bl:
            print(f"  rater {r} left blank: {bl}")

    cj = cj_table(m)
    cj.to_csv(d / "stage2_cj_agreement.csv", index=False)
    print("\n== CJ agreement ==")
    print(cj.to_string(index=False))

    rel = reliability_table(m, rng)
    rel.to_csv(d / "stage2_reliability.csv", index=False)
    print("\n== Rating reliability (both rated) ==")
    print(rel.T.to_string())

    conf = confusion(m)
    conf.to_csv(d / "stage2_confusion.csv", index=False)
    print("\n== Confusion (pooled; rows A, cols B) ==")
    print(conf[conf["industry"] == "pooled"].drop(columns="industry").to_string(index=False))

    val = validity_table(m, rng)
    val.to_csv(d / "stage2_validity.csv", index=False)
    print("\n== Validity: consensus score ==")
    cols = ["industry", "target", "n", "spearman", "ci_lo", "ci_hi", "p", "auc_fav_vs_unfav",
            "partial_sev", "partial_sev_p", "partial_sev_len", "partial_len", "weighted_spearman"]
    print(val[val["role"] == "primary"][cols].to_string(index=False))
    print("\n== Validity: champion configs, mean of both raters ==")
    print(val[(val["role"] == "champion") & (val["target"] == "mean_AB")]
          [["industry", "score_label", "n", "spearman", "ci_lo", "ci_hi", "p", "auc_fav_vs_unfav",
            "partial_sev", "partial_sev_len", "partial_len"]].to_string(index=False))
    print("\n== Mean rating (both rated) by consensus tercile ==")
    print(tercile_table(m).to_string())

    # consensus worklist: no indicator scores
    gap = (m["r_A"] - m["r_B"]).abs()
    why = np.select(
        [(m["state_A"] == "blank") | (m["state_B"] == "blank"),
         m["cj_A"] != m["cj_B"], gap >= 2],
        ["blank", "CJ discordant", "rating gap >= 2"], default="")
    w = m.loc[why != "", ["rating_id", "industry", "r_A", "cj_A", "note_A", "r_B", "cj_B", "note_B"]].copy()
    w.insert(2, "reason", why[why != ""])
    w = w.merge(sheet, on="rating_id", how="left")
    w["consensus_rating_1to5"] = ""
    w["consensus_cant_judge"] = ""
    w["consensus_note"] = ""
    w = w.sort_values(["industry", "reason", "rating_id"])
    w.to_csv(d / "stage2_for_consensus.csv", index=False)
    print("\n== Consensus worklist ==")
    print(pd.crosstab(w["industry"], w["reason"], margins=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
