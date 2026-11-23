"""Held-out sample: reliability under the frozen guide, model agreement, validity.

Protocol sections 5.4 and 5.5. Run by file path (the package __init__ imports torch):

    .llmVenv/bin/python dynclim/experiments/analyze_heldout.py \
        --dir validity_study/heldout_2026-11-09 [--adequate-qwk 0.60] \
        [--pool-validity --pool-dir validity_study/llm]

Reads, in --dir: heldout_sheet_raterA.xlsx / raterB.xlsx (or .csv exports),
heldout_key.csv, model_ratings_x3.jsonl. Writes, in --dir:

    HELDOUT_RESULTS.md       every table for supplement S11
    heldout_values.json      named values for the response letter and manuscript
                             (placeholders listed in RESPONSE_TRACKER, section 4);
                             null while a quantity is pending
    heldout_agreement.csv, heldout_validity.csv, heldout_merged.csv

Blinding. The key holds the indicator scores. This script never prints an
item-level score, and it computes NO validity figure (human or model against the
indicator) until both rater sheets are complete (--min-complete, default 0.98 of
300 rows), because one of the raters is the study owner. --force-validity
overrides that and should not be used before the sheets are returned.

Design. 75 narratives per industry in four strata on the model's pool call
(CJ, 1-2, 3, 4-5). Every statistic is reported two ways: "as designed"
(unweighted) and "as in the pool" (inclusion-weighted; weights are normalised
within industry so that the pooled figure counts the four industries equally).
Intervals are percentile bootstrap, resampling within industry x stratum.

Model call. The model was run three times on the held-out narratives. Human to
model agreement is computed against each run and averaged (the expected
agreement of a single run, which is how the model is applied at scale); the
range over runs is reported. Agreement with the pool call that defined the
strata is given as a supplementary row.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("heldout")
SEED = 20261122
N_BOOT = 2000
IND = ["msha", "rail", "nrc", "aviation"]
IND_LABEL = {"msha": "Mining", "rail": "Rail", "nrc": "Nuclear", "aviation": "Aviation"}
LABEL_IND = {v: k for k, v in IND_LABEL.items()}
SEV_IND = ["msha", "rail"]  # structured severity available (protocol 5.2 item 2)


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------

def read_sheet(d: Path, rater: str) -> pd.DataFrame | None:
    """A rater sheet as rating_id, call ('CJ', '1'..'5', or NA), notes."""
    cands = sorted(d.glob(f"*rater{rater}*.xlsx")) + sorted(d.glob(f"*rater{rater}*.csv"))
    cands = [p for p in cands if not p.name.startswith("~")]
    if not cands:
        return None
    # prefer a returned file ("Completed...", or the most recently modified)
    p = max(cands, key=lambda q: q.stat().st_mtime)
    df = pd.read_excel(p, dtype=str) if p.suffix == ".xlsx" else pd.read_csv(p, dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]
    r = df["rating_1to5"].fillna("").str.strip().str.replace(r"\.0$", "", regex=True)
    cj = df["cant_judge"].fillna("").str.strip().str.upper().isin(["Y", "YES", "X", "CJ", "1", "TRUE"])
    # "CJ" typed into the rating column (cant_judge left blank) is a cannot-judge call
    misplaced = r.str.upper().eq("CJ") & ~cj
    if misplaced.any():
        log.warning("rater %s: %d rows have CJ typed in the rating column; read as cannot judge: %s",
                    rater, misplaced.sum(), ", ".join(df.loc[misplaced, "rating_id"]))
        cj = cj | misplaced
        r = r.mask(misplaced, "")
    both = cj & r.ne("")
    if both.any():
        log.warning("rater %s: %d rows carry both a rating and CJ; treated as missing: %s",
                    rater, both.sum(), ", ".join(df.loc[both, "rating_id"]))
    bad = r.ne("") & ~r.isin(list("12345"))
    if bad.any():
        log.warning("rater %s: %d rows with a rating outside 1-5; treated as missing", rater, bad.sum())
    call = pd.Series(pd.NA, index=df.index, dtype="object")
    call[r.isin(list("12345")) & ~cj] = r[r.isin(list("12345")) & ~cj]
    call[cj & r.eq("")] = "CJ"
    out = pd.DataFrame({"rating_id": df["rating_id"].str.strip(), f"call_{rater}": call,
                        f"notes_{rater}": df.get("notes", pd.Series("", index=df.index)).fillna("")})
    log.info("rater %s: %s, %d/%d rows coded", rater, p.name, out[f"call_{rater}"].notna().sum(), len(out))
    return out


def load(d: Path) -> tuple[pd.DataFrame, dict]:
    key = pd.read_csv(d / "heldout_key.csv", dtype={"model_call": str})
    key["w"] = key["incl_weight"] / key.groupby("industry")["incl_weight"].transform("sum")
    m = key.copy()
    status = {}
    for r in "AB":
        s = read_sheet(d, r)
        if s is None:
            m[f"call_{r}"] = pd.NA
            status[r] = 0.0
        else:
            m = m.merge(s, on="rating_id", how="left")
            status[r] = float(m[f"call_{r}"].notna().mean())
    rows = [json.loads(l) for l in (d / "model_ratings_x3.jsonl").read_text().splitlines() if l.strip()]
    mr = pd.DataFrame(rows)
    if "error" in mr:
        mr = mr[mr["error"].isna()]
    mr["call"] = np.where(mr["cant_judge"].astype(bool), "CJ", mr["rating"].astype("Int64").astype(str))
    wide = mr.pivot(index="item_id", columns="repeat", values="call")
    wide.columns = [f"call_M{int(c)}" for c in wide.columns]
    m = m.merge(wide, left_on="rating_id", right_index=True, how="left")
    m["call_Mpool"] = m["model_call"].astype(str)
    items = d / "heldout_items_for_model.csv"
    if items.exists():
        t = pd.read_csv(items, dtype=str)
        idc = "item_id" if "item_id" in t else "rating_id"
        m = m.merge(t[[idc, "text"]].rename(columns={idc: "rating_id"}), on="rating_id", how="left")
        m["n_words"] = m["text"].fillna("").str.split().str.len()
        m = m.drop(columns="text")
    return m, status


# ----------------------------------------------------------------------------
# Statistics (all accept weights)
# ----------------------------------------------------------------------------

def num(call: pd.Series) -> pd.Series:
    return pd.to_numeric(call.where(call != "CJ"), errors="coerce")


def kappa2(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    """Cohen kappa for two binary vectors with weights."""
    if len(a) == 0:
        return np.nan
    w = w / w.sum()
    po = w[a == b].sum()
    pa, pb = w[a].sum(), w[b].sum()
    pe = pa * pb + (1 - pa) * (1 - pb)
    return float((po - pe) / (1 - pe)) if pe < 1 else np.nan


def qwk(x: np.ndarray, y: np.ndarray, w: np.ndarray, k: int = 5) -> float:
    if len(x) < 2:
        return np.nan
    O = np.zeros((k, k))
    np.add.at(O, (x.astype(int) - 1, y.astype(int) - 1), w)
    W = (np.subtract.outer(np.arange(k), np.arange(k)) ** 2) / (k - 1) ** 2
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    den = (W * E).sum()
    return float(1 - (W * O).sum() / den) if den > 0 else np.nan


def wcorr(x, y, w) -> float:
    x, y, w = map(np.asarray, (x, y, w))
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    w = w / w.sum()
    mx, my = (w * x).sum(), (w * y).sum()
    c = (w * (x - mx) * (y - my)).sum()
    return float(c / np.sqrt((w * (x - mx) ** 2).sum() * (w * (y - my) ** 2).sum()))


def rank(v) -> np.ndarray:
    return pd.Series(np.asarray(v, dtype=float)).rank().to_numpy()


def agree(d: pd.DataFrame, ca: str, cb: str, weighted: bool) -> dict:
    d = d[d[ca].notna() & d[cb].notna()]
    if d.empty:
        return {"n": 0}
    w = d["w"].to_numpy() if weighted else np.ones(len(d))
    a_cj, b_cj = (d[ca] == "CJ").to_numpy(), (d[cb] == "CJ").to_numpy()
    out = {"n": len(d),
           "cj_rate_a": float(np.average(a_cj, weights=w)), "cj_rate_b": float(np.average(b_cj, weights=w)),
           "cj_agree": float(np.average(a_cj == b_cj, weights=w)), "cj_kappa": kappa2(a_cj, b_cj, w)}
    r = ~a_cj & ~b_cj
    out["n_both_rated"] = int(r.sum())
    if r.sum() >= 2:
        x, y, wr = num(d[ca]).to_numpy()[r], num(d[cb]).to_numpy()[r], w[r]
        out.update(exact=float(np.average(x == y, weights=wr)),
                   within1=float(np.average(np.abs(x - y) <= 1, weights=wr)),
                   qwk=qwk(x, y, wr), spearman=wcorr(rank(x), rank(y), wr),
                   mean_diff=float(np.average(x - y, weights=wr)))
    return out


def boot(d: pd.DataFrame, fn, rng, reps: int = N_BOOT) -> tuple[float, float]:
    """Percentile interval, resampling within industry x stratum."""
    groups = [g.index.to_numpy() for _, g in d.groupby(["industry", "stratum"])]
    vals = []
    for _ in range(reps):
        idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        v = fn(d.loc[idx])
        if v is not None and np.isfinite(v):
            vals.append(v)
    if len(vals) < reps * 0.5:
        return (np.nan, np.nan)
    return tuple(np.percentile(vals, [2.5, 97.5]))


def agreement_table(m: pd.DataFrame, pairs: dict, rng) -> pd.DataFrame:
    rows = []
    for label, (ca, cb) in pairs.items():
        for scope in ["Pooled"] + IND:
            d = m if scope == "Pooled" else m[m["industry"] == scope]
            for weighted in (False, True):
                a = agree(d, ca, cb, weighted)
                if a.get("n", 0) == 0:
                    continue
                row = {"pair": label, "scope": IND_LABEL.get(scope, scope),
                       "weighting": "as in pool" if weighted else "as designed", **a}
                if scope == "Pooled":
                    row["qwk_lo"], row["qwk_hi"] = boot(d, lambda s: agree(s, ca, cb, weighted).get("qwk"), rng)
                    row["cj_kappa_lo"], row["cj_kappa_hi"] = boot(
                        d, lambda s: agree(s, ca, cb, weighted).get("cj_kappa"), rng)
                rows.append(row)
    return pd.DataFrame(rows)


def run_to_run(m: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in m.columns if c.startswith("call_M") and c != "call_Mpool"]
    rows = []
    for scope in ["Pooled"] + IND:
        d = m if scope == "Pooled" else m[m["industry"] == scope]
        d = d.dropna(subset=cols)
        c = d[cols]
        ident = c.nunique(axis=1).eq(1)
        cj = c.eq("CJ")
        cj_ident = cj.all(axis=1) | (~cj).all(axis=1)
        n = c.apply(num)
        within1 = cj_ident & ((n.max(axis=1) - n.min(axis=1)).fillna(0) <= 1)
        rows.append({"scope": IND_LABEL.get(scope, scope), "n": len(d), "identical": ident.mean(),
                     "within1": within1.mean(), "cj_identical": cj_ident.mean(),
                     "all_equal_pool_call": (ident & c[cols[0]].eq(d["call_Mpool"])).mean()})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Validity (gated)
# ----------------------------------------------------------------------------

def partial_rank(y, x, Z, w) -> float:
    """Weighted correlation of rank residuals after regressing on Z."""
    Z = np.column_stack([np.ones(len(y))] + [np.asarray(z, float) for z in Z])
    sw = np.sqrt(w / w.sum())
    res = []
    for v in (rank(y), rank(x)):
        beta, *_ = np.linalg.lstsq(Z * sw[:, None], v * sw, rcond=None)
        res.append(v - Z @ beta)
    return wcorr(res[0], res[1], w)


def sev_code(s: pd.Series) -> np.ndarray:
    return pd.Categorical(s.fillna("unknown")).codes.astype(float)


def validity_rows(d: pd.DataFrame, target: str, tlabel: str, score: str, rng, ci: bool = True) -> list[dict]:
    rows = []
    d = d[d[target].notna() & d[score].notna()].copy()
    d["_t"] = d.groupby("industry")[target].rank()
    d["_s"] = d.groupby("industry")[score].rank()
    for scope in ["Pooled (within-industry ranks)"] + IND:
        g = d if scope.startswith("Pooled") else d[d["industry"] == scope]
        if len(g) < 5:
            continue
        for weighted in (False, True):
            w = g["w"].to_numpy() if weighted else np.ones(len(g))
            row = {"target": tlabel, "score": score, "scope": IND_LABEL.get(scope, scope),
                   "weighting": "as in pool" if weighted else "as designed", "n": len(g),
                   "n_eff": float(w.sum() ** 2 / (w ** 2).sum()),  # Kish effective n; = n when unweighted
                   "rho": wcorr(g["_t"], g["_s"], w)}
            if ci:
                wt = weighted
                row["lo"], row["hi"] = boot(
                    g, lambda s: wcorr(s["_t"], s["_s"], s["w"].to_numpy() if wt else np.ones(len(s))), rng)
            if scope in IND and "n_words" in g:
                Z = [np.log1p(g["n_words"])]
                row["rho_length"] = partial_rank(g[target], g[score], Z, w)
                if scope in SEV_IND:
                    Z = Z + [pd.get_dummies(g["severity"].fillna("unknown"), drop_first=True).to_numpy(float)[:, i]
                             for i in range(max(g["severity"].fillna("unknown").nunique() - 1, 0))]
                    row["rho_severity_length"] = partial_rank(g[target], g[score], Z, w)
            rows.append(row)
    return rows


def pool_validity(pool_dir: Path, rng) -> pd.DataFrame:
    """Protocol 5.5 item 1: model rating against the indicator on the pools."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from llm_rating import load_ratings, load_sampler  # noqa: E402
    scores = load_sampler().scores
    r = load_ratings(pool_dir / "pool_ratings.jsonl")
    r = r[r["repeat"] == 0]
    rows = []
    for ind in IND:
        p = pd.read_csv(pool_dir / f"pool_{ind}.csv", dtype=str).merge(
            r[["item_id", "cant_judge", "rating"]], on="item_id")
        p = p.merge(scores(ind), on="eid", how="left")
        p["n_words"] = p["text"].str.split().str.len()
        rated = p[~p["cant_judge"].astype(bool) & p["rating"].notna() & p["score_consensus"].notna()].copy()
        rated["rating"] = rated["rating"].astype(float)
        w = np.ones(len(rated))
        for sc in [c for c in rated.columns if c.startswith("score_")]:
            g = rated[rated[sc].notna()]
            ww = np.ones(len(g))
            row = {"industry": IND_LABEL[ind], "score": sc, "n_pool": len(p), "n_rated": len(g),
                   "rho": wcorr(rank(g["rating"]), rank(g[sc]), ww),
                   "rho_length": partial_rank(g["rating"], g[sc], [np.log1p(g["n_words"])], ww)}
            if ind in SEV_IND:
                dm = pd.get_dummies(g["severity"].fillna("unknown"), drop_first=True).to_numpy(float)
                row["rho_severity_length"] = partial_rank(
                    g["rating"], g[sc], [np.log1p(g["n_words"])] + [dm[:, i] for i in range(dm.shape[1])], ww)
            if sc == "score_consensus":
                vals = []
                for _ in range(N_BOOT):
                    b = g.iloc[rng.integers(0, len(g), len(g))]
                    vals.append(wcorr(rank(b["rating"]), rank(b[sc]), np.ones(len(b))))
                row["lo"], row["hi"] = np.nanpercentile(vals, [2.5, 97.5])
                # is the CJ decision itself related to the score? (rated vs CJ, point-biserial on ranks)
                pc = p[p["score_consensus"].notna()]
                row["rho_rateable_vs_score"] = wcorr((~pc["cant_judge"].astype(bool)).astype(float),
                                                     rank(pc["score_consensus"]), np.ones(len(pc)))
            rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------

def f(x, nd=2, pct=False, sign=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    if pct:
        return f"{100 * x:.0f}%"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def md_table(df: pd.DataFrame, cols: dict) -> str:
    head = "| " + " | ".join(cols.values()) + " |\n|" + "---|" * len(cols) + "\n"
    body = ""
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r.get(c)
            if isinstance(v, str):
                cells.append(v)
            elif c in ("n", "n_both_rated", "n_pool", "n_rated"):
                cells.append("—" if pd.isna(v) else str(int(v)))
            elif c in ("cj_rate_a", "cj_rate_b", "cj_agree", "exact", "within1", "identical",
                       "cj_identical", "all_equal_pool_call"):
                cells.append(f(v, pct=True))
            elif c in ("mean_diff", "rho", "rho_length", "rho_severity_length", "lo", "hi"):
                cells.append(f(v, sign=True))
            else:
                cells.append(f(v))
        body += "| " + " | ".join(cells) + " |\n"
    return head + body


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--min-complete", type=float, default=0.98)
    ap.add_argument("--adequate-qwk", type=float, default=None,
                    help="reading-rule threshold on pooled human-human weighted kappa (protocol open decision 10)")
    ap.add_argument("--adequate-cj-kappa", type=float, default=None)
    ap.add_argument("--pool-validity", action="store_true", help="also run protocol 5.5 item 1 on the pools")
    ap.add_argument("--pool-dir", default="validity_study/llm")
    ap.add_argument("--force-validity", action="store_true")
    a = ap.parse_args(argv)
    d = Path(a.dir)
    rng = np.random.default_rng(SEED)
    m, status = load(d)
    humans_done = all(status[r] >= a.min_complete for r in "AB")
    V: dict = {"generated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
               "sheet_completion": status, "humans_complete": humans_done}
    out = [f"# Held-out sample: results ({V['generated']})\n",
           f"Script `dynclim/experiments/analyze_heldout.py`, seed {SEED}, {N_BOOT} bootstrap resamples "
           "within industry x stratum. \"As designed\" is unweighted; \"as in pool\" applies the inclusion "
           "weights, normalised within industry.\n",
           f"Sheet completion: rater A {status['A']:.0%}, rater B {status['B']:.0%}. "
           + ("Both sheets complete.\n" if humans_done else
              "**Human sections are pending; no validity figure is computed until both sheets are complete.**\n")]

    # ---- model run to run
    rr = run_to_run(m)
    out += ["\n## 1. Model run-to-run agreement (three independent runs)\n",
            md_table(rr, {"scope": "", "n": "n", "identical": "Identical call", "within1": "Within one point",
                          "cj_identical": "CJ decision identical", "all_equal_pool_call": "All three = pool call"})]
    p = rr.iloc[0]
    V.update(MM_IDENT=round(p["identical"], 3), MM_WITHIN1=round(p["within1"], 3), MM_CJ_IDENT=round(p["cj_identical"], 3))

    # ---- strata and model calls (design description; no scores)
    out += ["\n## 2. Design: strata by industry\n", "```\n" + pd.crosstab(m["industry"].map(IND_LABEL), m["stratum"]).to_string() + "\n```\n"]

    names = ["HH_QWK_POOLED", "HH_QWK_CI", "HH_CJ_KAPPA", "HH_CJ_KAPPA_CI", "HH_N_BOTH", "HH_EXACT", "HH_WITHIN1",
             "HM_QWK_POOLED", "HM_QWK_RANGE", "HM_CJ_KAPPA", "HM_QWK_A", "HM_QWK_B", "HM_CJ_KAPPA_A", "HM_CJ_KAPPA_B",
             "RULE_OK", "VAL_H_RHO_POOLED", "VAL_H_RHO_CI"]
    names += [f"HH_QWK_{i.upper()}" for i in IND] + [f"VAL_H_RHO_{i.upper()}" for i in IND]
    V.update({k: None for k in names})

    if humans_done:
        runs = [c for c in m.columns if c.startswith("call_M") and c != "call_Mpool"]
        pairs = {"Rater A vs rater B": ("call_A", "call_B")}
        for r in "AB":
            for c in runs:
                pairs[f"Model run {c[-1]} vs rater {r}"] = (c, f"call_{r}")
            pairs[f"Model pool call vs rater {r}"] = ("call_Mpool", f"call_{r}")
        ag = agreement_table(m, pairs, rng)
        ag.to_csv(d / "heldout_agreement.csv", index=False)
        cols = {"scope": "", "weighting": "Weighting", "n": "n", "cj_rate_a": "CJ rate (first)", "cj_rate_b": "CJ rate (second)",
                "cj_agree": "CJ agreement", "cj_kappa": "CJ κ", "n_both_rated": "Both rated", "exact": "Exact",
                "within1": "Within 1", "qwk": "Weighted κ", "spearman": "Spearman", "mean_diff": "Mean difference"}
        hh = ag[ag["pair"] == "Rater A vs rater B"]
        out += ["\n## 3. Human-to-human agreement under the frozen guide\n", md_table(hh, cols)]
        hp = hh[(hh["scope"] == "Pooled") & (hh["weighting"] == "as designed")].iloc[0]
        V.update(HH_QWK_POOLED=round(hp["qwk"], 2), HH_QWK_CI=[round(hp["qwk_lo"], 2), round(hp["qwk_hi"], 2)],
                 HH_CJ_KAPPA=round(hp["cj_kappa"], 2),
                 HH_CJ_KAPPA_CI=[round(hp["cj_kappa_lo"], 2), round(hp["cj_kappa_hi"], 2)],
                 HH_N_BOTH=int(hp["n_both_rated"]), HH_EXACT=round(hp["exact"], 2), HH_WITHIN1=round(hp["within1"], 2))
        for i in IND:
            r_ = hh[(hh["scope"] == IND_LABEL[i]) & (hh["weighting"] == "as designed")]
            V[f"HH_QWK_{i.upper()}"] = None if r_.empty or pd.isna(r_.iloc[0].get("qwk")) else round(r_.iloc[0]["qwk"], 2)
        if a.adequate_qwk is not None:
            ok = hp["qwk"] >= a.adequate_qwk and (a.adequate_cj_kappa is None or hp["cj_kappa"] >= a.adequate_cj_kappa)
            V["RULE_OK"] = bool(ok)
            out.append(f"\nReading rule (pooled weighted κ ≥ {a.adequate_qwk}"
                       + (f", CJ κ ≥ {a.adequate_cj_kappa}" if a.adequate_cj_kappa is not None else "")
                       + f"): **{'met' if ok else 'not met'}**.\n")
        else:
            out.append("\nReading rule: threshold not supplied (`--adequate-qwk`); protocol open decision 10.\n")

        # human-model: average over the three runs
        hm = ag[ag["pair"].str.startswith("Model run")].copy()
        hm["rater"] = hm["pair"].str[-1]
        avg = (hm.groupby(["rater", "scope", "weighting"], sort=False)
                 .agg(n=("n", "first"), cj_agree=("cj_agree", "mean"), cj_kappa=("cj_kappa", "mean"),
                      n_both_rated=("n_both_rated", "mean"), exact=("exact", "mean"), within1=("within1", "mean"),
                      qwk=("qwk", "mean"), qwk_min=("qwk", "min"), qwk_max=("qwk", "max"),
                      spearman=("spearman", "mean"), mean_diff=("mean_diff", "mean")).reset_index())
        avg["scope"] = "Rater " + avg["rater"] + ", " + avg["scope"]
        out += ["\n## 4. Human-to-model agreement (mean over the three model runs; mean difference is model minus rater)\n",
                md_table(avg, {k: v for k, v in cols.items() if k not in ("cj_rate_a", "cj_rate_b")} | {"qwk_min": "κ min", "qwk_max": "κ max"})]
        pa = avg[avg["scope"].str.endswith("Pooled") & (avg["weighting"] == "as designed")].set_index("rater")
        V.update(HM_QWK_A=round(pa.loc["A", "qwk"], 2), HM_QWK_B=round(pa.loc["B", "qwk"], 2),
                 HM_CJ_KAPPA_A=round(pa.loc["A", "cj_kappa"], 2), HM_CJ_KAPPA_B=round(pa.loc["B", "cj_kappa"], 2),
                 HM_QWK_POOLED=round(pa["qwk"].mean(), 2), HM_CJ_KAPPA=round(pa["cj_kappa"].mean(), 2),
                 HM_QWK_RANGE=[round(pa["qwk_min"].min(), 2), round(pa["qwk_max"].max(), 2)])
        sup = ag[ag["pair"].str.startswith("Model pool call") & (ag["scope"] == "Pooled")].copy()
        sup["scope"] = sup["pair"]
        out += ["\nSupplementary: the pool call that defined the strata.\n",
                md_table(sup, {k: v for k, v in cols.items() if k not in ("cj_rate_a", "cj_rate_b")})]

        # confusion, raters
        both = m[(m["call_A"] != "CJ") & (m["call_B"] != "CJ")]
        out += ["\n## 5. Confusion (rows rater A, columns rater B; CJ included)\n",
                "```\n" + pd.crosstab(m["call_A"], m["call_B"]).to_string() + "\n```\n"]
        m.drop(columns=[c for c in m.columns if c.startswith("score_")]).to_csv(d / "heldout_merged.csv", index=False)

    do_val = humans_done or a.force_validity
    if do_val:
        m["mean_AB"] = (num(m["call_A"]) + num(m["call_B"])) / 2
        runs = [c for c in m.columns if c.startswith("call_M") and c != "call_Mpool"]
        m["mean_M"] = pd.concat([num(m[c]) for c in runs], axis=1).mean(axis=1, skipna=False)
        rows = []
        rows += validity_rows(m, "mean_AB", "Mean of two raters", "score_consensus", rng)
        m["rA"], m["rB"] = num(m["call_A"]), num(m["call_B"])
        rows += validity_rows(m, "rA", "Rater A", "score_consensus", rng, ci=False)
        rows += validity_rows(m, "rB", "Rater B", "score_consensus", rng, ci=False)
        rows += validity_rows(m, "mean_M", "Model (mean of three runs)", "score_consensus", rng)
        for sc in [c for c in m.columns if c.startswith("score_cfg")]:
            sub = m[m[sc].notna()]
            rows += [r for r in validity_rows(sub, "mean_AB", "Mean of two raters", sc, rng, ci=True)
                     if not r["scope"].startswith("Pooled")]
        val = pd.DataFrame(rows)
        val.to_csv(d / "heldout_validity.csv", index=False)
        vc = {"target": "Rating", "scope": "", "weighting": "Weighting", "n": "n", "n_eff": "Effective n", "rho": "ρ", "lo": "2.5%", "hi": "97.5%",
              "rho_length": "ρ, length controlled", "rho_severity_length": "ρ, severity and length controlled"}
        out += ["\n## 6. Validity: rating against the consensus indicator score (Spearman, within industry)\n",
                "Positive means narratives rated more favourable carry higher indicator scores. Exploratory; small samples. "
                "Effective n is the Kish effective sample size under the inclusion weights; where it is far below n, "
                "the weighted estimate and its interval rest on a few heavily weighted narratives and should not be read.\n\n",
                md_table(val[val["score"] == "score_consensus"], vc),
                "\n### Best-performing configurations (mean of two raters; secondary, uncorrected)\n",
                md_table(val[val["score"] != "score_consensus"], {"score": "Configuration"} | {k: v for k, v in vc.items() if k != "target"})]
        h = val[(val["target"] == "Mean of two raters") & (val["score"] == "score_consensus") & (val["weighting"] == "as in pool")]
        hp = h[h["scope"].str.startswith("Pooled")]
        if not hp.empty:
            V.update(VAL_H_RHO_POOLED=round(hp.iloc[0]["rho"], 2), VAL_H_RHO_CI=[round(hp.iloc[0]["lo"], 2), round(hp.iloc[0]["hi"], 2)])
        for i in IND:
            r_ = h[h["scope"] == IND_LABEL[i]]
            V[f"VAL_H_RHO_{i.upper()}"] = None if r_.empty else round(r_.iloc[0]["rho"], 2)
            V[f"VAL_H_NEFF_{i.upper()}"] = None if r_.empty else round(r_.iloc[0]["n_eff"], 1)
        hu = val[(val["target"] == "Mean of two raters") & (val["score"] == "score_consensus") & (val["weighting"] == "as designed")]
        for _, r_ in hu.iterrows():
            tag = "POOLED" if r_["scope"].startswith("Pooled") else LABEL_IND[r_["scope"]].upper()
            V[f"VAL_H_RHO_UNW_{tag}"] = round(r_["rho"], 2)
            V[f"VAL_H_RHO_UNW_CI_{tag}"] = [round(r_["lo"], 2), round(r_["hi"], 2)]
            V[f"VAL_H_N_{tag}"] = int(r_["n"])
    else:
        out.append("\n## 6. Validity\n\nPending: computed when both sheets are complete.\n")

    for i in IND:
        V[f"VAL_M_RHO_{i.upper()}"] = None
        V[f"VAL_M_RHO_{i.upper()}_SEV"] = None
    if a.pool_validity:
        if not do_val:
            log.warning("--pool-validity ignored until both sheets are complete (blinding); use --force-validity to override")
            out.append("\n## 7. Validity at scale (pools)\n\nPending: run with `--pool-validity` once both sheets are complete.\n")
        else:
            pv = pool_validity(Path(a.pool_dir), rng)
            pv.to_csv(d / "pool_validity.csv", index=False)
            out += ["\n## 7. Validity at scale: model rating against the indicator on the pools (protocol 5.5 item 1)\n",
                    md_table(pv, {"industry": "", "score": "Score", "n_pool": "Pool", "n_rated": "Rated by model", "rho": "ρ",
                                  "lo": "2.5%", "hi": "97.5%", "rho_length": "ρ, length controlled",
                                  "rho_severity_length": "ρ, severity and length controlled"})]
            for i in IND:
                r_ = pv[(pv["industry"] == IND_LABEL[i]) & (pv["score"] == "score_consensus")]
                if not r_.empty:
                    V[f"VAL_M_RHO_{i.upper()}"] = round(r_.iloc[0]["rho"], 2)
                    sv = r_.iloc[0].get("rho_severity_length")
                    V[f"VAL_M_RHO_{i.upper()}_SEV"] = None if pd.isna(sv) else round(sv, 2)

    (d / "HELDOUT_RESULTS.md").write_text("\n".join(out))
    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        return o
    (d / "heldout_values.json").write_text(json.dumps(clean(V), indent=2))
    pending = [k for k, v in V.items() if v is None]
    log.info("wrote HELDOUT_RESULTS.md and heldout_values.json; %d values pending", len(pending))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
