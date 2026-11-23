"""
Construct-validity rating sample (PNAS Nexus revision, R1.1 / R2)
==================================================================

Draws a blinded, stratified sample of event narratives for the two-stage
human workflow agreed 2026-10-08 (protocol: docs/validity_study/
rating_protocol_draft.md and the shared protocol page):

  Stage 1 — CJ SCREEN: every drawn narrative is screened for "can't judge"
    (no information about safety practice) by one of two reviewers, with a
    20% overlap screened by both for CJ reliability. This script writes the
    two reviewer sheets + the assignment key.
  Stage 2 — RATING: both raters rate every non-CJ narrative (built later from
    the returned screening sheets; see --stage2).

Design:
  * Scope matches the paper-1 analyses: msha = coal only; nrc = power-reactor
    events (rx_type populated); rail and aviation = all reports.
  * Stratification score = CONSENSUS across the 81 weighted_average /
    attention_weight configs (mean of within-config z-scores of
    overall_final_score). The key also carries the raw score of every
    main-text champion configuration (timeseries champions from the analysis
    repo's champions_best.csv), so validity can be reported per champion.
  * Strata: industry x consensus tercile x structured severity (msha: none /
    low / mid / high from degree_ordinal; rail: damage_only / injury / fatal).
    Within each cell, mining and rail are drawn with LENGTH weights
    (--length-alloc, default 60% longest quartile / 25% third / 15% shorter
    half) because practice-relevant content concentrates in long narratives
    (keyword cues: 3.5% in Q4 vs 0.2% in Q1 for mining). Length is
    indicator-independent, so this enriches without circularity; the key
    records each narrative's length stratum and sampling weight so prevalence
    can be weighted back to the population.
  * Exact-duplicate boilerplate collapsed; strings under --min-words excluded;
    eids in --exclude-eids files (protocol examples, pilot) excluded.

Run (by file path; record the seed in the protocol BEFORE drawing):
    .dynClimVenv/bin/python dynclim/experiments/sample_validity_ratings.py \\
        --seed 20261008 --out validity_study/stage1_2026-10-08 \\
        --exclude-eids validity_study/protocol_example_eids.txt,validity_study/pilot_2026-10-06/rating_key_seed11.csv
"""
from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("validity_sample")
REPO = Path(__file__).resolve().parents[2]
P = REPO / "dynclim/data/processed"
CK = REPO / "dynclim/notebooks/checkpoints"
ANALYSIS = Path("/Users/michaelrosen/Documents/data_anlaysis/dyn_nar_clim_mv_analysis")

STORES = {"msha": "msha_2026-07-13", "rail": "rail_2026-07-14",
          "aviation": "asrs_2026-07-14", "nrc": "nrc_2026-07-15"}
IDCOL = {"msha": "eid", "rail": "eid", "aviation": "eid", "nrc": "event_num"}
TXTCOL = {"msha": "narrative", "rail": "narrative", "aviation": "narrative",
          "nrc": "event_text"}
LENGTH_STRAT = {"msha", "rail"}          # industries drawn with length weights
INDS = ["msha", "rail", "aviation", "nrc"]


def load_events(ind: str) -> pd.DataFrame:
    extra = {"msha": ["commodity", "degree_ordinal"],
             "rail": ["Total Persons Killed", "Total Persons Injured"],
             "nrc": ["rx_type"], "aviation": []}[ind]
    ev = pd.read_parquet(P / ind / "events.parquet",
                         columns=[IDCOL[ind], TXTCOL[ind]] + extra)
    ev = ev.rename(columns={IDCOL[ind]: "eid", TXTCOL[ind]: "text"})
    ev["eid"] = ev["eid"].astype(str)
    if ind == "msha":
        ev = ev[ev["commodity"].astype(str).str.lower() == "coal"]
        d = pd.to_numeric(ev["degree_ordinal"], errors="coerce").astype("float64").to_numpy()
        ev["severity"] = np.select([np.isnan(d), d <= 2, d <= 4, d >= 5],
                                   ["none", "low", "mid", "high"], default="none")
    elif ind == "rail":
        k = pd.to_numeric(ev["Total Persons Killed"], errors="coerce").fillna(0).astype("float64").to_numpy()
        i = pd.to_numeric(ev["Total Persons Injured"], errors="coerce").fillna(0).astype("float64").to_numpy()
        ev["severity"] = np.select([k > 0, i > 0], ["fatal", "injury"], default="damage_only")
    elif ind == "nrc":
        rx = ev["rx_type"].astype(str).str.strip()
        ev = ev[~rx.isin(["", "None", "nan", "NaN"])]   # reactor events only
        ev["severity"] = "na"
    else:
        ev["severity"] = "na"
    return ev[["eid", "text", "severity"]].copy()


def champion_configs(ind: str) -> list[int]:
    f = ANALYSIS / "report_figures_panel" / ind / "champions_best.csv"
    if not f.exists():
        log.warning("%s: no champions_best.csv at %s; no champion scores carried", ind, f)
        return []
    c = pd.read_csv(f)
    return sorted(c.loc[c["cv_strategy"] == "timeseries", "best_config_id"].astype(int).unique())


def scores(ind: str) -> pd.DataFrame:
    """Consensus z over wa/att configs + raw score per main-text champion config."""
    store = CK / STORES[ind]
    reg = pd.read_csv(store / "config_registry.csv")
    strict = set(reg.loc[reg["comp__method"] != "power_attention", "config_id"].astype(int))
    champs = champion_configs(ind)
    zsum = zcnt = None
    cols = {}
    for cid in sorted(strict | set(champs)):
        df = pd.read_parquet(store / "_cfg" / str(cid) / "results.parquet",
                             columns=["report_id", "overall_final_score"])
        s = df.set_index(df["report_id"].astype(str))["overall_final_score"]
        if cid in strict:
            z = (s - s.mean()) / s.std()
            zsum = z if zsum is None else zsum.add(z, fill_value=0)
            zcnt = z.notna().astype(int) if zcnt is None else zcnt.add(z.notna().astype(int), fill_value=0)
        if cid in champs:
            cols[f"score_cfg{cid}"] = s
    out = (zsum / zcnt.replace(0, np.nan)).rename("score_consensus").to_frame()
    for k, v in cols.items():
        out[k] = v
    out.index.name = "eid"
    log.info("%s: consensus over %d configs; champion scores %s", ind, len(strict), champs)
    return out.reset_index()


def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def sample_industry(ev: pd.DataFrame, n_target: int, rng: np.random.Generator,
                    min_words: int, length_alloc: dict | None) -> pd.DataFrame:
    ev = ev[ev["score_consensus"].notna()].copy()
    ev["n_words"] = ev["text"].astype(str).str.split().str.len()
    ev = ev[ev["n_words"] >= min_words]
    ev = ev.loc[~ev["text"].map(norm_text).duplicated(keep="first")]
    ev["tercile"] = pd.qcut(ev["score_consensus"], 3, labels=["low", "mid", "high"])
    ev["len_q"] = pd.qcut(ev["n_words"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    ev["len_stratum"] = ev["len_q"].map({"Q1": "short", "Q2": "short", "Q3": "Q3", "Q4": "Q4"})
    # sampling weight per narrative: proportional to allocation / stratum size
    if length_alloc:
        size = ev["len_stratum"].value_counts()
        ev["w"] = ev["len_stratum"].map(lambda s: length_alloc[s] / size[s]).astype(float)
    else:
        ev["w"] = 1.0
    ev["w"] = ev["w"] / ev["w"].sum()
    ev = ev[ev["w"] > 0]                 # a zero allocation (e.g. Q4-only top-up) drops the stratum
    cells = ev.groupby(["tercile", "severity"], observed=True)
    per_cell = max(1, n_target // cells.ngroups)
    picked = []
    for _, g in cells:
        take = min(per_cell, len(g))
        picked.append(g.sample(take, weights=g["w"], random_state=int(rng.integers(1 << 31))))
    out = pd.concat(picked)
    short = n_target - len(out)
    if short > 0:
        rest = ev.drop(out.index)
        out = pd.concat([out, rest.sample(min(short, len(rest)), weights=rest["w"],
                                          random_state=int(rng.integers(1 << 31)))])
    # inclusion probability (approx.) for weighting prevalence back to population
    out["incl_weight"] = (1.0 / (out["w"] * len(ev))).round(3)
    return out


def assign_reviewers(smp: pd.DataFrame, overlap: float, rng: np.random.Generator) -> pd.DataFrame:
    """Within each industry: `overlap` share screened by both, the rest split evenly."""
    smp = smp.copy()
    smp["reviewer"] = ""
    for ind, g in smp.groupby("industry"):
        idx = rng.permutation(g.index.to_numpy())
        n_both = int(round(len(idx) * overlap))
        both, rest = idx[:n_both], idx[n_both:]
        half = len(rest) // 2
        smp.loc[both, "reviewer"] = "both"
        smp.loc[rest[:half], "reviewer"] = "A"
        smp.loc[rest[half:], "reviewer"] = "B"
    return smp


def write_sheets(smp: pd.DataFrame, out_dir: Path, tag: str, rng: np.random.Generator):
    sheet_cols = ["rating_id", "industry", "text"]
    for rv in ["A", "B"]:
        s = smp[smp["reviewer"].isin([rv, "both"])].copy()
        s = s.sample(frac=1, random_state=int(rng.integers(1 << 31)))
        s = s.sort_values("industry", kind="stable")
        sheet = s[sheet_cols].copy()
        sheet["cant_judge_YN"] = ""
        sheet["notes"] = ""
        sheet.to_csv(out_dir / f"screen_sheet_reviewer{rv}_{tag}.csv", index=False)
        try:
            sheet.to_excel(out_dir / f"screen_sheet_reviewer{rv}_{tag}.xlsx", index=False)
        except Exception as e:
            log.warning("xlsx for reviewer %s not written (%s)", rv, e)
        log.info("reviewer %s: %d narratives (%s)", rv, len(sheet),
                 s.groupby("industry").size().to_dict())


def _screen_survivors(screen_dir: Path) -> pd.DataFrame:
    key = pd.read_csv(next(screen_dir.glob("screen_key_seed*.csv")))
    tag = next(screen_dir.glob("screen_key_seed*.csv")).stem.replace("screen_key_", "")
    text_by_id = {}
    for rv in ["A", "B"]:
        x = pd.read_excel(screen_dir / f"screen_sheet_reviewer{rv}_{tag}.xlsx")
        v = x["cant_judge_YN"].astype(str).str.strip().str.upper().map({"Y": True, "N": False})
        key = key.merge(pd.DataFrame({"rating_id": x["rating_id"], f"cj_{rv}": v}),
                        on="rating_id", how="left")
        text_by_id.update(dict(zip(x["rating_id"], x["text"])))
    key["survivor"] = (key["cj_A"] == False) | (key["cj_B"] == False)
    key["screen_discordant"] = ((key["reviewer"] == "both") & key["cj_A"].notna()
                                & key["cj_B"].notna() & (key["cj_A"] != key["cj_B"]))
    s = key[key["survivor"]].copy()
    s["text"] = s["rating_id"].map(text_by_id)
    s["screen_batch"] = tag
    return s


def stage2(screen_dirs: list[Path], out_dir: Path, seed: int, rng: np.random.Generator,
           tag: str = "all"):
    """Build ONE stage-2 rating sheet per rater from one or more returned
    stage-1 screens. Survivor = rateable (N) per at least one screener;
    overlap disagreements are included and flagged, since stage-2 raters keep
    a CJ option and the final CJ call is their consensus. Both raters rate
    every survivor; the sheets are identical. rating_ids are unique across
    batches (each draw used its own seed-tagged id space; verified below)."""
    s = pd.concat([_screen_survivors(d) for d in screen_dirs], ignore_index=True)
    dup = s["rating_id"].duplicated()
    if dup.any():                      # ids collide across draws: disambiguate
        s["rating_id"] = s["rating_id"] + "_" + s["screen_batch"].str.replace("seed", "")
    s = s.sample(frac=1, random_state=int(rng.integers(1 << 31)))
    s = s.sort_values("industry", kind="stable")
    out_dir.mkdir(parents=True, exist_ok=True)
    sheet = s[["rating_id", "industry", "text"]].copy()
    sheet["rating_1to5"] = ""
    sheet["cant_judge"] = ""
    sheet["notes"] = ""
    for rv in ["A", "B"]:
        sheet.to_csv(out_dir / f"rating_sheet_rater{rv}_{tag}.csv", index=False)
        try:
            sheet.to_excel(out_dir / f"rating_sheet_rater{rv}_{tag}.xlsx", index=False)
        except Exception as e:
            log.warning("xlsx for rater %s not written (%s)", rv, e)
    s.drop(columns=["text"]).to_csv(out_dir / f"rating_key_{tag}.csv", index=False)
    log.info("stage 2: %d survivors from %d screen(s) -> %s (discordant screens: %d; "
             "ids disambiguated: %s)", len(s), len(screen_dirs), out_dir,
             int(s["screen_discordant"].sum()), bool(dup.any()))
    print(pd.crosstab(s["industry"], s["screen_batch"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True,
                    help="record in the protocol BEFORE drawing the final sample")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage2", default=None,
                    help="comma-separated returned stage-1 screen dirs; builds ONE "
                         "consolidated stage-2 rating sheet per rater into --out and exits")
    ap.add_argument("--stage2-tag", default="all",
                    help="filename tag for the consolidated stage-2 sheets")
    ap.add_argument("--n", default="msha=800,rail=400,aviation=100,nrc=100",
                    help="per-industry draw sizes for the stage-1 CJ screen")
    ap.add_argument("--length-alloc", default="Q4=0.60,Q3=0.25,short=0.15",
                    help="length-stratum allocation for mining and rail")
    ap.add_argument("--overlap", type=float, default=0.20,
                    help="share of each industry screened by BOTH reviewers")
    ap.add_argument("--min-words", type=int, default=5)
    ap.add_argument("--exclude-eids", default=None,
                    help="comma-separated files; .txt = one eid per line, .csv = an 'eid' column")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    rng = np.random.default_rng(args.seed)
    if args.stage2:
        stage2([Path(p.strip()) for p in args.stage2.split(",")], Path(args.out),
               args.seed, rng, args.stage2_tag)
        return
    targets = {k: int(v) for k, v in (kv.split("=") for kv in args.n.split(","))}
    alloc = {k: float(v) for k, v in (kv.split("=") for kv in args.length_alloc.split(","))}
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    exclude = set()
    for f in (args.exclude_eids.split(",") if args.exclude_eids else []):
        f = Path(f.strip())
        if f.suffix == ".csv":
            exclude |= set(pd.read_csv(f)["eid"].astype(str))
        else:
            exclude |= {l.strip() for l in f.read_text().splitlines() if l.strip()}
    log.info("excluding %d eids", len(exclude))

    frames = []
    for ind in INDS:
        if targets.get(ind, 0) <= 0:
            continue
        ev = load_events(ind).merge(scores(ind), on="eid", how="inner")
        ev = ev[~ev["eid"].isin(exclude)]
        s = sample_industry(ev, targets[ind], rng, args.min_words,
                            alloc if ind in LENGTH_STRAT else None)
        s["industry"] = ind
        frames.append(s)
        log.info("%s: drew %d; length strata %s", ind, len(s),
                 s["len_stratum"].value_counts().to_dict())
    smp = pd.concat(frames, ignore_index=True)
    smp["rating_id"] = [f"R{i:04d}" for i in rng.permutation(len(smp)) + 1]
    smp = assign_reviewers(smp, args.overlap, rng)

    tag = f"seed{args.seed}"
    key_cols = (["rating_id", "industry", "eid", "reviewer", "tercile", "severity",
                 "len_stratum", "n_words", "incl_weight", "score_consensus"]
                + sorted(c for c in smp.columns if c.startswith("score_cfg")))
    smp[key_cols].to_csv(out_dir / f"screen_key_{tag}.csv", index=False)
    write_sheets(smp, out_dir, tag, rng)
    log.info("wrote %d narratives -> %s", len(smp), out_dir)
    print(pd.crosstab(smp["industry"], smp["reviewer"]))


if __name__ == "__main__":
    main()
