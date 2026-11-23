"""Draw the held-out human sample, stratified on the model's call.

Protocol §5.4. Input: the pool CSVs (from `llm_rating pool`) and the model's
pool ratings (from `llm_rating rate`/`collect`). Per industry, four strata on
the model's call - CJ, 1-2, 3, 4-5 - with equal targets (n/4); a stratum
shorter than its target is filled from the others in proportion to what is
left. Inclusion weight = stratum size in the pool / stratum size in the
sample, so results can be reweighted to the pool.

Writes, into --out:
  heldout_sheet_raterA.xlsx / raterB.xlsx  identical, blind: rating_id, industry, text, rating_1to5, cant_judge, notes
  heldout_key.csv                           eid, stratum, model call, weights, severity, indicator scores. KEEP FROM RATERS.

    .llmVenv/bin/python dynclim/experiments/draw_heldout_sample.py \
        --pool <msha.csv> <rail.csv> <nrc.csv> <aviation.csv> --ratings <pool_ratings.jsonl> \
        --n 75 --seed 20261110 --out validity_study/heldout_2026-11-XX
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("heldout")
REPO = Path(__file__).resolve().parents[2]
STRATA = ["CJ", "1-2", "3", "4-5"]
IND_LABEL = {"msha": "Mining", "rail": "Rail", "nrc": "Nuclear", "aviation": "Aviation"}


def stratum(call: str) -> str:
    return {"CJ": "CJ", "1": "1-2", "2": "1-2", "3": "3", "4": "4-5", "5": "4-5"}[str(call)]


def allocate(avail: dict, n: int) -> dict:
    """Equal targets, shortfalls redistributed in proportion to remaining supply."""
    take = {s: 0 for s in STRATA}
    remaining = n
    open_ = set(STRATA)
    while remaining > 0 and open_:
        share = remaining / len(open_)
        progressed = False
        for s in sorted(open_):
            room = avail[s] - take[s]
            add = int(min(room, np.floor(share)))
            if add > 0:
                take[s] += add; remaining -= add; progressed = True
            if take[s] >= avail[s]:
                open_.discard(s)
        if not progressed:  # distribute the last few one at a time
            for s in sorted(open_, key=lambda s: avail[s] - take[s], reverse=True):
                if remaining == 0:
                    break
                if avail[s] - take[s] > 0:
                    take[s] += 1; remaining -= 1
            break
    return take


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", nargs="+", required=True, help="pool CSVs from `llm_rating pool`")
    ap.add_argument("--ratings", required=True, help="model ratings JSONL on the pool (repeat 0 is used)")
    ap.add_argument("--n", type=int, default=75, help="narratives per industry")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from llm_rating import load_ratings, load_sampler  # noqa: E402
    scores = load_sampler().scores

    pool = pd.concat([pd.read_csv(f, dtype=str) for f in a.pool], ignore_index=True)
    r = load_ratings(Path(a.ratings))
    r = r[r["repeat"] == 0][["item_id", "call", "model", "prompt_sha256"]]
    df = pool.merge(r, on="item_id", how="inner")
    log.info("pool %d narratives, %d with a model call", len(pool), len(df))
    df["stratum"] = df["call"].map(stratum)

    rng = np.random.default_rng(a.seed)
    parts = []
    for ind, g in df.groupby("industry"):
        avail = g["stratum"].value_counts().reindex(STRATA, fill_value=0).to_dict()
        take = allocate(avail, a.n)
        log.info("%s: available %s -> take %s", ind, avail, take)
        for s, k in take.items():
            if k == 0:
                continue
            idx = rng.choice(g.index[g["stratum"] == s], size=k, replace=False)
            part = g.loc[idx].copy()
            part["n_stratum_pool"] = avail[s]
            part["n_stratum_sample"] = k
            part["incl_weight"] = avail[s] / k
            parts.append(part)
    smp = pd.concat(parts, ignore_index=True)

    # shuffle within industry block, assign blind ids
    smp = smp.sample(frac=1, random_state=a.seed).sort_values("industry", kind="stable").reset_index(drop=True)
    smp["rating_id"] = [f"H{i:04d}" for i in range(1, len(smp) + 1)]
    smp["industry_label"] = smp["industry"].map(IND_LABEL)

    # indicator scores for the key (never shown to raters)
    keys = []
    for ind, g in smp.groupby("industry"):
        sc = scores(ind)
        keys.append(g.merge(sc, on="eid", how="left"))
    key = pd.concat(keys, ignore_index=True).sort_values("rating_id")
    missing = key["score_consensus"].isna().sum()
    if missing:
        log.warning("%d sampled narratives have no indicator score", missing)

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    sheet = key[["rating_id", "industry_label", "text"]].rename(columns={"industry_label": "industry"}).copy()
    sheet["text"] = sheet["text"].map(lambda s: " ".join(str(s).split()))
    for c in ["rating_1to5", "cant_judge", "notes"]:
        sheet[c] = ""
    for rater in "AB":
        p = out / f"heldout_sheet_rater{rater}.xlsx"
        sheet.to_excel(p, index=False, sheet_name="Ratings")
        from openpyxl import load_workbook
        from openpyxl.styles import Alignment, Font
        wb = load_workbook(p); ws = wb.active
        for i, w in enumerate([10, 10, 120, 12, 11, 40], 1):
            ws.column_dimensions[ws.cell(1, i).column_letter].width = w
            ws.cell(1, i).font = Font(bold=True)
        for row in ws.iter_rows():
            for c in row:
                c.alignment = Alignment(wrap_text=True, vertical="top")
        ws.freeze_panes = "A2"
        wb.save(p)
    key_cols = ["rating_id", "industry", "eid", "severity", "stratum", "call", "model", "prompt_sha256",
                "n_stratum_pool", "n_stratum_sample", "incl_weight", "pool_population", "pool_seed"]
    key_cols += [c for c in key.columns if c.startswith("score_")]
    key[key_cols].rename(columns={"call": "model_call"}).to_csv(out / "heldout_key.csv", index=False)
    (out / "README.md").write_text(
        f"# Held-out sample (drawn {pd.Timestamp.today():%Y-%m-%d}, seed {a.seed})\n\n"
        f"{len(smp)} narratives, {a.n} per industry, stratified on the model's call "
        f"(protocol §5.4). Sheets are identical and blind; `heldout_key.csv` holds the "
        f"strata, the model's call and the indicator scores. KEEP THE KEY FROM RATERS.\n\n"
        + "```\n" + pd.crosstab(smp["industry"], smp["stratum"]).to_string() + "\n```\n")
    print(pd.crosstab(smp["industry"], smp["stratum"]))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
