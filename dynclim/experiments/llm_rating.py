"""Apply the frozen narrative rating guide with a language model.

Protocol: docs/validity_study/rating_protocol_draft.md §5.3-5.5. The prompt is
the frozen RATER_GUIDE.md verbatim plus a short frame and an output schema.
The guide's SHA-256 is checked against the value recorded at the freeze before
any request is sent, and the SHA-256 of the full system prompt is written into
every output row.

Runs under the model environment (.llmVenv, Python 3.12, anthropic SDK), not
the toolkit venv, and by file path (the dynclim package __init__ imports the
transformer stack, which this environment does not have):

    .llmVenv/bin/python dynclim/experiments/llm_rating.py prompt
    .llmVenv/bin/python dynclim/experiments/llm_rating.py pool --industry msha --n 6000 --seed 20261110 --out <csv>
    .llmVenv/bin/python dynclim/experiments/llm_rating.py rate --input <csv> --out <jsonl> [--mode live|batch]
    .llmVenv/bin/python dynclim/experiments/llm_rating.py collect --out <jsonl>
    .llmVenv/bin/python dynclim/experiments/llm_rating.py check --ratings <jsonl> --consensus <csv>

Input CSVs for `rate` need columns item_id, industry, text. Industry is one of
msha, rail, nrc, aviation. Output is JSONL, one row per (item_id, repeat), and
`rate` resumes: rows already present in --out are skipped.

Credentials: ANTHROPIC_API_KEY or MY_API_KEY in the environment. No indicator score is read or written by this module.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Literal, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict

log = logging.getLogger("llm_rating")
REPO = Path(__file__).resolve().parents[2]
STUDY = REPO / "validity_study" / "stage2_consolidated_2026-10-26"
GUIDE = STUDY / "RATER_GUIDE.md"
GUIDE_SHA256 = "c24d9b0ba0187ddf4d300ea2e7d18a322b43fc2c3597d259804d6d30b640a1b1"  # frozen 2026-11-09

DEFAULT_MODEL = "claude-opus-5-5"
DEFAULT_EFFORT = "high"
MAX_TOKENS = 16000

INDUSTRY_LABEL = {
    "msha": "Mining (MSHA coal mine accident narrative)",
    "rail": "Rail (FRA railroad accident narrative)",
    "nrc": "Nuclear (NRC power-reactor event notification)",
    "aviation": "Aviation (ASRS air traffic controller report)",
}

FRAME = (
    "You are one of the raters in a content-validity study of safety narratives. "
    "Apply the rating guide below exactly as written. It is the same guide the two "
    "human raters use, and nothing outside it should influence the rating. Work "
    "through the steps in order and return only the JSON object described by the "
    "output schema.\n\n"
    "Output fields: cant_judge (true when the answer is CJ); rating (1 to 5, or null "
    "when cant_judge is true); content_present (which of A, B, C, D from step 2 the "
    "narrative states; empty when cant_judge is true); tone_adjustment (-1, 0 or 1, "
    "the step 4 adjustment already included in the rating; 0 when cant_judge is "
    "true); basis (one sentence naming the step 1 gate or the step 2 content and "
    "step 3 row that decided the rating).\n\n"
    "=============== RATING GUIDE ===============\n\n"
)


class Rating(BaseModel):
    model_config = ConfigDict(extra="forbid")  # schema needs additionalProperties: false

    cant_judge: bool
    rating: Optional[Literal[1, 2, 3, 4, 5]]
    content_present: List[Literal["A", "B", "C", "D"]]
    tone_adjustment: Literal[-1, 0, 1]
    basis: str


# ----------------------------------------------------------------------------
# Prompt
# ----------------------------------------------------------------------------

def guide_text() -> str:
    data = GUIDE.read_bytes()
    h = hashlib.sha256(data).hexdigest()
    if h != GUIDE_SHA256:
        sys.exit(f"RATER_GUIDE.md hash {h[:12]}... does not match the frozen value "
                 f"{GUIDE_SHA256[:12]}...; refusing to run. Restore the frozen guide "
                 f"from RATER_GUIDE_v1.1_frozen_2026-11-09.md or update GUIDE_SHA256 "
                 f"deliberately with a new freeze.")
    return data.decode("utf-8")


def system_prompt() -> str:
    return FRAME + guide_text()


def prompt_sha256() -> str:
    return hashlib.sha256(system_prompt().encode("utf-8")).hexdigest()


def user_message(industry: str, text: str) -> str:
    return f"Industry: {INDUSTRY_LABEL[industry]}\n\nNarrative:\n{' '.join(str(text).split())}"


def request_params(industry: str, text: str, model: str, effort: str) -> dict:
    return dict(
        model=model,
        max_tokens=MAX_TOKENS,
        system=[{"type": "text", "text": system_prompt(),
                 "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user_message(industry, text)}],
        output_config={"effort": effort,
                       "format": {"type": "json_schema",
                                  "schema": Rating.model_json_schema()}},
    )


# ----------------------------------------------------------------------------
# Pool draw
# ----------------------------------------------------------------------------

def prior_eids() -> set:
    vs = REPO / "validity_study"
    files = sorted(vs.glob("stage1_*/screen_key_seed*.csv")) + [vs / "pilot_2026-10-06/rating_key_seed11.csv"]
    out = set()
    for f in files:
        out |= set(pd.read_csv(f, dtype=str)["eid"])
    out |= {l.strip() for l in (vs / "protocol_example_eids.txt").read_text().splitlines() if l.strip()}
    return out


def load_sampler():
    """Import sample_validity_ratings.py by path, bypassing the dynclim package __init__."""
    import importlib.util
    f = REPO / "dynclim" / "experiments" / "sample_validity_ratings.py"
    spec = importlib.util.spec_from_file_location("sample_validity_ratings", f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def draw_pool(industry: str, n: int, seed: int, min_words: int = 5) -> pd.DataFrame:
    smp = load_sampler()
    load_events, norm_text = smp.load_events, smp.norm_text
    ev = load_events(industry)
    ev["text"] = ev["text"].fillna("").astype(str)
    ev = ev[ev["text"].str.split().str.len() >= min_words]
    ev = ev.loc[~ev["text"].map(norm_text).duplicated()]
    excl = prior_eids()
    ev = ev[~ev["eid"].isin(excl)]
    n_pop = len(ev)
    s = ev.sample(n=min(n, n_pop), random_state=seed).copy()
    s.insert(0, "item_id", s["eid"])
    s["industry"] = industry
    s["pool_population"] = n_pop
    s["pool_seed"] = seed
    log.info("%s: population after exclusions %d; drew %d (seed %d)", industry, n_pop, len(s), seed)
    return s[["item_id", "eid", "industry", "severity", "text", "pool_population", "pool_seed"]]


# ----------------------------------------------------------------------------
# Rating: live and batch
# ----------------------------------------------------------------------------

def _client():
    import os
    import anthropic
    key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("MY_API_KEY")
    return anthropic.Anthropic(api_key=key, max_retries=6, timeout=300.0)


def _row_base(item_id: str, rep: int, model: str, effort: str, psha: str) -> dict:
    return {"item_id": item_id, "repeat": rep, "model": model, "effort": effort,
            "prompt_sha256": psha, "guide_sha256": GUIDE_SHA256}


def _parse_response(msg) -> dict:
    out = {"stop_reason": msg.stop_reason, "response_model": msg.model,
           "request_id": getattr(msg, "_request_id", None),
           "input_tokens": msg.usage.input_tokens, "output_tokens": msg.usage.output_tokens,
           "cache_read_input_tokens": getattr(msg.usage, "cache_read_input_tokens", None),
           "cache_creation_input_tokens": getattr(msg.usage, "cache_creation_input_tokens", None)}
    if msg.stop_reason == "refusal":
        sd = getattr(msg, "stop_details", None)
        out["error"] = f"refusal:{getattr(sd, 'category', None)}"
        return out
    text = next((b.text for b in msg.content if b.type == "text"), "")
    try:
        r = Rating.model_validate_json(text)
    except Exception as e:  # noqa: BLE001
        out["error"] = f"parse:{e}"
        out["raw_text"] = text[:2000]
        return out
    if r.cant_judge != (r.rating is None):
        out["error"] = "inconsistent:cant_judge/rating"
    out.update({"cant_judge": r.cant_judge, "rating": r.rating,
                "content_present": "".join(r.content_present),
                "tone_adjustment": r.tone_adjustment, "basis": r.basis,
                "call": "CJ" if r.cant_judge else str(r.rating)})
    return out


def _done_keys(out_path: Path) -> set:
    if not out_path.exists():
        return set()
    keys = set()
    with out_path.open() as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                if "error" not in d:
                    keys.add((d["item_id"], d["repeat"]))
    return keys


def rate_live(items: pd.DataFrame, out_path: Path, model: str, effort: str,
              repeats: int, workers: int) -> None:
    psha = prompt_sha256()
    done = _done_keys(out_path)
    todo = [(r.item_id, r.industry, r.text, rep) for r in items.itertuples(index=False)
            for rep in range(repeats) if (r.item_id, rep) not in done]
    log.info("live: %d requests to send (%d already done)", len(todo), len(done))
    client = _client()

    def one(item_id, industry, text, rep):
        base = _row_base(item_id, rep, model, effort, psha)
        try:
            msg = client.messages.create(**request_params(industry, text, model, effort))
            base.update(_parse_response(msg))
        except Exception as e:  # noqa: BLE001
            base["error"] = f"api:{type(e).__name__}:{str(e)[:300]}"
        return base

    with out_path.open("a") as f, ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(one, *t) for t in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            row = fut.result()
            f.write(json.dumps(row) + "\n")
            f.flush()
            if i % 25 == 0 or i == len(futs):
                log.info("live: %d/%d written", i, len(futs))


def rate_batch_submit(items: pd.DataFrame, out_path: Path, model: str, effort: str,
                      repeats: int, chunk: int) -> None:
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request
    psha = prompt_sha256()
    done = _done_keys(out_path)
    manifest = out_path.with_suffix(".batches.json")
    man = json.loads(manifest.read_text()) if manifest.exists() else {"batches": []}
    pending = {(b["item_id"], b["repeat"]) for bb in man["batches"] for b in bb["items"]
               if not bb.get("collected")}
    todo = [(r.item_id, r.industry, r.text, rep) for r in items.itertuples(index=False)
            for rep in range(repeats) if (r.item_id, rep) not in done and (r.item_id, rep) not in pending]
    log.info("batch: %d requests to submit (%d done, %d pending in earlier batches)",
             len(todo), len(done), len(pending))
    client = _client()
    for start in range(0, len(todo), chunk):
        part = todo[start:start + chunk]
        reqs = [Request(custom_id=f"{i}", params=MessageCreateParamsNonStreaming(
            **request_params(ind, text, model, effort))) for i, (_, ind, text, _) in enumerate(part)]
        b = client.messages.batches.create(requests=reqs)
        man["batches"].append({"id": b.id, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                               "model": model, "effort": effort, "prompt_sha256": psha,
                               "items": [{"custom_id": f"{i}", "item_id": it, "repeat": rep}
                                         for i, (it, _, _, rep) in enumerate(part)]})
        manifest.write_text(json.dumps(man, indent=1))
        log.info("batch: submitted %s with %d requests", b.id, len(part))


def rate_batch_collect(out_path: Path, wait: bool) -> None:
    manifest = out_path.with_suffix(".batches.json")
    man = json.loads(manifest.read_text())
    client = _client()
    for bb in man["batches"]:
        if bb.get("collected"):
            continue
        while True:
            b = client.messages.batches.retrieve(bb["id"])
            if b.processing_status == "ended":
                break
            log.info("%s: %s, processing %d", bb["id"], b.processing_status, b.request_counts.processing)
            if not wait:
                break
            time.sleep(60)
        if b.processing_status != "ended":
            continue
        lookup = {it["custom_id"]: it for it in bb["items"]}
        n = 0
        with out_path.open("a") as f:
            for res in client.messages.batches.results(bb["id"]):
                it = lookup[res.custom_id]
                base = _row_base(it["item_id"], it["repeat"], bb["model"], bb["effort"], bb["prompt_sha256"])
                base["batch_id"] = bb["id"]
                if res.result.type == "succeeded":
                    base.update(_parse_response(res.result.message))
                else:
                    err = getattr(res.result, "error", None)
                    base["error"] = f"batch:{res.result.type}:{getattr(err, 'type', '')}"
                f.write(json.dumps(base) + "\n")
                n += 1
        bb["collected"] = True
        manifest.write_text(json.dumps(man, indent=1))
        log.info("%s: collected %d results (succeeded %d, errored %d)", bb["id"], n,
                 b.request_counts.succeeded, b.request_counts.errored)


def load_ratings(path: Path) -> pd.DataFrame:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    ok = df[df.get("error").isna()] if "error" in df else df
    log.info("%s: %d rows, %d usable", path.name, len(df), len(ok))
    return ok


# ----------------------------------------------------------------------------
# Check against the settled stage-2 calls
# ----------------------------------------------------------------------------

def _qwk(a: pd.Series, b: pd.Series) -> float:
    import numpy as np
    a = a.astype(int).to_numpy(); b = b.astype(int).to_numpy()
    k = 5
    O = np.zeros((k, k))
    for x, y in zip(a, b):
        O[x - 1, y - 1] += 1
    W = np.array([[(i - j) ** 2 for j in range(k)] for i in range(k)]) / (k - 1) ** 2
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    return float(1 - (W * O).sum() / (W * E).sum()) if (W * E).sum() else float("nan")


def _kappa2(a: pd.Series, b: pd.Series) -> float:
    a = a.astype(bool); b = b.astype(bool)
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def check(ratings: pd.DataFrame, consensus: pd.DataFrame) -> pd.DataFrame:
    r = ratings[ratings["repeat"] == 0].merge(consensus, left_on="item_id", right_on="rating_id")
    r["model_cj"] = r["call"] == "CJ"
    r["human_cj"] = r["final_call"] == "CJ"
    rows = []
    for ind, g in [("all", r)] + list(r.groupby("industry")):
        both = g[~g.model_cj & ~g.human_cj]
        m = both["call"].astype(int); h = both["final_call"].astype(int)
        rows.append({"industry": ind, "n": len(g), "cj_agreement": round((g.model_cj == g.human_cj).mean(), 2),
                     "cj_kappa": round(_kappa2(g.model_cj, g.human_cj), 2),
                     "model_cj_rate": round(g.model_cj.mean(), 2), "human_cj_rate": round(g.human_cj.mean(), 2),
                     "both_rated": len(both),
                     "exact": round((m == h).mean(), 2) if len(both) else None,
                     "within_1": round(((m - h).abs() <= 1).mean(), 2) if len(both) else None,
                     "qwk": round(_qwk(m, h), 2) if len(both) > 2 else None,
                     "mean_model_minus_human": round((m - h).mean(), 2) if len(both) else None})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------

def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("prompt", help="verify the frozen guide and print the system prompt and its hash")

    p = sub.add_parser("pool", help="draw a random pool from one industry, excluding all prior eids")
    p.add_argument("--industry", required=True, choices=list(INDUSTRY_LABEL))
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", required=True)

    p = sub.add_parser("rate", help="rate the narratives in a CSV (item_id, industry, text)")
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True, help="JSONL; appended to and resumed")
    p.add_argument("--mode", choices=["live", "batch"], default="live")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--effort", default=DEFAULT_EFFORT, choices=["low", "medium", "high", "xhigh", "max"])
    p.add_argument("--repeats", type=int, default=1, help="independent runs per narrative (3 for run-to-run agreement)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--chunk", type=int, default=5000, help="requests per batch submission")
    p.add_argument("--limit", type=int, default=None, help="only the first N input rows (smoke test)")

    p = sub.add_parser("collect", help="collect finished batches into the JSONL")
    p.add_argument("--out", required=True)
    p.add_argument("--wait", action="store_true", help="poll until every batch has ended")

    p = sub.add_parser("check", help="agreement of model ratings with the settled stage-2 calls")
    p.add_argument("--ratings", required=True)
    p.add_argument("--consensus", default=str(STUDY / "stage2_consensus_v1.1.csv"))

    a = ap.parse_args(argv)

    if a.cmd == "prompt":
        print(system_prompt())
        print(f"\n--- guide sha256 {GUIDE_SHA256}\n--- prompt sha256 {prompt_sha256()}", file=sys.stderr)
    elif a.cmd == "pool":
        df = draw_pool(a.industry, a.n, a.seed)
        df.to_csv(a.out, index=False)
        print(f"wrote {len(df)} rows to {a.out}")
    elif a.cmd == "rate":
        items = pd.read_csv(a.input, dtype=str).fillna("")
        missing = {"item_id", "industry", "text"} - set(items.columns)
        if missing:
            sys.exit(f"input is missing columns {missing}")
        if a.limit:
            items = items.head(a.limit)
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        if a.mode == "live":
            rate_live(items, out, a.model, a.effort, a.repeats, a.workers)
        else:
            rate_batch_submit(items, out, a.model, a.effort, a.repeats, a.chunk)
    elif a.cmd == "collect":
        rate_batch_collect(Path(a.out), a.wait)
    elif a.cmd == "check":
        ratings = load_ratings(Path(a.ratings))
        cons = pd.read_csv(a.consensus, dtype=str)
        tab = check(ratings, cons)
        print(tab.to_string(index=False))
        tokens = ratings[["input_tokens", "output_tokens", "cache_read_input_tokens"]].astype(float).mean()
        print("\nmean tokens per request:", tokens.round(0).to_dict())


if __name__ == "__main__":
    main()
