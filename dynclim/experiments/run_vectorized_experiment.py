"""
Vectorized full-corpus climate-scoring experiment
=================================================

Production driver for large corpora (built for OSHA ITA, 2.17M reports).
Produces a checkpoint store **identical in layout to ExperimentRunner's**
(checkpoints/<experiment>/config_registry.csv + registry_constants.json +
_cfg/<config_id>/{results.parquet, meta.json}) so the R-side loaders work
unchanged — but drives the scoring through
:mod:`dynclim.features.safety.vectorized_scorer` kernels (~500x faster
per config; equivalence-verified against the per-report path).

Phased + resumable (checkpoints/<experiment>/_corpus/):
    S  segment every report        → seg chunks (indices + unique texts)
    E  embed unique segments       → embs.f32.npy
    V  sentiment (vader)           → sents.f32.npy
    R  score configs + write store → _cfg/<id>/results.parquet

Each phase skips work whose artifacts already exist; killing and
relaunching is safe at any point.

Config grid: the 15-combo production axis (top_k k3/k5 x 5 post_threshold
composites + sentinel x 5 apply_over='all' composites) crossed with every
requested (embedding, sentiment) pair. --embedding/--sentiment take
comma-separated lists; --full-grid is shorthand for the historical
3 x 3 x 15 = 135-config production grid (as in msha_06-01-2026). The
registry is built ONCE over the full cross, so config_ids are identical
to the historical ExperimentRunner stores (ids are assigned by sorted
signature, verified 15/15 against osha/msha registries). power_attention
arms use the SPEC semantics (attn_quantile 0.0 = no-op) — historical MPS
stores' power arms are corrupted by the torch-2.0.1 nanquantile bug (see
test_scoring_equivalence.py).

Segmentation is shared across pairs; embeddings/sentiments are cached per
model/method in _corpus/ (embs_<slug>.f32.npy / sents_<slug>.f32.npy,
falling back to the legacy embs.f32.npy / sents.f32.npy written by the
single-pair driver for MiniLM / vader).

Run (by file path, from the directory whose checkpoints/ you want):
    cd dynclim/notebooks && \\
    ../../.dynClimVenv/bin/python ../experiments/run_vectorized_experiment.py \\
        --events ../data/processed/msha/events.parquet \\
        --experiment msha_2026-07-13 --full-grid
"""
from __future__ import annotations

import argparse
import gc
import gzip
import importlib.util
import json
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import pandas as pd
import torch
import torch.mps

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("vectorized_experiment")

REPO = Path(__file__).resolve().parents[2]
SEG_CHUNK = 100_000
EMB_BATCH = 200_000
MAX_CELLS = 200_000_000

# The historical production grid (msha_06-01-2026 and siblings)
FULL_EMBEDDINGS = ["all-MiniLM-L6-v2", "all-roberta-large-v1", "BAAI/bge-m3"]
FULL_SENTIMENTS = ["vader", "roberta", "siebert"]


def _slug(name: str) -> str:
    return name.replace("/", "_").replace(":", "-")


def _emb_path(cdir: Path, model: str) -> Path:
    p = cdir / f"embs_{_slug(model)}.f32.npy"
    if not p.exists() and model == "all-MiniLM-L6-v2" \
            and (cdir / "embs.f32.npy").exists():
        return cdir / "embs.f32.npy"     # legacy single-pair layout
    return p


def _sent_path(cdir: Path, method: str) -> Path:
    p = cdir / f"sents_{_slug(method)}.f32.npy"
    if not p.exists() and method == "vader" \
            and (cdir / "sents.f32.npy").exists():
        return cdir / "sents.f32.npy"    # legacy single-pair layout
    return p


def _grid():
    from dynclim.features.safety import CompositeConfig, ThresholdConfig
    post = [
        CompositeConfig(method="weighted_average", similarity_weight=0.5,
                        sentiment_weight=0.5, apply_over="post_threshold"),
        CompositeConfig(method="attention_weight", temperature=0.15,
                        apply_over="post_threshold"),
        CompositeConfig(method="attention_weight", temperature=1.00,
                        apply_over="post_threshold"),
        CompositeConfig(method="power_attention", power=0.5,
                        apply_over="post_threshold"),
        CompositeConfig(method="power_attention", power=2.0,
                        apply_over="post_threshold"),
    ]
    allc = [
        CompositeConfig(method="weighted_average", similarity_weight=0.5,
                        sentiment_weight=0.5, apply_over="all"),
        CompositeConfig(method="attention_weight", temperature=0.15,
                        apply_over="all"),
        CompositeConfig(method="attention_weight", temperature=1.00,
                        apply_over="all"),
        CompositeConfig(method="power_attention", power=0.5, apply_over="all"),
        CompositeConfig(method="power_attention", power=2.0, apply_over="all"),
    ]
    ths = [ThresholdConfig(method="top_k", k=3),
           ThresholdConfig(method="top_k", k=5)]
    sentinel = ThresholdConfig(method="top_k", k=99999)
    combos = [(t, c) for t in ths for c in post] + [(sentinel, c) for c in allc]
    return combos


def _build_processor(embedding_model, sentiment_method, store_path):
    from dynclim.features.safety.seg_core import SegmentationStore
    from dynclim.features.safety.unified_processor import Processor
    from dynclim.utils import loadDomainsFromYaml
    from tokenizers import Tokenizer

    proc = Processor(
        domain_definitions=loadDomainsFromYaml(
            REPO / "config/domains/safety_domains.yaml"),
        embedding_model_name=embedding_model,
        use_acceleration=True,
        sentiment_method=sentiment_method,
    )
    proc.store = SegmentationStore(str(store_path))
    proc.tok = Tokenizer.from_pretrained("bert-base-uncased")
    return proc


# ---------------------------------------------------------------------------
# Phase S — segmentation (chunked, resumable)
# ---------------------------------------------------------------------------

def phase_segment(proc, seg_cfg, events, cdir: Path):
    done = sorted(cdir.glob("seg_idx_*.npy"))
    start_chunk = len(done)
    n_chunks = (len(events) + SEG_CHUNK - 1) // SEG_CHUNK
    if start_chunk >= n_chunks:
        log.info("[S] all %d segmentation chunks present", n_chunks)
        return

    # Rebuild the unique-segment dict from prior chunks (resume support)
    uniq = {}
    for p in sorted(cdir.glob("seg_uniq_*.txt.gz")):
        with gzip.open(p, "rt", encoding="utf-8") as fh:
            for line in fh:
                uniq[line.rstrip("\n")] = len(uniq)
    log.info("[S] resume at chunk %d/%d (%d uniques so far)",
             start_chunk, n_chunks, len(uniq))

    texts = events["narrative"].astype(str).tolist()
    for ci in range(start_chunk, n_chunks):
        t0 = time.time()
        lo, hi = ci * SEG_CHUNK, min((ci + 1) * SEG_CHUNK, len(texts))
        idx_rows, lens, new_uniq = [], [], []
        for text in texts[lo:hi]:
            segs = [s if isinstance(s, str) else getattr(s, "text", str(s))
                    for s in proc.segment_text(text, seg_cfg)]
            for s in segs:
                if s not in uniq:
                    uniq[s] = len(uniq)
                    new_uniq.append(s)
                idx_rows.append(uniq[s])
            lens.append(len(segs))
        np.save(cdir / f"seg_len_{ci:04d}.npy",
                np.asarray(lens, dtype=np.int32))
        with gzip.open(cdir / f"seg_uniq_{ci:04d}.txt.gz", "wt",
                       encoding="utf-8") as fh:
            fh.write("\n".join(new_uniq) + ("\n" if new_uniq else ""))
        # write idx last: its presence marks the chunk complete
        np.save(cdir / f"seg_idx_{ci:04d}.npy",
                np.asarray(idx_rows, dtype=np.int64))
        proc._text_segment_cache.clear()   # cap in-memory cache growth
        log.info("[S] chunk %d/%d: %d reports in %.0fs (%.0f/s), "
                 "%d uniques total", ci + 1, n_chunks, hi - lo,
                 time.time() - t0, (hi - lo) / (time.time() - t0), len(uniq))


def load_corpus_segments(cdir: Path):
    lens = np.concatenate([np.load(p) for p in sorted(cdir.glob("seg_len_*.npy"))])
    idx = np.concatenate([np.load(p) for p in sorted(cdir.glob("seg_idx_*.npy"))])
    uniques = []
    for p in sorted(cdir.glob("seg_uniq_*.txt.gz")):
        with gzip.open(p, "rt", encoding="utf-8") as fh:
            uniques.extend(line.rstrip("\n") for line in fh)
    assert lens.sum() == len(idx), "segment index/length mismatch"
    return lens, idx, uniques


# ---------------------------------------------------------------------------
# Phase E / V — unique-segment embeddings + sentiment (chunked, resumable)
# ---------------------------------------------------------------------------

def phase_embed(proc, uniques, out: Path) -> np.ndarray:
    if out.exists():
        return np.load(out, mmap_mode="r")
    import torch.nn.functional as F
    proc.cache_embeddings = False          # no 11.7M-entry dict, just encode
    parts = []
    for lo in range(0, len(uniques), EMB_BATCH):
        t0 = time.time()
        part = proc.encode_segments(uniques[lo:lo + EMB_BATCH])
        part = F.normalize(torch.from_numpy(np.ascontiguousarray(part)),
                           p=2, dim=1).numpy()
        parts.append(part.astype(np.float32, copy=False))
        log.info("[E] embedded %d/%d uniques (%.0f/s)",
                 min(lo + EMB_BATCH, len(uniques)), len(uniques),
                 len(part) / (time.time() - t0))
    embs = np.concatenate(parts) if parts else np.zeros((0, 384), np.float32)
    np.save(out, embs)
    del embs, parts
    return np.load(out, mmap_mode="r")


def phase_sentiment(proc, uniques, out: Path) -> np.ndarray:
    if out.exists():
        return np.load(out)
    proc.cache_sentiments = False
    parts = []
    for lo in range(0, len(uniques), EMB_BATCH):
        t0 = time.time()
        vals = proc.get_sentiments_batch(uniques[lo:lo + EMB_BATCH])
        parts.append(np.asarray(vals, dtype=np.float32))
        log.info("[V] sentiment %d/%d uniques (%.0f/s)",
                 min(lo + EMB_BATCH, len(uniques)), len(uniques),
                 len(vals) / (time.time() - t0))
    sents = np.concatenate(parts) if parts else np.zeros(0, np.float32)
    np.save(out, sents)
    return sents


# ---------------------------------------------------------------------------
# Phase R — score + write ExperimentRunner-compatible store
# ---------------------------------------------------------------------------

def pair_combos(embedding_model, sentiment_method, seg_cfg):
    return [{"embedding_model": embedding_model, "segmentation": seg_cfg,
             "threshold": t, "composite": c, "sentiment_method":
             sentiment_method} for t, c in _grid()]


def pair_done(runner, experiment, registry, combos) -> bool:
    for combo in combos:
        cid = runner._config_id_for_combo(registry, combo)
        _, results_base, _ = runner._paths_for_config(experiment, cid)
        if not Path(results_base + ".parquet").exists():
            return False
    return True


def phase_score(proc, runner, experiment, embedding_model, sentiment_method,
                seg_cfg, eids, lens, idx, embs, sents, registry):
    from dynclim.experiments.runner import _canon_payload
    from dynclim.features.safety.vectorized_scorer import (
        _attention_composite, _power_composite, _select_topk, _wa_composite)

    proc._ensure_items_ready()
    I = len(proc._item_keys)
    col_of = [proc.item_to_column_name[obj] for obj in proc.all_items]
    item_domains = [proc.item_to_domain[obj] for obj in proc.all_items]
    domains = list(proc.domain_definitions.keys())
    dom_masks = {d: np.array([g == d for g in item_domains]) for d in domains}

    combos = pair_combos(embedding_model, sentiment_method, seg_cfg)
    todo = []
    for combo in combos:
        cid = runner._config_id_for_combo(registry, combo)
        cfg_dir, results_base, meta_path = runner._paths_for_config(experiment, cid)
        if Path(results_base + ".parquet").exists():
            log.info("[R] config %d already written, skipping", cid)
            continue
        todo.append((cid, combo, cfg_dir, results_base, meta_path))
    if not todo:
        log.info("[R] all configs already written")
        return

    n = len(eids)
    usims = torch.from_numpy(np.asarray(embs)) @ \
        torch.from_numpy(proc._item_matrix_np.astype(np.float32)).T
    usent = torch.from_numpy(sents)
    offsets = np.concatenate([[0], np.cumsum(lens)])

    # Preallocated wide outputs per config (float32; NaN = no segments)
    buf = {cid: {"final": np.full((n, I), np.nan, np.float32),
                 "sent": np.full((n, I), np.nan, np.float32),
                 "strength": np.full((n, I), np.nan, np.float32),
                 "nseg": np.full((n, I), np.nan, np.float32)}
           for cid, *_ in todo}
    timing = {cid: 0.0 for cid, *_ in todo}

    nonzero = np.flatnonzero(lens > 0)
    order = nonzero[np.argsort(lens[nonzero], kind="stable")]
    chunks, start = [], 0
    while start < len(order):
        end = start + 1
        while end < len(order) and \
                (end + 1 - start) * lens[order[end]] * I <= MAX_CELLS:
            end += 1
        chunks.append(order[start:end])
        start = end

    for ci, chunk in enumerate(chunks):
        R, S = len(chunk), int(lens[chunk].max())
        sims = torch.full((R, S, I), float("-inf"), dtype=torch.float32)
        sent_t = torch.zeros((R, S, 1), dtype=torch.float32)
        valid = torch.zeros((R, S), dtype=torch.bool)
        for ri, r in enumerate(chunk):
            sl = idx[offsets[r]:offsets[r + 1]]
            sims[ri, :len(sl)] = usims[sl]
            sent_t[ri, :len(sl), 0] = usent[sl]
            valid[ri, :len(sl)] = True
        valid3 = valid.unsqueeze(-1).expand(R, S, I)
        m = sent_t.expand(R, S, I)
        n_valid = torch.from_numpy(lens[chunk].astype(np.float32))

        for cid, combo, *_ in todo:
            t0 = time.time()
            tcfg, ccfg = combo["threshold"], combo["composite"]
            if getattr(ccfg, "apply_over", "post_threshold") == "all":
                sel = valid3
                nseg = n_valid.unsqueeze(1).expand(R, I)
            else:
                sel = _select_topk(sims, valid3, getattr(tcfg, "k", 3))
                nseg = sel.sum(dim=1).to(torch.float32)
            meth = ccfg.method
            if meth == "weighted_average":
                ss, st = _wa_composite(sims, m, sel)
            elif meth == "attention_weight":
                ss, st = _attention_composite(sims, m, sel, ccfg.temperature)
            elif meth == "power_attention":
                ss, st = _power_composite(sims, m, sel, ccfg.power)
            else:
                raise ValueError(meth)
            b = buf[cid]
            rows = chunk
            b["final"][rows] = (ss * st).numpy()
            b["sent"][rows] = ss.numpy()
            b["strength"][rows] = st.numpy()
            b["nseg"][rows] = np.asarray(nseg)
            timing[cid] += time.time() - t0
        if (ci + 1) % 20 == 0 or ci + 1 == len(chunks):
            log.info("[R] chunk %d/%d done", ci + 1, len(chunks))

    from dynclim.experiments.runner import _atomic_write_df_parquet
    for cid, combo, cfg_dir, results_base, meta_path in todo:
        b = buf[cid]
        has_segs = lens > 0
        df = pd.DataFrame({
            "config_id": np.int32(cid),
            "report_id": eids,
            "overall_final_score": b["final"].mean(axis=1),
            "overall_sent": b["sent"].mean(axis=1),
            "overall_strength": b["strength"].mean(axis=1),
            "processing_success": True,
            "processing_time": np.float32(0.0),
            "error_message": None,
        })
        for j, col in enumerate(col_of):
            df[f"{col}_final"] = b["final"][:, j]
            df[f"{col}_sent"] = b["sent"][:, j]
            df[f"{col}_strength"] = b["strength"][:, j]
            df[f"{col}_n_segments"] = b["nseg"][:, j]
        for d, mask in dom_masks.items():
            ds = b["final"][:, mask].mean(axis=1)
            ds[~has_segs] = 0.0          # per-report path: empty → domain 0.0
            df[f"{d}_domain_score"] = ds
        os.makedirs(cfg_dir, exist_ok=True)
        _atomic_write_df_parquet(results_base + ".parquet", df,
                                 compression="zstd")
        payload = _canon_payload(combo["embedding_model"],
                                 combo["segmentation"], combo["threshold"],
                                 combo["composite"],
                                 combo["sentiment_method"])
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump({"config_id": cid, "experiment_name": experiment,
                       "params": payload, "timing_sec": timing[cid],
                       "scoring_path": "vectorized_scorer"},
                      fh, indent=2, sort_keys=True)
        log.info("[R] wrote config %d (%.1fs scoring)", cid, timing[cid])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True)
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--embedding", default="all-MiniLM-L6-v2",
                    help="comma-separated embedding model list")
    ap.add_argument("--sentiment", default="vader",
                    help="comma-separated sentiment method list")
    ap.add_argument("--full-grid", action="store_true",
                    help="historical 3x3x15=135-config production grid")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    sys.path.insert(0, str(REPO))
    from dynclim.experiments.runner import ExperimentRunner
    from dynclim.features.safety import SegmentationConfig
    from dynclim.utils import loadDomainsFromYaml

    if args.full_grid:
        embeddings, sentiments = FULL_EMBEDDINGS, FULL_SENTIMENTS
    else:
        embeddings = [s.strip() for s in args.embedding.split(",") if s.strip()]
        sentiments = [s.strip() for s in args.sentiment.split(",") if s.strip()]

    root = Path("checkpoints") / args.experiment
    cdir = root / "_corpus"
    cdir.mkdir(parents=True, exist_ok=True)

    events = pd.read_parquet(args.events, columns=["eid", "narrative"])
    if args.limit:
        events = events.head(args.limit)
    eids = events["eid"].astype(str).tolist()
    log.info("%d reports | experiment=%s | %d embeddings x %d sentiments "
             "x 15 = %d configs", len(events), args.experiment,
             len(embeddings), len(sentiments),
             len(embeddings) * len(sentiments) * 15)

    seg_cfg = SegmentationConfig(method="sentence", min_length=5,
                                 backend="auto")

    # Registry over the FULL cross, built once up front: config_ids are
    # assigned by sorted signature, so they match historical stores that
    # used the same grid (e.g. msha_06-01-2026) and never depend on which
    # pair runs first.
    runner = ExperimentRunner(
        domain_definitions=loadDomainsFromYaml(
            REPO / "config/domains/safety_domains.yaml"),
        embedding_model_names=embeddings)
    all_combos = [c for emb in embeddings for sent in sentiments
                  for c in pair_combos(emb, sent, seg_cfg)]
    reg_path = Path(runner._registry_path(args.experiment))
    const_path = root / "registry_constants.json"
    snapshots = {p: p.read_bytes() for p in (reg_path, const_path)
                 if p.exists()}
    old_reg = pd.read_csv(reg_path) if reg_path.exists() else None
    registry = runner._build_or_load_registry(args.experiment, all_combos)
    if old_reg is not None and (root / "_cfg").exists():
        # Rebuilding the registry over a different grid reassigns ids by
        # sorted signature; existing _cfg/<id> dirs would be misattributed.
        chk = old_reg[["signature", "config_id"]].merge(
            registry[["signature", "config_id"]], on="signature",
            how="left", suffixes=("_old", "_new"))
        moved = chk[chk["config_id_old"] != chk["config_id_new"]]
        if len(moved):
            for p, raw in snapshots.items():   # undo the overwrite
                p.write_bytes(raw)
            raise SystemExit(
                f"FATAL: {len(moved)} existing configs would change id under "
                f"the requested grid (e.g. signature "
                f"{moved.iloc[0]['signature'][:80]}…: "
                f"{moved.iloc[0]['config_id_old']} -> "
                f"{moved.iloc[0]['config_id_new']}). Existing _cfg/ results "
                f"would be misattributed. Use a fresh --experiment dir, or "
                f"rerun with the exact grid the store was created with.")

    # Segmentation is model-independent; only build a processor for it if
    # chunks are actually missing (skips a model load on resume).
    n_chunks = (len(events) + SEG_CHUNK - 1) // SEG_CHUNK
    if len(list(cdir.glob("seg_idx_*.npy"))) < n_chunks:
        proc = _build_processor(embeddings[0], sentiments[0],
                                root / "seg_cache.sqlite")
        phase_segment(proc, seg_cfg, events, cdir)
        del proc
        gc.collect()
    lens, idx, uniques = load_corpus_segments(cdir)
    log.info("corpus: %d reports, %d segment slots, %d uniques",
             len(lens), len(idx), len(uniques))

    for emb in embeddings:
        for sent in sentiments:
            combos = pair_combos(emb, sent, seg_cfg)
            if pair_done(runner, args.experiment, registry, combos):
                log.info("=== %s + %s: all 15 configs written, skipping",
                         emb, sent)
                continue
            log.info("=== %s + %s", emb, sent)
            proc = _build_processor(emb, sent, root / "seg_cache.sqlite")
            embs = phase_embed(proc, uniques, _emb_path(cdir, emb))
            sents = phase_sentiment(proc, uniques, _sent_path(cdir, sent))
            phase_score(proc, runner, args.experiment, emb, sent, seg_cfg,
                        eids, lens, idx, embs, sents, registry)
            del proc, embs, sents
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
    log.info("DONE")


if __name__ == "__main__":
    main()
