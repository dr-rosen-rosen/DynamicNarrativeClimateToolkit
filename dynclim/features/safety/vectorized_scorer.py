"""
Vectorized corpus-level scoring path
====================================

Re-implements the Processor's per-report GPU scoring loop
(`_process_report_all_items`) as corpus-level tensor operations, for
large-corpus production runs. Numerically equivalent to the per-report
path — verified against the frozen goldens in ``scoring_goldens/`` by
``test_scoring_equivalence.py``. Any change here must keep that test green.

Why: profiling showed ~90% of per-config time in the per-report loop is
device-transfer and Python overhead (one CPU→GPU→CPU round trip per report
per config); the arithmetic itself is a small matmul. This path computes
unique-segment similarities ONCE per corpus, then does selection +
composite aggregation per config as padded batched tensor ops on CPU.

Input preparation (segmentation, embeddings, sentiment) deliberately REUSES
the Processor's own methods, so cached values and numeric outputs are
identical; only the aggregation is reimplemented.

Supported configs (the production axis): threshold ``top_k``; composites
``weighted_average`` / ``attention_weight`` / ``power_attention`` (with
``apply_over`` ∈ {post_threshold, all}); anything else raises rather than
silently diverging.

Usage:
    corpus = prepare_corpus(proc, report_data, seg_cfg)
    df = score_configs(proc, corpus, [(label, thresh_cfg, comp_cfg), ...])
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_PAD = float("-inf")


@dataclass
class VectorizedCorpus:
    """Segment-indexed corpus ready for config scoring."""
    eids: List[str]
    seg_indices: List[np.ndarray]      # per report: indices into unique arrays
    unique_embs: np.ndarray            # (U, D) float32, L2-normalized
    unique_sents: np.ndarray           # (U,) float32

    @property
    def n_reports(self) -> int:
        return len(self.eids)


def prepare_corpus(proc, report_data: Sequence[Tuple[str, str]],
                   segmentation_config,
                   log_every: int = 200_000) -> VectorizedCorpus:
    """Segment + embed + sentiment via the Processor's own (cached) methods."""
    uniq: Dict[str, int] = {}
    eids, seg_indices = [], []
    ordered_segments: List[str] = []

    for i, (rid, text) in enumerate(report_data):
        segs = [s if isinstance(s, str) else getattr(s, "text", str(s))
                for s in proc.segment_text(text, segmentation_config)]
        idx = np.empty(len(segs), dtype=np.int64)
        for j, s in enumerate(segs):
            u = uniq.get(s)
            if u is None:
                u = len(ordered_segments)
                uniq[s] = u
                ordered_segments.append(s)
            idx[j] = u
        eids.append(rid)
        seg_indices.append(idx)
        if log_every and (i + 1) % log_every == 0:
            logger.info("segmented %d/%d reports (%d unique segments)",
                        i + 1, len(report_data), len(ordered_segments))

    logger.info("encoding %d unique segments…", len(ordered_segments))
    embs = proc.encode_segments(ordered_segments) if ordered_segments else \
        np.zeros((0, getattr(proc, "embedding_dim", 384)), dtype=np.float32)
    # _gpu_matmul re-normalizes segment embeddings before the matmul; match it.
    embs = F.normalize(torch.from_numpy(np.ascontiguousarray(embs)),
                       p=2, dim=1).numpy() if len(embs) else embs

    logger.info("sentiment for %d unique segments…", len(ordered_segments))
    sents = (np.asarray(proc.get_sentiments_batch(ordered_segments),
                        dtype=np.float32)
             if ordered_segments else np.zeros(0, dtype=np.float32))

    return VectorizedCorpus(eids=eids, seg_indices=seg_indices,
                            unique_embs=embs.astype(np.float32, copy=False),
                            unique_sents=sents)


# ---------------------------------------------------------------------------
# Composite kernels — exact ports of the toolkit's compute_composite_gpu
# implementations, operating on (R, S, I) padded tensors with a (R, S, I)
# boolean selection mask. All computation float32, matching the GPU path.
# ---------------------------------------------------------------------------

def _wa_composite(s, m, sel):
    n = sel.sum(dim=1).clamp(min=1).to(torch.float32)          # (R, I)
    sent = (m * sel).sum(dim=1) / n
    strength = (s.masked_fill(~sel, 0.0)).sum(dim=1) / n
    none_sel = sel.sum(dim=1) == 0
    sent = sent.masked_fill(none_sel, 0.0)
    strength = strength.masked_fill(none_sel, 0.0)
    return sent, strength


def _attention_composite(s, m, sel, temperature):
    temp = max(float(temperature), 1e-6)
    x = (s / temp).masked_fill(~sel, _PAD)                     # (R, S, I)
    has_any = sel.any(dim=1)                                   # (R, I)
    col_max = torch.where(has_any, x.max(dim=1).values,
                          torch.zeros_like(has_any, dtype=x.dtype))
    w = torch.softmax(x - col_max.unsqueeze(1), dim=1)
    w = torch.nan_to_num(w, nan=0.0)
    sent = (w * m).sum(dim=1)
    strength = (w * s.masked_fill(~sel, 0.0)).sum(dim=1)
    sent = sent.masked_fill(~has_any, 0.0)
    strength = strength.masked_fill(~has_any, 0.0)
    return sent, strength


def _power_composite(s, m, sel, power):
    p = max(float(power), 1e-8)
    s_clip = s.masked_fill(~sel, 0.0).clamp(0.0, 1.0)
    w_raw = torch.where(sel & torch.isfinite(s), s_clip.pow(p),
                        torch.zeros_like(s_clip))
    w_sum = w_raw.sum(dim=1, keepdim=True)
    w = torch.where(w_sum > 0, w_raw / w_sum, torch.zeros_like(w_raw))
    sent = (w * m).sum(dim=1)
    strength = (w * s_clip).sum(dim=1)
    has_any = (sel & torch.isfinite(s)).any(dim=1)
    sent = sent.masked_fill(~has_any, 0.0)
    strength = strength.masked_fill(~has_any, 0.0)
    return sent, strength


def _select_topk(s, valid, k):
    """torch.topk selection per (report, item), matching
    TopKThresholder.select_segments_gpu with per-report k=min(k, S)."""
    S = s.shape[1]
    kk = int(min(int(k), S))
    x = s.masked_fill(~valid, _PAD)
    _, idx = torch.topk(x, k=kk, dim=1, largest=True, sorted=False)
    sel = torch.zeros_like(valid)
    sel.scatter_(1, idx, True)
    return sel & valid       # padded -inf rows may be picked when S < k


def score_configs(proc, corpus: VectorizedCorpus,
                  configs: Sequence[Tuple[str, object, object]],
                  max_chunk_cells: int = 200_000_000,
                  progress_every: int = 20) -> pd.DataFrame:
    """
    Score every (label, threshold_config, composite_config) over the corpus.

    Returns a long DataFrame: config_label, eid, item_key, final_score,
    sent_signal, strength_signal, n_segments — plus one
    ``item_key='domain::<name>'`` row per domain per report (final only),
    mirroring process_single_report's domain means (0.0 for reports with
    no segments, which also emit no item rows).
    """
    proc._ensure_items_ready()
    item_keys: List[str] = list(proc._item_keys)
    I = len(item_keys)
    item_domains = [proc.item_to_domain[obj] for obj in proc.all_items]
    domains = list(proc.domain_definitions.keys())
    dom_masks = {d: np.array([g == d for g in item_domains]) for d in domains}

    # Unique-segment similarities: ONE matmul for the whole corpus,
    # shared by every config (sims are config-independent).
    item_t = torch.from_numpy(
        np.ascontiguousarray(proc._item_matrix_np.astype(np.float32)))
    usims = (torch.from_numpy(corpus.unique_embs) @ item_t.T).contiguous() \
        if len(corpus.unique_embs) else torch.zeros((0, I))
    usent = torch.from_numpy(corpus.unique_sents)

    lengths = np.array([len(ix) for ix in corpus.seg_indices])
    nonempty = np.flatnonzero(lengths > 0)
    empty = np.flatnonzero(lengths == 0)
    # Length-sorted chunking keeps padding tight.
    order = nonempty[np.argsort(lengths[nonempty], kind="stable")]

    chunks: List[np.ndarray] = []
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order):
            s_max = lengths[order[end]]          # sorted → max is last
            if (end + 1 - start) * s_max * I > max_chunk_cells:
                break
            end += 1
        chunks.append(order[start:end])
        start = end

    frames: List[pd.DataFrame] = []
    for ci, chunk in enumerate(chunks):
        R = len(chunk)
        S = int(lengths[chunk].max())
        flat = np.concatenate([corpus.seg_indices[r] for r in chunk])
        flat_t = torch.from_numpy(flat)

        sims = torch.full((R, S, I), _PAD, dtype=torch.float32)
        sent = torch.zeros((R, S, 1), dtype=torch.float32)
        valid = torch.zeros((R, S), dtype=torch.bool)
        pos = 0
        rows_sims = usims.index_select(0, flat_t)
        rows_sent = usent.index_select(0, flat_t)
        for ri, r in enumerate(chunk):
            n = lengths[r]
            sims[ri, :n] = rows_sims[pos:pos + n]
            sent[ri, :n, 0] = rows_sent[pos:pos + n]
            valid[ri, :n] = True
            pos += n
        valid3 = valid.unsqueeze(-1).expand(R, S, I)
        m = sent.expand(R, S, I)
        n_valid = torch.from_numpy(lengths[chunk]).to(torch.int32)

        chunk_eids = [corpus.eids[r] for r in chunk]

        for label, thresh_cfg, comp_cfg in configs:
            apply_over = getattr(comp_cfg, "apply_over", "post_threshold")
            if apply_over == "all":
                sel = valid3
                n_seg = n_valid.unsqueeze(1).expand(R, I)
            else:
                method = getattr(thresh_cfg, "method", None)
                if method != "top_k":
                    raise ValueError(
                        f"vectorized path supports top_k thresholds only, "
                        f"got {method!r} — extend _select_topk first")
                sel = _select_topk(sims, valid3, getattr(thresh_cfg, "k", 3))
                n_seg = sel.sum(dim=1).to(torch.int32)

            cmethod = getattr(comp_cfg, "method", None)
            if cmethod == "weighted_average":
                sent_sig, strength = _wa_composite(sims, m, sel)
            elif cmethod == "attention_weight":
                sent_sig, strength = _attention_composite(
                    sims, m, sel, getattr(comp_cfg, "temperature", 0.25))
            elif cmethod == "power_attention":
                # CompositeConfig.__post_init__ coerces None -> 0.0, and
                # q=0.0 makes the quantile prefilter a no-op (threshold is
                # the per-column min); only a real quantile is unsupported.
                aq = getattr(comp_cfg, "attn_quantile", None)
                if aq is not None and float(aq) > 0.0:
                    raise ValueError("attn_quantile > 0 not supported in "
                                     "vectorized power_attention")
                sent_sig, strength = _power_composite(
                    sims, m, sel, getattr(comp_cfg, "power", 1.0))
            else:
                raise ValueError(f"unsupported composite method {cmethod!r}")

            final = sent_sig * strength                        # (R, I)
            f_np = final.numpy()
            frames.append(pd.DataFrame({
                "config_label": label,
                "eid": np.repeat(chunk_eids, I),
                "item_key": np.tile(item_keys, R),
                "final_score": f_np.ravel().astype(np.float64),
                "sent_signal": sent_sig.numpy().ravel().astype(np.float64),
                "strength_signal": strength.numpy().ravel().astype(np.float64),
                "n_segments": np.asarray(n_seg).ravel(),
            }))
            # Domain means over item finals
            dom_frames = {
                d: f_np[:, mask].mean(axis=1) for d, mask in dom_masks.items()
            }
            frames.append(pd.DataFrame({
                "config_label": label,
                "eid": np.repeat(chunk_eids, len(domains)),
                "item_key": np.tile([f"domain::{d}" for d in domains], R),
                "final_score": np.column_stack(
                    [dom_frames[d] for d in domains]).ravel(),
                "sent_signal": np.nan,
                "strength_signal": np.nan,
                "n_segments": -1,
            }))
        if progress_every and (ci + 1) % progress_every == 0:
            logger.info("scored chunk %d/%d (%d reports x %d configs)",
                        ci + 1, len(chunks), R, len(configs))

    # Reports with no segments: no item rows; domain rows with 0.0
    if len(empty):
        for label, _, _ in configs:
            frames.append(pd.DataFrame({
                "config_label": label,
                "eid": np.repeat([corpus.eids[r] for r in empty],
                                 len(domains)),
                "item_key": np.tile([f"domain::{d}" for d in domains],
                                    len(empty)),
                "final_score": 0.0,
                "sent_signal": np.nan,
                "strength_signal": np.nan,
                "n_segments": -1,
            }))

    return pd.concat(frames, ignore_index=True)
