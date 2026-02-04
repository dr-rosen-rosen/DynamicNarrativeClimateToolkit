import numpy as np
import pandas as pd
import torch
from typing import List, Dict, Tuple, Optional, Callable, Union, Literal
from dataclasses import dataclass
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy
from abc import ABC, abstractmethod
from dynclim.features.safety.seg_core import SegmentationStore, sha1, char_offsets_from_sentences, slice_by_offsets, filter_min_length, HAVE_BLING
from dynclim.features.safety.diagnostic_utils import prep_arrays, diagnostics, softmax_weights
if HAVE_BLING:
    from blingfire import text_to_sentences

# ============================================================================
# Configuration classes
# ============================================================================

def _norm(x: Optional[str]) -> str:
    return (x or "").strip().lower().replace("-", "_")

@dataclass
class SegmentationConfig:
    method: str  # 'sentence','clause','paragraph','sliding_window','semantic'
    # General knobs
    min_length: int = 10
    min_unit: Literal["chars","tokens"] = "chars"
    backend: Optional[Literal["blingfire","spacy","auto"]] = "auto"
    # Sliding window
    window_size: Optional[int] = None
    overlap: Optional[int] = None

    def __post_init__(self):
        self.method = _norm(self.method)
        if self.method not in {"sentence","clause","paragraph","sliding_window","semantic"}:
            self.method = "sentence"

        b = _norm(self.backend) or "auto"
        self.backend = b if b in {"blingfire","spacy","auto"} else "auto"

        self.min_unit = "tokens" if self.min_unit == "tokens" else "chars"
        self.min_length = max(int(self.min_length), 1)

        if self.method == "sliding_window":
            default_w = 400 if self.min_unit == "chars" else 128
            default_o = 200 if self.min_unit == "chars" else 64
            self.window_size = int(self.window_size) if self.window_size is not None else default_w
            self.overlap     = int(self.overlap)     if self.overlap     is not None else default_o
            if self.overlap >= self.window_size:
                self.overlap = max(0, self.window_size // 2)
        else:
            # keep defined ints even if unused
            self.window_size = int(self.window_size or 0)
            self.overlap     = int(self.overlap or 0)

@dataclass
class ThresholdConfig:
    method: str  # 'hard','top_k','soft_weight','adaptive','quantile','topp','percent_of_max','zscore'
    threshold: Optional[float] = None      # for 'hard'
    k: Optional[int] = None                # for 'top_k'
    decay_factor: Optional[float] = None   # for 'soft_weight'
    lambda_: Optional[float] = None        # for 'adaptive' & 'diversity_top_k'
    q: Optional[float] = None              # for 'quantile','adaptive'
    p: Optional[float] = None              # for 'topp'
    frac: Optional[float] = None           # for 'percent_of_max'
    zmin: Optional[float] = None           # for 'zscore'
    min_k: Optional[int] = None
    max_k: Optional[int] = None
    clip_quantile: Optional[float] = None  # e.g., cap z-scores by a quantile before std

    def __post_init__(self):
        m = (self.method or "").lower().replace("-", "_")
        # Global sane defaults used by multiple methods
        if self.min_k is None: self.min_k = 1
        if self.max_k is None: self.max_k = 100000  # large cap; tighten if you like
        if self.clip_quantile is None: self.clip_quantile = 0.999

        if m in ("hard", "absolute"):
            if self.threshold is None: self.threshold = 0.5
            self.threshold = float(self.threshold)

        elif m in ("top_k", "topk"):
            if self.k is None: self.k = 3
            self.k = max(int(self.k), self.min_k)
            if self.k > self.max_k: self.k = self.max_k

        elif m == "soft_weight":
            if self.decay_factor is None: self.decay_factor = 0.85
            self.decay_factor = float(self.decay_factor)

        elif m == "adaptive":
            if self.q is None: self.q = 0.75
            if self.lambda_ is None: self.lambda_ = 0.5
            self.q = float(min(max(self.q, 0.0), 1.0))
            self.lambda_ = float(self.lambda_)

        elif m in ("quantile", "q"):
            if self.q is None: self.q = 0.9
            self.q = float(min(max(self.q, 0.0), 1.0))

        elif m in ("topp", "top_p", "top_pmass"):
            if self.p is None: self.p = 0.9
            self.p = float(min(max(self.p, 1e-6), 1.0))

        elif m in ("percent_of_max", "frac_of_max"):
            if self.frac is None: self.frac = 0.8
            self.frac = float(min(max(self.frac, 0.0), 1.0))

        elif m in ("zscore", "zmin"):
            if self.zmin is None: self.zmin = 0.0
            self.zmin = float(self.zmin)

        else:
            # Safe default selector
            self.method = "quantile"
            if self.q is None: self.q = 0.9
            self.q = float(min(max(self.q, 0.0), 1.0))


def _norm_name(x: Optional[str]) -> str:
    if not x: return ""
    return x.strip().lower().replace("-", "_")

@dataclass
class CompositeConfig:
    # methods:
    #   'weighted_average' | 'attention_weight' | 'separate_features'
    #   'topk_attention' | 'power_attention' | 'trimmed_weighted'
    method: str

    # linear mixer (legacy/CPU parity)
    similarity_weight: Optional[float] = 0.5
    sentiment_weight: Optional[float]  = 0.5

    # where to apply the scorer
    apply_over: Optional[str] = "post_threshold"   # "post_threshold" | "all" | "loose_quantile"
    loose_quantile: Optional[float] = 0.6          # used iff apply_over == "loose_quantile"

    # attention/scorer knobs
    temperature: Optional[float] = 0.25            # softmax attention sharpness
    k: Optional[int]       = None                  # for topk_attention
    power: Optional[float] = None                  # for power_attention (e.g., 0.5)
    trim: Optional[float]  = None                  # for trimmed_weighted (e.g., 0.1)
    trim_on: Optional[Literal["sentiment","similarity"]] = "sentiment"
    winsorize: Optional[bool] = False
    attn_quantile: Optional[float] = None          # optional prefilter for attention

    def __post_init__(self):
        # Normalize method and apply_over names
        m = _norm_name(self.method)
        aliases = {
            "attention": "attention_weight",      # allow short alias
            "softmax_attention": "attention_weight",
        }
        self.method = aliases.get(m, m)

        ao = _norm_name(self.apply_over or "post_threshold")
        if ao not in ("post_threshold", "all", "loose_quantile"):
            ao = "post_threshold"
        self.apply_over = ao

        # Fill / clamp numeric defaults
        # linear mixer
        self.similarity_weight = float(self.similarity_weight if self.similarity_weight is not None else 0.5)
        self.sentiment_weight  = float(self.sentiment_weight  if self.sentiment_weight  is not None else 0.5)

        # softmax temperature
        self.temperature = float(self.temperature if self.temperature is not None else 0.25)
        if self.temperature <= 0.0:
            self.temperature = 0.25

        # loose quantile (only used if requested)
        if self.apply_over == "loose_quantile":
            q = self.loose_quantile if self.loose_quantile is not None else 0.6
            q = float(q)
            self.loose_quantile = min(max(q, 1e-6), 1.0)
        else:
            # not used; keep a valid default anyway
            self.loose_quantile = float(self.loose_quantile if self.loose_quantile is not None else 0.6)

        # top-k attention
        if _norm_name(self.method) == "topk_attention":
            self.k = int(self.k) if self.k is not None else 3
            if self.k < 1: self.k = 1
        else:
            self.k = int(self.k) if self.k is not None else 0  # ignored

        # power attention: weights ∝ similarity**power
        if _norm_name(self.method) == "power_attention":
            self.power = float(self.power) if self.power is not None else 1.0
            if self.power <= 0.0: self.power = 1.0
        else:
            self.power = float(self.power) if self.power is not None else 1.0  # ignored

        # trimmed / winsorized mean
        if _norm_name(self.method) == "trimmed_weighted":
            self.trim = float(self.trim) if self.trim is not None else 0.1
            self.trim = min(max(self.trim, 0.0), 0.49)  # keep < 0.5
            self.trim_on = "sentiment" if (self.trim_on or "sentiment") not in ("sentiment","similarity") else self.trim_on
            self.winsorize = bool(self.winsorize)
        else:
            # safe defaults even if unused
            self.trim = float(self.trim) if self.trim is not None else 0.0
            self.winsorize = bool(self.winsorize)
            self.trim_on = "sentiment" if (self.trim_on or "sentiment") not in ("sentiment","similarity") else self.trim_on

        # optional attention prefilter (keeps top (1 - attn_quantile) by similarity before attention)
        if self.attn_quantile is None:
            self.attn_quantile = 0.0
        self.attn_quantile = float(self.attn_quantile)
        self.attn_quantile = min(max(self.attn_quantile, 0.0), 0.99)

        # Final sanity on weights for 'weighted_average'
        if self.method == "weighted_average":
            wsum = self.similarity_weight + self.sentiment_weight
            if wsum == 0.0:
                self.similarity_weight, self.sentiment_weight = 0.5, 0.5
            else:
                self.similarity_weight /= wsum
                self.sentiment_weight  /= wsum

@dataclass
class FeatureResult:
    similarity_scores: List[float]
    sentiment_scores: List[float]
    composite_score: float
    strength_score: float
    selected_segments: List[str]
    segment_similarities: List[float]
    segment_sentiments: List[float]

# Abstract base classes
class Segmenter(ABC):
    @abstractmethod
    def segment(self, text: str) -> List[str]:
        pass

class Thresholder(ABC):
    @abstractmethod
    def select_segments(self, segments: List[str], similarities: List[float]) -> Tuple[List[str], List[float]]:
        """CPU path: Select segments based on similarities"""
        pass
    
    @abstractmethod
    def select_segments_gpu(self, sims_tensor, device):
        """
        GPU path: Create boolean mask for segment selection.
        
        Args:
            sims_tensor: torch.Tensor of shape (S, B) - similarities for S segments, B items
            device: torch device
            
        Returns:
            mask: torch.Tensor of shape (S, B) - boolean mask, or None to select all
        """
        pass

class CompositeScorer(ABC):
    @abstractmethod
    def compute_composite(self, similarities: List[float], sentiments: List[float]) -> Tuple[float, float]:
        """CPU path: Compute composite score from selected segments"""
        pass
    
    @abstractmethod
    def compute_composite_gpu(self, sims_tensor, sentiment_tensor, mask, device):
        """
        GPU path: Compute composite scores for batch of items.
        
        Args:
            sims_tensor: torch.Tensor of shape (S, B)
            sentiment_tensor: torch.Tensor of shape (S,)
            mask: torch.Tensor of shape (S, B) boolean, or None
            device: torch device
            
        Returns:
            scores: torch.Tensor of shape (B,) - composite scores
            strength: torch.Tensor of shape (B,) - strength scores
            strength_type: str - type of strength metric used
            entropy: torch.Tensor of shape (B,) or None - for attention scorers
        """
        pass

# ============================================================================
# Segmenters
# ============================================================================

class SentenceSegmenter:
    def __init__(self, cfg: SegmentationConfig, store=None, tokenizer=None):
        self.cfg = cfg
        self.store = store
        self.min_len = cfg.min_length
        self.min_unit = cfg.min_unit
        self.backend = self._choose_backend(cfg.backend, tokenizer)

        if self.backend == "blingfire":
            import blingfire
            def split_fn(text: str):
                # Blingfire returns '\n' separated sentences
                sents = blingfire.text_to_sentences(text).splitlines()
                return [s for s in sents if self._keep(s)]
            self._split_fn = split_fn

        elif self.backend == "spacy":
            # tokenizer can be a pre-warmed nlp or None (then lazy-load)
            self._nlp = tokenizer
            def split_fn(text: str):
                nlp = self._nlp or self._get_spacy()
                doc = nlp(text)
                return [s.text for s in doc.sents if self._keep(s.text)]
            self._split_fn = split_fn

        else:
            # ultra-simple rule-based fallback
            def split_fn(text: str):
                parts = [p.strip() for p in text.replace("?", ".").replace("!", ".").split(".")]
                return [p for p in parts if self._keep(p)]
            self._split_fn = split_fn

        print(f"[seg] SentenceSegmenter backend={self.backend}, min_length={self.min_len}, unit={self.min_unit}")

    def _choose_backend(self, want: str, tokenizer):
        want = (want or "auto").lower()
        have_bf = False
        try:
            import blingfire  # noqa
            have_bf = True
        except Exception:
            pass

        if want == "blingfire" and have_bf:
            return "blingfire"
        if want in ("auto", "", None) and have_bf:
            return "blingfire"
        # if spaCy is explicitly requested or blingfire unavailable
        return "spacy"

    def _get_spacy(self):
        # lazy-load small English model
        import spacy
        if not hasattr(self, "_nlp") or self._nlp is None:
            try:
                self._nlp = spacy.load("en_core_web_sm", exclude=["ner","tagger","lemmatizer"])
            except Exception:
                # blank pipeline with sentencizer
                self._nlp = spacy.blank("en")
                if "sentencizer" not in self._nlp.pipe_names:
                    self._nlp.add_pipe("sentencizer")
        return self._nlp

    def _keep(self, s: str) -> bool:
        if not s: return False
        if self.min_unit == "chars":
            return len(s) >= self.min_len
        # tokens: naive token count; spaCy path could use len(doc)
        return len(s.split()) >= self.min_len

    def segment(self, text: str):
        return self._split_fn(text or "")

class ClauseSegmenter(Segmenter):
    """ Needs updating like SentenceSegmenter... hodling off for now, because not likely useful (unit too small)"""
    def __init__(self, config: SegmentationConfig, store = None, tokenizer=None):
        self.config = config
        self.store = store or SegmentationStore()
        self.tokenizer = tokenizer
        try:
            self.nlp = spacy.load("en_core_web_sm", exclude=["ner", "lemmatizer", "tagger"])
            self.nlp.enable_pipe("senter")
        except OSError:
            raise ValueError("Clause segmentation requires spaCy model 'en_core_web_sm'")

    def _base_sig(self) -> str:
        # Include spaCy model/version here; bump if your splitting logic changes.
        return "spacy_clause-v1"

    def segment_offsets(self, text: str) -> np.ndarray:
        doc_hash = sha1(text)
        base_sig = self._base_sig()
        cached = self.store.get(doc_hash, "clause", base_sig)
        if cached is not None:
            return cached

        doc = self.nlp(text)
        offsets: list[tuple[int, int]] = []

        for sent in doc.sents:
            # Identify clause breakpoints within the sentence by dependency roles
            clause_breaks = [tok.idx for tok in sent if tok.dep_ in ("cc", "mark")]
            # Always include the sentence end
            cut_points = sorted(set(clause_breaks + [sent.end_char]))
            start = sent.start_char
            for cut in cut_points:
                # produce a clause [start, cut)
                if cut > start:
                    offsets.append((start, cut))
                start = cut

        arr = np.asarray(offsets, dtype=np.int32) if offsets else np.zeros((0, 2), dtype=np.int32)
        self.store.put(doc_hash, "clause", base_sig, arr)
        return arr

    def segment(self, text: str) -> List[str]:
        offsets = self.segment_offsets(text)
        unit = getattr(self.config, "min_unit", "chars")
        min_len = getattr(self.config, "min_length", 0) or 0
        filtered = filter_min_length(text, offsets, min_len=min_len, unit=unit, tokenizer=self.tokenizer)
        return slice_by_offsets(text, filtered)

class ParagraphSegmenter(Segmenter):
    def __init__(self, config: SegmentationConfig):
        self.config = config
    
    def segment(self, text: str) -> List[str]:
        segments = text.split('\n\n')
        segments = [seg.strip() for seg in segments if len(seg.strip()) >= self.config.min_length]
        return segments

class SlidingWindowSegmenter(Segmenter):
    def __init__(self, config: SegmentationConfig):
        self.config = config
        if not config.window_size:
            raise ValueError("Sliding window segmentation requires window_size")
        if not config.overlap:
            self.config.overlap = config.window_size // 2
    
    def segment(self, text: str) -> List[str]:
        words = text.split()
        segments = []
        
        for i in range(0, len(words) - self.config.window_size + 1, 
                      self.config.window_size - self.config.overlap):
            segment = ' '.join(words[i:i + self.config.window_size])
            if len(segment) >= self.config.min_length:
                segments.append(segment)
        
        return segments

class SemanticSegmenter(Segmenter):
    def __init__(self, config: SegmentationConfig, model_name: str = 'all-MiniLM-L6-v2'):
        self.config = config
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = 0.7  # Similarity threshold for grouping sentences
    
    def segment(self, text: str) -> List[str]:
        # First get sentences
        sentence_segmenter = SentenceSegmenter(self.config)
        sentences = sentence_segmenter.segment(text)
        
        if len(sentences) <= 1:
            return sentences
        
        # Get embeddings for sentences
        embeddings = self.model.encode(sentences)
        
        # Group semantically similar sentences
        segments = []
        used = set()
        
        for i, sentence in enumerate(sentences):
            if i in used:
                continue
            
            # Find similar sentences
            similarities = cosine_similarity([embeddings[i]], embeddings[i+1:])[0]
            similar_indices = [j + i + 1 for j, sim in enumerate(similarities) 
                             if sim > self.similarity_threshold and (j + i + 1) not in used]
            
            # Create segment with current sentence and similar ones
            segment_sentences = [sentence]
            used.add(i)
            
            for idx in similar_indices:
                segment_sentences.append(sentences[idx])
                used.add(idx)
            
            segment = ' '.join(segment_sentences)
            if len(segment) >= self.config.min_length:
                segments.append(segment)
        
        return segments

# ============================================================================
# ???
# ============================================================================

def _prep_sims(similarities):
    s = np.asarray(similarities, float)
    s = np.nan_to_num(s, nan=0.0, posinf=1.0, neginf=0.0)
    # sims already normalized to [0,1] upstream; keep the clip anyway
    return np.clip(s, 0.0, 1.0)

def _stable_topk_indices(s, k):
    n = s.size
    if k >= n:
        idx = np.arange(n)
        return idx[np.argsort(-s, kind="mergesort")]
    part = np.argpartition(-s, k-1)[:k]
    return part[np.argsort(-s[part], kind="mergesort")]

def _ensure_min_k(idx, n, min_k=1):
    if idx.size >= min_k or n == 0:
        return idx
    # deterministically add the next best by similarity
    remaining = np.setdiff1d(np.arange(n), idx, assume_unique=False)
    add = remaining[np.argsort(-remaining, kind="mergesort")]  # stable fallback
    need = min(min_k - idx.size, remaining.size)
    return np.concatenate([idx, remaining[:need]])

# ============================================================================
# Thresholders
# ============================================================================

class HardThresholder(Thresholder):
    def __init__(self, config: ThresholdConfig):
        self.threshold = float(config.threshold if config.threshold is not None else 0.5)
        self.min_k = int(getattr(config, "min_k", 1))       # optional; not in dataclass
        self.max_k = getattr(config, "max_k", None)         # optional

    def select_segments(self, segments, similarities):
        s = _prep_sims(similarities)
        keep = np.where(s >= self.threshold)[0]
        if self.max_k is not None and keep.size > self.max_k:
            keep = _stable_topk_indices(s, int(self.max_k))
        keep = _ensure_min_k(keep, s.size, self.min_k)
        return [segments[i] for i in keep], s[keep].tolist()
    
    @torch.inference_mode()
    def select_segments_gpu(self, sims_tensor, device):
        """GPU implementation of hard threshold"""
        mask = sims_tensor >= self.threshold
        return mask

class TopKThresholder(Thresholder):
    def __init__(self, config: ThresholdConfig):
        self.k = getattr(config, "k", 3)                   # int; float in (0,1] treated as proportion
        self.min_k = int(getattr(config, "min_k", 1))

    def _resolve_k(self, n):
        k = self.k
        if isinstance(k, float):
            k = int(np.ceil(max(0.0, min(1.0, k)) * n))
        return int(max(1, min(n, k)))

    def select_segments(self, segments, similarities):
        s = _prep_sims(similarities)
        k = max(self._resolve_k(s.size), self.min_k)
        idx = _stable_topk_indices(s, k)
        return [segments[i] for i in idx], s[idx].tolist()
    
    @torch.inference_mode()
    def select_segments_gpu(self, sims_tensor, device):
        """GPU implementation of top-k selection"""
        S, B = sims_tensor.shape
        k = min(self.k, S)
        
        # Get top-k indices for each item
        _, idx = torch.topk(sims_tensor.T, k=k, dim=1, largest=True, sorted=False)  # (B, k)
        
        # Create mask
        mask = torch.zeros_like(sims_tensor, dtype=torch.bool)
        mask.scatter_(0, idx.T, True)
        return mask

class SoftWeightThresholder(Thresholder):
    """Rescales similarities; does not drop segments. Use when you want a cheap, monotone emphasis."""

    """Don't use this. it doesn't really make sense. leaving for now but will excise eventually."""

    def __init__(self, config: ThresholdConfig):
        self.decay = float(config.decay_factor if config.decay_factor is not None else 0.5)

    def select_segments(self, segments, similarities):
        s = _prep_sims(similarities)
        if s.size == 0:
            return [], []
        m = float(s.max())
        w = np.exp(-self.decay * (m - s))
        s_new = (s * w).tolist()
        return segments, s_new
    
    @torch.inference_mode()
    def select_segments_gpu(self, sims_tensor, device):
        """GPU implementation of soft weight (selects all)"""
        # Soft weight doesn't filter, returns None to select all
        return None

class AdaptiveThresholder(Thresholder):
    def __init__(self, config: ThresholdConfig):
        self.lambda_ = getattr(config, "lambda_", 0.5)  # Renamed from lam
        self.min_k = getattr(config, "min_k", 1)
        self.max_k = getattr(config, "max_k", None)
        self.clip_quantile = getattr(config, "clip_quantile", 0.95)
        self.last_diagnostics = {}

    def select_segments(self, segments, similarities):
        s = _prep_sims(similarities)
        n = s.size
        if n == 0:
            return [], []

        med = np.median(s)
        mad = np.median(np.abs(s - med)) + 1e-8
        thr = med + self.lambda_ * 1.4826 * mad

        # cap threshold to avoid filtering everything when the tail is skinny
        hi = np.quantile(s, self.clip_quantile) if n > 4 else s.max()
        thr = min(thr, hi)

        keep = np.where(s >= thr)[0]
        keep = _ensure_min_k(keep, n, self.min_k)

        if self.max_k is not None and keep.size > self.max_k:
            keep = _stable_topk_indices(s, int(self.max_k))

        return [segments[i] for i in keep], s[keep].tolist()
    
    @torch.inference_mode()
    def select_segments_gpu(self, sims_tensor, device):
        """GPU implementation of adaptive threshold (matches CPU logic)"""
        
        S, B = sims_tensor.shape
        
        # Compute median for each item
        sorted_sims, _ = torch.sort(sims_tensor, dim=0)
        med_idx = S // 2
        med = sorted_sims[med_idx:med_idx+1, :]  # (1, B)
        
        # Compute MAD (Median Absolute Deviation)
        abs_dev = torch.abs(sims_tensor - med)
        sorted_abs_dev, _ = torch.sort(abs_dev, dim=0)
        mad = sorted_abs_dev[med_idx:med_idx+1, :] + 1e-8
        
        # Adaptive threshold: median + lambda * 1.4826 * MAD
        thr = med + self.lambda_ * 1.4826 * mad
        
        # Cap threshold to clip_quantile (safety)
        if self.clip_quantile is not None and S > 4:
            clip_idx = int(torch.floor(torch.tensor(self.clip_quantile * S)).item())
            clip_idx = min(max(clip_idx, 0), S - 1)
            hi = sorted_sims[clip_idx:clip_idx+1, :]
            thr = torch.minimum(thr, hi)
        
        # Create mask
        mask = sims_tensor >= thr
        
        # Note: min_k and max_k constraints not implemented on GPU
        # (would require per-item loops, defeating parallelization)
        
        return mask

class QuantileThresholder(Thresholder):
    def __init__(self, config: ThresholdConfig):
        self.q = float(getattr(config, "q", 0.7))          # keep top (1-q)
        self.min_k = int(getattr(config, "min_k", 1))
        self.max_k = getattr(config, "max_k", None)

    def select_segments(self, segments, similarities):
        s = _prep_sims(similarities)
        if s.size == 0:
            return [], []
        thr = np.quantile(s, self.q)
        keep = np.where(s >= thr)[0]
        keep = _ensure_min_k(keep, s.size, self.min_k)
        if self.max_k is not None and keep.size > self.max_k:
            keep = _stable_topk_indices(s, int(self.max_k))
        return [segments[i] for i in keep], s[keep].tolist()
    
    @torch.inference_mode()
    def select_segments_gpu(self, sims_tensor, device):
        """GPU implementation of quantile threshold"""
        
        # Compute quantile threshold for each item
        # failing with mps limitation
        # qv = torch.quantile(sims_tensor.T, self.q, dim=1).unsqueeze(0)  # (1, B)
        # MPS-compatible: use sorting instead of quantile
        S, B = sims_tensor.shape
        sorted_sims, _ = torch.sort(sims_tensor, dim=0)  # Sort each column
        idx = int(torch.floor(torch.tensor(self.q * S)).item())
        idx = min(max(idx, 0), S - 1)
        qv = sorted_sims[idx:idx+1, :]  # Pick the q-th value
        mask = sims_tensor >= qv
        
        return mask
    
class DiversityTopKThresholder(Thresholder):
    """
    Maximal Marginal Relevance using only similarity as a cheap redundancy proxy.
    If you can supply segment–segment cosine, replace the redundancy update accordingly.
    """
    def __init__(self, config: ThresholdConfig):
        self.k = int(getattr(config, "k", 5))
        self.lambda_ = float(getattr(config, "lambda_", 0.7))  # 1.0=relevance-only; 0.0=novelty-only

    def select_segments(self, segments, similarities):
        s = _prep_sims(similarities)
        n = s.size
        if n == 0:
            return [], []
        k = max(1, min(self.k, n))
        chosen = []
        remaining = list(range(n))
        # simple redundancy proxy; if you have seg-seg sims, use max cosine to chosen
        redundancy = np.zeros(n, dtype=float)

        for _ in range(k):
            if not remaining: break
            # MMR: lambda*relevance - (1-lambda)*redundancy
            mmr_scores = self.lambda_ * s[remaining] - (1 - self.lambda_) * redundancy[remaining]
            j = remaining[int(np.argmax(mmr_scores))]
            chosen.append(j)
            remaining.remove(j)
            redundancy = np.maximum(redundancy, s * s[j])  # cheap proxy

        chosen = np.array(chosen, dtype=int)
        # output in stable, descending similarity order
        chosen = chosen[np.argsort(-s[chosen], kind="mergesort")]
        return [segments[i] for i in chosen], s[chosen].tolist()

class TopPThresholder(Thresholder):
    """Select top segments until cumulative similarity reaches p fraction of total"""
    
    def __init__(self, config: ThresholdConfig):
        self.p = config.p if hasattr(config, 'p') else 0.9
        self.last_diagnostics = {}
    
    def select_segments(self, segments: List[str], similarities: List[float]) -> Tuple[List[str], List[float]]:
        """CPU implementation of top-p selection"""
        if not segments:
            return [], []
        
        # Sort by similarity descending
        sorted_pairs = sorted(zip(similarities, segments), reverse=True)
        
        # Compute cumulative sum
        total = sum(similarities)
        cumsum = 0
        selected = []
        selected_sims = []
        
        for sim, seg in sorted_pairs:
            cumsum += sim
            selected.append(seg)
            selected_sims.append(sim)
            if cumsum / total >= self.p:
                break
        
        # Ensure at least one
        if not selected and segments:
            selected = [segments[0]]
            selected_sims = [similarities[0]]
        
        self.last_diagnostics = {"n": len(segments), "n_selected": len(selected)}
        return selected, selected_sims
    
    @torch.inference_mode()
    def select_segments_gpu(self, sims_tensor, device):
        """GPU implementation of top-p selection"""
        
        S, B = sims_tensor.shape
        
        # Sort descending
        vals, idx = torch.sort(sims_tensor.T, dim=1, descending=True)  # (B, S)
        
        # Cumulative sum
        cum = vals.cumsum(dim=1)
        totals = cum[:, -1].unsqueeze(1)
        totals = torch.clamp(totals, min=1e-8)
        
        # Find where cumsum exceeds p * total
        need = (cum / totals) <= self.p
        need[:, 0] = True  # Always include top item
        
        # Create mask
        mask = torch.zeros_like(sims_tensor, dtype=torch.bool)
        for b in range(B):
            need_b = need[b]
            if need_b.any():
                sel_indices = idx[b, need_b]
                mask[sel_indices, b] = True
        
        return mask

class PercentOfMaxThresholder(Thresholder):
    """Select segments with similarity >= frac * max_similarity"""
    
    def __init__(self, config: ThresholdConfig):
        self.frac = config.frac if hasattr(config, 'frac') else 0.8
        self.last_diagnostics = {}
    
    def select_segments(self, segments: List[str], similarities: List[float]) -> Tuple[List[str], List[float]]:
        """CPU implementation"""
        if not segments:
            return [], []
        
        max_sim = max(similarities)
        thresh = self.frac * max_sim
        
        selected = []
        selected_sims = []
        for seg, sim in zip(segments, similarities):
            if sim >= thresh:
                selected.append(seg)
                selected_sims.append(sim)
        
        # Ensure at least one
        if not selected:
            max_idx = similarities.index(max_sim)
            selected = [segments[max_idx]]
            selected_sims = [max_sim]
        
        self.last_diagnostics = {"n": len(segments), "n_selected": len(selected)}
        return selected, selected_sims
    
    @torch.inference_mode()
    def select_segments_gpu(self, sims_tensor, device):
        """GPU implementation"""
        
        vmax = sims_tensor.max(dim=0).values.unsqueeze(0)  # (1, B)
        mask = sims_tensor >= (self.frac * vmax)
        
        return mask

class ZScoreThresholder(Thresholder):
    """Select segments with z-score >= zmin"""
    
    def __init__(self, config: ThresholdConfig):
        self.zmin = config.zmin if hasattr(config, 'zmin') else 1.0
        self.last_diagnostics = {}
    
    def select_segments(self, segments: List[str], similarities: List[float]) -> Tuple[List[str], List[float]]:
        """CPU implementation"""
        if not segments:
            return [], []
        
        sims = np.array(similarities)
        mu = sims.mean()
        sd = sims.std()
        
        if sd < 1e-8:
            # All similar - select all
            return segments, similarities
        
        z = (sims - mu) / sd
        
        selected = []
        selected_sims = []
        for seg, sim, z_score in zip(segments, similarities, z):
            if z_score >= self.zmin:
                selected.append(seg)
                selected_sims.append(sim)
        
        # Ensure at least one
        if not selected:
            max_idx = int(z.argmax())
            selected = [segments[max_idx]]
            selected_sims = [similarities[max_idx]]
        
        self.last_diagnostics = {"n": len(segments), "n_selected": len(selected)}
        return selected, selected_sims
    
    @torch.inference_mode()
    def select_segments_gpu(self, sims_tensor, device):
        """GPU implementation"""
        
        mu = sims_tensor.mean(dim=0, keepdim=True)
        sd = sims_tensor.std(dim=0, unbiased=False, keepdim=True) + 1e-8
        z = (sims_tensor - mu) / sd
        mask = z >= self.zmin
        
        return mask

# ============================================================================
# Composite scoring implementations
# ============================================================================

class WeightedAverageScorer(CompositeScorer):
    def __init__(self, config: CompositeConfig):
        # kept for backward compatibility; not used here
        self.similarity_weight = getattr(config, "similarity_weight", 0.5)
        self.sentiment_weight  = getattr(config, "sentiment_weight", 0.5)
        self.last_diagnostics = {}

    def compute_composite(self, similarities, sentiments):
        s, m = prep_arrays(similarities, sentiments)
        n = int(s.size)
        if n == 0:
            self.last_diagnostics = {
                "ess": 0.0, "breadth_norm": 0.0, "weight_entropy": 0.0,
                "max_weight": 0.0, "var_sent": 0.0, "n": 0,
                "strength_sim_mean": 0.0, "strength_type": "uniform_mean",
            }
            return 0.0, 0.0

        # uniform weights over selected segments
        w = np.full(n, 1.0 / n, dtype=float)

        sent_signal     = float((w * m).sum())     # == m.mean()
        strength_signal = float((w * s).sum())     # == s.mean()

        # diagnostics under uniform w
        w_safe = np.clip(w, 1e-12, 1.0)
        entropy = float(-(w_safe * np.log(w_safe)).sum())    # = log(n)
        ess = float(1.0 / np.sum(w**2))                      # = n
        breadth_norm = float((ess - 1.0) / max(n - 1, 1))    # = 1 if n>1
        var_sent = float((w * (m - sent_signal) ** 2).sum())

        self.last_diagnostics = {
            "ess": ess, "breadth_norm": breadth_norm, "weight_entropy": entropy,
            "max_weight": float(w.max()), "var_sent": var_sent, "n": n,
            "strength_sim_mean": float(s.mean()),
            "strength_type": "uniform_mean",
        }
        return sent_signal, strength_signal
    
    @torch.inference_mode()
    def compute_composite_gpu(self, sims_tensor, sentiment_tensor, mask, device):
        """
        Uniform weighting across selected segments.
        sims_tensor: (S,B) in [0,1]; sentiment_tensor: (S,)
        mask: (S,B) bool or None; True = keep.
        Returns: sent_signal (B,), strength_signal (B,)
        """
        import torch
        S, B = sims_tensor.shape
        s = sims_tensor
        m = sentiment_tensor.view(S, 1).expand(S, B)

        if mask is None:
            n = torch.full((B,), S, device=device, dtype=torch.float32)
            sent_signal     = m.mean(dim=0)           # (B,)
            strength_signal = s.mean(dim=0)           # (B,)
        else:
            keep = mask
            n = keep.sum(dim=0).clamp(min=1).to(torch.float32)  # (B,)
            # means over selected rows per column
            sent_signal = (m * keep).sum(dim=0) / n
            strength_signal = (s * keep).sum(dim=0) / n

            # handle columns with no selected rows
            zero_cols = (keep.sum(dim=0) == 0)
            if zero_cols.any():
                sent_signal[zero_cols] = 0.0
                strength_signal[zero_cols] = 0.0

        # You can compute diagnostics here if you need parity with CPU (uniform w)
        return sent_signal, strength_signal

class AttentionWeightScorer(CompositeScorer):
    def __init__(self, config: CompositeConfig):
        self.temperature = getattr(config, "temperature", 0.25)
        self.last_diagnostics = {}

    def compute_composite(self, similarities, sentiments):
        """
        Inputs
          similarities: iterable of s in [0,1]
          sentiments:   iterable of m in [-1,1]
        Returns
          sent_signal:        float in [-1,1]
          strength_signal:    float in [0,1]  (attention-weighted similarity = 'relevance')
        Diagnostics (self.last_diagnostics)
          ess, breadth_norm, weight_entropy, max_weight, var_sent, n, strength_sim_mean
        """
        s, m = prep_arrays(similarities, sentiments)  # np.ndarrays
        n = int(s.size)

        if n == 0:
            self.last_diagnostics = {
                "ess": 0.0,
                "breadth_norm": 0.0,
                "weight_entropy": 0.0,
                "max_weight": 0.0,
                "var_sent": 0.0,
                "n": 0,
                "strength_sim_mean": 0.0,
                "strength_type": "weighted_similarity",
            }
            return 0.0, 0.0

        # Attention weights over selected segments
        # stable softmax with temperature
        x = (s / max(self.temperature, 1e-6)).astype(np.float64)
        x = x - x.max()  # stability
        w = np.exp(x)
        Z = w.sum()
        if not np.isfinite(Z) or Z <= 0:
            # degenerate: fallback to uniform over n
            w = np.full_like(w, 1.0 / n)
        else:
            w = w / Z

        # Sentiment signal ([-1,1]), focused by attention
        sent_signal = float((w * m).sum())

        # Strength = attention-weighted mean similarity ([0,1])
        strength_signal = float((w * s).sum())

        # Diagnostics
        # Entropy of attention
        w_safe = np.clip(w, 1e-12, 1.0)
        entropy = float(-(w_safe * np.log(w_safe)).sum())  # >= 0

        # Effective sample size and normalized breadth
        ess = float(1.0 / np.sum(w**2))
        breadth_norm = float((ess - 1.0) / (max(n - 1, 1)))  # [0,1]

        # Dispersion of weighted sentiments
        var_sent = float((w * (m - sent_signal) ** 2).sum())

        self.last_diagnostics = {
            "ess": ess,
            "breadth_norm": breadth_norm,
            "weight_entropy": entropy,
            "max_weight": float(w.max()),
            "var_sent": var_sent,
            "n": n,
            "strength_sim_mean": float(s.mean()),  # simple mean (legacy reference)
            "strength_type": "weighted_similarity",
        }

        return sent_signal, strength_signal
    @torch.inference_mode()
    def compute_composite_gpu(self, sims_tensor, sentiment_tensor, mask, device):
        """
        GPU implementation
        sims_tensor:      (S, B)   similarities in [0,1]
        sentiment_tensor: (S,)     sentiments in [-1,1]
        mask:             (S, B) bool or None  (True = keep)
        Returns:
          sent_signal:     (B,) tensor in [-1,1]
          strength_signal: (B,) tensor in [0,1]
        Diagnostics in self.last_diagnostics are per-batch tensors (length B).
        """
        S, B = sims_tensor.shape
        s = sims_tensor
        m = sentiment_tensor.view(S, 1).expand(S, B)  # broadcast to (S,B)

        # Mask handling
        if mask is not None:
            valid = mask
        else:
            valid = torch.ones_like(s, dtype=torch.bool, device=s.device)

        # If a column has no valid rows, treat it gracefully
        col_has_any = valid.any(dim=0)  # (B,)

        # Build attention logits with temperature and max-shift (only on valid entries)
        temp = max(self.temperature, 1e-6)
        x = s / temp

        # set invalid to -inf so softmax ignores them
        x_masked = x.masked_fill(~valid, float("-inf"))

        # subtract per-column max (over valid rows) for stability
        col_max = torch.where(
            col_has_any,
            x_masked.max(dim=0).values,
            torch.zeros(B, device=s.device),
        )
        x_stable = x_masked - col_max  # (S,B)

        # softmax over rows
        w = torch.softmax(x_stable, dim=0)  # (S,B)
        # for all-invalid columns, softmax returns NaN; fix by setting weights uniform 0
        w = torch.where(
            torch.isnan(w),
            torch.zeros_like(w),
            w
        )

        # Weighted sentiment (B,)
        sent_signal = (w * m).sum(dim=0)

        # Strength = attention-weighted mean similarity (B,)
        strength_signal = (w * s).sum(dim=0)

        # Diagnostics (per batch)
        w_safe = w.clamp_min(1e-12)
        entropy = -(w_safe * w_safe.log()).sum(dim=0)  # (B,)

        ess = 1.0 / (w.pow(2).sum(dim=0).clamp_min(1e-12))  # (B,)
        n_selected = valid.sum(dim=0).to(s.dtype)  # (B,)
        breadth_norm = torch.where(
            n_selected > 1,
            (ess - 1.0) / (n_selected - 1.0),
            torch.zeros_like(ess),
        )

        # weighted variance of sentiments around sent_signal
        var_sent = (w * (m - sent_signal.unsqueeze(0))**2).sum(dim=0)

        # fix outputs for all-invalid columns
        zero_cols = ~col_has_any
        if zero_cols.any():
            sent_signal[zero_cols] = 0.0
            strength_signal[zero_cols] = 0.0
            entropy[zero_cols] = 0.0
            ess[zero_cols] = 0.0
            breadth_norm[zero_cols] = 0.0
            var_sent[zero_cols] = 0.0

        # store diagnostics as CPU tensors for easy logging/JSON
        self.last_diagnostics = {
            "ess": ess.detach().cpu(),
            "breadth_norm": breadth_norm.detach().cpu(),
            "weight_entropy": entropy.detach().cpu(),
            "max_weight": w.max(dim=0).values.detach().cpu(),
            "var_sent": var_sent.detach().cpu(),
            "n": n_selected.detach().cpu(),
            "strength_sim_mean": s.mean(dim=0).detach().cpu(),
            "strength_type": "weighted_similarity",
        }

        return sent_signal, strength_signal
    
# class TopKAttentionScorer(CompositeScorer):
#     def __init__(self, cfg: CompositeConfig):
#         self.k = getattr(cfg, "k", 5)                  # int or float in (0,1]
#         self.tau = getattr(cfg, "temperature", 0.25)   # used if use_softmax=True
#         self.use_softmax = getattr(cfg, "use_softmax", True)
#         self.last_diagnostics = {}

#     def _resolve_k(self, n: int) -> int:
#         # allow proportion
#         if isinstance(self.k, float):
#             k = max(1, int(np.ceil(self.k * n)))
#         else:
#             k = int(self.k)
#         return min(max(k, 1), n)

#     def compute_composite(self, similarities, sentiments):
#         s, m = prep_arrays(similarities, sentiments)
#         n = s.size
#         if n == 0:
#             self.last_diagnostics = {
#                 "ess":0,"weight_entropy":0,"n":0,"max_weight":0,
#                 "strength_sim_mean": 0.0,
#                 "strength_entropy": None,
#                 "strength_type": "neg_entropy"
#             }
#             return 0.0, 0.0

#         k = self._resolve_k(n)
#         # Stable top-k by similarity (highest first)
#         # argpartition is O(n), then sort the top-k for stability
#         if k < n:
#             idx_k = np.argpartition(-s, k-1)[:k]
#             # sort those K by similarity desc to make weights deterministic
#             idx_k = idx_k[np.argsort(-s[idx_k], kind="mergesort")]
#             s_k = s[idx_k]
#             m_k = m[idx_k]
#         else:
#             idx_k = np.arange(n)
#             s_k, m_k = s, m

#         # Weights on the K set
#         if self.use_softmax:
#             w = softmax_weights(s_k, tau=self.tau)
#         else:
#             wsum = s_k.sum() + 1e-8
#             w = s_k / wsum

#         composite = float((w * m_k).sum())
#         sim_mean = float(s.mean())
#         entropy = -float((w * np.log(w + 1e-8)).sum())

#         # Diagnostics on the effective K set
#         self.last_diagnostics = diagnostics(w, m_k)
#         self.last_diagnostics.update({
#             "k_used": int(k),
#             "n_total": int(n),
#             "strength_sim_mean": sim_mean,
#             "strength_entropy": entropy,
#             "strength_type": "neg_entropy",  
#         })
#         return composite, entropy
    
#     @torch.inference_mode()
#     def compute_composite_gpu(self, sims_tensor, sentiment_tensor, mask, device):
#         """
#         GPU implementation of top-k attention scorer.
        
#         Takes top-k segments by similarity, then computes attention-weighted sentiment.
#         """
        
#         S, B = sims_tensor.shape
#         sent = sentiment_tensor  # (S,)
        
#         # Apply mask if provided (from thresholding)
#         if mask is not None:
#             sims_work = sims_tensor.clone()
#             sims_work[~mask] = -1e9  # Mask out non-selected
#         else:
#             sims_work = sims_tensor
        
#         # Resolve k (handle float proportion)
#         if isinstance(self.k, float):
#             k = max(1, int(torch.ceil(torch.tensor(self.k * S))))
#         else:
#             k = int(self.k)
#         k = min(k, S)
        
#         # Get top-k by similarity for each item
#         topk_vals, topk_idx = torch.topk(sims_work, k=k, dim=0, largest=True)  # (k, B)
        
#         # Gather corresponding similarities and sentiments
#         # topk_idx has shape (k, B), we need to gather from sims_tensor
#         # Create expanded indices for gathering
#         batch_idx = torch.arange(B, device=device).unsqueeze(0).expand(k, -1)  # (k, B)
        
#         # Gather top-k similarities (already have topk_vals)
#         s_k = topk_vals  # (k, B)
        
#         # Gather sentiments at top-k positions
#         # this is failing because mps support for gather
#         # sent_expanded = sent.unsqueeze(1).expand(-1, B)  # (S, B)
#         # m_k = torch.gather(sent_expanded, 0, topk_idx)  # (k, B)
#         # NEW (MPS-compatible):
#         m_k = torch.zeros_like(topk_vals, device=device)
#         for b in range(B):
#             m_k[:, b] = sent[topk_idx[:, b]]
#         # Compute weights on the k set
#         if self.use_softmax:
#             # Softmax attention
#             att = s_k / max(self.tau, 1e-6)
#             w = torch.softmax(att, dim=0)  # (k, B)
#         else:
#             # Normalized similarity weights
#             w_sum = s_k.sum(dim=0, keepdim=True) + 1e-8  # (1, B)
#             w = s_k / w_sum  # (k, B)
        
#         # Weighted sentiment
#         scores = (w * m_k).sum(dim=0)  # (B,)
        
#         # Strength = mean similarity over all segments (not just top-k)
#         if mask is None:
#             strength = sims_tensor.mean(dim=0)
#         else:
#             cnt = mask.sum(dim=0).float().clamp(min=1)
#             strength = (sims_tensor * mask).sum(dim=0) / cnt
        
#         # Compute entropy for diagnostics
#         entropy = -(w.clamp_min(1e-8) * w.clamp_min(1e-8).log()).sum(dim=0)
        
#         strength_type = "mean_sim"
        
#         return scores, strength, strength_type, entropy

class PowerAttentionScorer(CompositeScorer):
    """
    Power-attention weights: w_i ∝ (sim_i)^p over the (optionally) filtered set.
    Strength = attention-weighted similarity (relevance) in [0,1]
    Sentiment = attention-weighted sentiment in [-1,1]
    Diagnostics: ess, breadth_norm, weight_entropy, max_weight, var_sent, n_selected, power, attn_quantile
    """

    def __init__(self, cfg: CompositeConfig):
        self.p = float(getattr(cfg, "power", 0.5))                 # 0.5 flattens, 1.0 identity, >1 hardens
        self.attn_quantile = getattr(cfg, "attn_quantile", None)   # e.g., 0.7 keeps top 30% by sim
        self.last_diagnostics = {}

    def compute_composite(self, similarities, sentiments):
        s, m = prep_arrays(similarities, sentiments)  # np arrays
        n = int(s.size)

        def _empty_diag():
            return {
                "ess": 0.0,
                "breadth_norm": 0.0,
                "weight_entropy": 0.0,
                "max_weight": 0.0,
                "var_sent": 0.0,
                "n": 0,
                "power": float(self.p),
                "attn_quantile": self.attn_quantile,
                "strength_sim_mean": 0.0,
                "strength_type": "weighted_similarity",
            }

        if n == 0:
            self.last_diagnostics = _empty_diag()
            return 0.0, 0.0

        # Optional similarity prefilter (by quantile of s)
        if self.attn_quantile is not None:
            q = float(self.attn_quantile)
            finite = np.isfinite(s)
            if finite.any():
                thr = np.quantile(s[finite], q)
                keep = (s >= thr) & finite
                s = s[keep]; m = m[keep]
                n = int(s.size)
                if n == 0:
                    self.last_diagnostics = _empty_diag()
                    return 0.0, 0.0
            else:
                self.last_diagnostics = _empty_diag()
                return 0.0, 0.0

        # Power-law attention weights: w ∝ s^p (clip to [0,1] for safety)
        sims = np.clip(s, 0.0, 1.0).astype(np.float64)
        p = max(self.p, 1e-8)
        w = np.power(sims, p)
        Z = float(w.sum())

        if not np.isfinite(Z) or Z <= 0.0:
            # If everything zero after filtering, produce empty-like outputs
            self.last_diagnostics = _empty_diag()
            return 0.0, 0.0

        w = w / Z

        # Sentiment ([-1,1]) and Strength ([0,1])
        sent_signal = float((w * m).sum())
        strength_signal = float((w * sims).sum())

        # Diagnostics
        w_safe = np.clip(w, 1e-12, 1.0)
        entropy = float(-(w_safe * np.log(w_safe)).sum())
        ess = float(1.0 / np.sum(w**2))
        breadth_norm = float((ess - 1.0) / max(n - 1, 1))  # [0,1]
        var_sent = float((w * (m - sent_signal) ** 2).sum())

        self.last_diagnostics = {
            "ess": ess,
            "breadth_norm": breadth_norm,
            "weight_entropy": entropy,
            "max_weight": float(w.max()),
            "var_sent": var_sent,
            "n": n,
            "power": float(self.p),
            "attn_quantile": self.attn_quantile,
            "strength_sim_mean": float(sims.mean()),
            "strength_type": "weighted_similarity",
        }

        return sent_signal, strength_signal

    @torch.inference_mode()
    def compute_composite_gpu(self, sims_tensor, sentiment_tensor, mask, device):
        """
        GPU vectorized implementation.
        sims_tensor:      (S, B) similarities in [0,1]
        sentiment_tensor: (S,)   sentiments in [-1,1]
        mask:             (S, B) bool (True=keep) or None
        Returns: sent_signal (B,), strength_signal (B,)
        Diagnostics are stored in self.last_diagnostics as CPU tensors/vectors length B.
        """
        import torch

        S, B = sims_tensor.shape
        s = sims_tensor
        m = sentiment_tensor.view(S, 1).expand(S, B)

        # Base validity (mask) — if none, all valid
        if mask is not None:
            valid = mask
        else:
            valid = torch.ones_like(s, dtype=torch.bool, device=s.device)

        # Optional quantile prefilter per column
        if self.attn_quantile is not None:
            q = float(self.attn_quantile)
            # torch.quantile operates over valid entries by masking; do per column
            # Build a per-column threshold using masked quantile
            # Fallback: if no valid in a column, mark column as all-invalid.
            s_masked = s.masked_fill(~valid, float("nan"))  # NaNs ignored by nanquantile
            # Compute column-wise quantile (keepdim for broadcasting)
            # torch.nanquantile available in newer versions; if not, emulate via sort/dropna.
            try:
                thr = torch.nanquantile(s_masked, q, dim=0, keepdim=True)
            except Exception:
                # Fallback emulation
                sorted_s, _ = torch.sort(s_masked, dim=0)
                # Count non-NaNs per column
                counts = torch.sum(torch.isfinite(sorted_s), dim=0)
                # Index = floor(q * (count-1))
                idx = torch.clamp((q * (counts - 1).float()).floor().long(), min=0)
                # Gather threshold; when count==0, set thr to +inf so nothing passes
                arangeB = torch.arange(B, device=s.device)
                thr_vals = torch.full((B,), float("inf"), device=s.device)
                nonzero = counts > 0
                thr_vals[nonzero] = sorted_s[idx[nonzero], arangeB[nonzero]]
                thr = thr_vals.view(1, B)

            attn_mask = valid & torch.isfinite(s) & (s >= thr)
        else:
            attn_mask = valid & torch.isfinite(s)

        col_has_any = attn_mask.any(dim=0)  # (B,)

        # Power weights on the filtered similarities
        sims_clipped = torch.clamp(s, 0.0, 1.0)
        p = max(self.p, 1e-8)
        w_raw = torch.where(attn_mask, torch.pow(sims_clipped, p), torch.zeros_like(sims_clipped))

        # Normalize per column; guard zero columns
        w_sum = w_raw.sum(dim=0, keepdim=True)  # (1,B)
        # Safe division
        w = torch.where(w_sum > 0, w_raw / w_sum, torch.zeros_like(w_raw))

        # Sentiment and Strength
        sent_signal = (w * m).sum(dim=0)                   # (B,)
        strength_signal = (w * sims_clipped).sum(dim=0)    # (B,)

        # Diagnostics
        w_safe = w.clamp_min(1e-12)
        entropy = -(w_safe * w_safe.log()).sum(dim=0)                  # (B,)
        ess = 1.0 / (w.pow(2).sum(dim=0).clamp_min(1e-12))             # (B,)
        n_selected = attn_mask.sum(dim=0).to(s.dtype)                   # (B,)
        breadth_norm = torch.where(
            n_selected > 1, (ess - 1.0) / (n_selected - 1.0), torch.zeros_like(ess)
        )
        var_sent = (w * (m - sent_signal.unsqueeze(0))**2).sum(dim=0)   # (B,)

        # Zero-out degenerate columns (no selected segments)
        if (~col_has_any).any():
            z = ~col_has_any
            sent_signal[z] = 0.0
            strength_signal[z] = 0.0
            entropy[z] = 0.0
            ess[z] = 0.0
            breadth_norm[z] = 0.0
            var_sent[z] = 0.0

        # Store diagnostics as CPU tensors for easy serialization
        self.last_diagnostics = {
            "ess": ess.detach().cpu(),
            "breadth_norm": breadth_norm.detach().cpu(),
            "weight_entropy": entropy.detach().cpu(),
            "max_weight": w.max(dim=0).values.detach().cpu(),
            "var_sent": var_sent.detach().cpu(),
            "n": n_selected.detach().cpu(),
            "power": torch.full((B,), float(self.p)).detach().cpu(),
            "attn_quantile": torch.full((B,), float(self.attn_quantile) if self.attn_quantile is not None else float("nan")).detach().cpu(),
            "strength_sim_mean": sims_clipped.mean(dim=0).detach().cpu(),
            "strength_type": "weighted_similarity",
        }

        return sent_signal, strength_signal

class TrimmedWeightedScorer(CompositeScorer):
    """
    Robust composite via symmetric trimming or winsorization on the chosen axis
    ('sentiment' or 'similarity'), then similarity-weighted combination.

    Strength = (w · s_trim) ∈ [0,1]
    Sentiment = (w · m_trim) ∈ [-1,1]
    Diagnostics: ess, breadth_norm, entropy, max_weight, var_sent, n, trim, trim_on, winsorize
    """

    def __init__(self, cfg: CompositeConfig):
        self.trim = float(getattr(cfg, "trim", 0.1))               # fraction in (0, 0.5]
        self.trim_on = getattr(cfg, "trim_on", "sentiment")        # "sentiment" | "similarity"
        self.winsorize = bool(getattr(cfg, "winsorize", False))
        self.last_diagnostics = {}

    def _empty_diag(self):
        return {
            "ess": 0.0, "breadth_norm": 0.0, "weight_entropy": 0.0,
            "max_weight": 0.0, "var_sent": 0.0, "n": 0,
            "trim": float(self.trim), "trim_on": self.trim_on, "winsorize": self.winsorize,
            "strength_sim_mean": 0.0,              # simple mean (reference only)
            "strength_type": "weighted_similarity",
        }

    def _boundaries(self, x):
        n = x.size
        lo = int(np.floor(self.trim * n))
        hi = n - lo
        order = np.argsort(x, kind="mergesort")
        return order, lo, hi

    def compute_composite(self, similarities, sentiments):
        s, m = prep_arrays(similarities, sentiments)
        n = s.size
        if n == 0:
            self.last_diagnostics = self._empty_diag()
            return 0.0, 0.0

        # If no trimming requested, just operate on full set (robust ‘off’)
        if self.trim <= 0.0:
            s_trim, m_trim = s, m
        else:
            axis = m if self.trim_on == "sentiment" else s
            order, lo, hi = self._boundaries(axis)
            if not self.winsorize:
                keep_idx = order[lo:hi]
                s_trim, m_trim = s[keep_idx], m[keep_idx]
            else:
                # winsorize the chosen axis; keep length
                x = axis.copy()
                lo_val = axis[order[min(lo, n-1)]]
                hi_val = axis[order[max(hi-1, 0)]]
                x[order[:lo]] = lo_val
                x[order[hi:]] = hi_val
                s_trim, m_trim = (s, x) if self.trim_on == "sentiment" else (x, m)

        n_eff = s_trim.size
        if n_eff == 0:
            self.last_diagnostics = self._empty_diag()
            return 0.0, 0.0

        # weights from trimmed similarities
        s_trim = np.clip(s_trim, 0.0, 1.0).astype(np.float64)
        Z = s_trim.sum()
        if not np.isfinite(Z) or Z <= 0.0:
            self.last_diagnostics = self._empty_diag()
            return 0.0, 0.0
        w = s_trim / Z

        sent_signal = float((w * m_trim).sum())         # [-1,1]
        strength_signal = float((w * s_trim).sum())     # [0,1]

        # diagnostics (on trimmed set)
        w_safe = np.clip(w, 1e-12, 1.0)
        entropy = float(-(w_safe * np.log(w_safe)).sum())
        ess = float(1.0 / np.sum(w**2))
        breadth_norm = float((ess - 1.0) / max(n_eff - 1, 1))
        var_sent = float((w * (m_trim - sent_signal) ** 2).sum())

        self.last_diagnostics = {
            "ess": ess,
            "breadth_norm": breadth_norm,
            "weight_entropy": entropy,
            "max_weight": float(w.max()),
            "var_sent": var_sent,
            "n": int(n_eff),
            "trim": float(self.trim),
            "trim_on": self.trim_on,
            "winsorize": self.winsorize,
            "strength_sim_mean": float(np.clip(s, 0, 1).mean()),  # reference
            "strength_type": "weighted_similarity",
        }
        return sent_signal, strength_signal

    @torch.inference_mode()
    def compute_composite_gpu(self, sims_tensor, sentiment_tensor, mask, device):
        """
        GPU version. Trimming/winsor is done per column (item) in a loop (reasonable S).
        """
        import torch
        S, B = sims_tensor.shape
        s = sims_tensor
        sent = sentiment_tensor  # (S,)
        scores, strengths = [], []

        for b in range(B):
            if mask is None:
                active = torch.ones(S, dtype=torch.bool, device=device)
            else:
                active = mask[:, b]

            s_b = s[:, b][active]
            m_b = sent[active]
            n_eff = s_b.shape[0]

            if n_eff == 0:
                scores.append(torch.tensor(0.0, device=device))
                strengths.append(torch.tensor(0.0, device=device))
                continue

            if self.trim <= 0.0:
                s_trim, m_trim = s_b, m_b
            else:
                axis = m_b if self.trim_on == "sentiment" else s_b
                # sort
                sorted_vals, order = torch.sort(axis)
                lo = int(torch.floor(torch.tensor(self.trim * n_eff)))
                hi = n_eff - lo
                lo = max(min(lo, n_eff), 0)
                hi = max(min(hi, n_eff), 0)

                if not self.winsorize:
                    keep_idx = order[lo:hi]
                    s_trim, m_trim = s_b[keep_idx], m_b[keep_idx]
                else:
                    lo_val = sorted_vals[min(lo, n_eff-1)]
                    hi_val = sorted_vals[max(hi-1, 0)]
                    axis_w = axis.clone()
                    if lo > 0:
                        axis_w[order[:lo]] = lo_val
                    if hi < n_eff:
                        axis_w[order[hi:]] = hi_val
                    s_trim, m_trim = (s_b, axis_w) if self.trim_on == "sentiment" else (axis_w, m_b)

            n_trim = s_trim.shape[0]
            if n_trim == 0:
                scores.append(torch.tensor(0.0, device=device))
                strengths.append(torch.tensor(0.0, device=device))
                continue

            s_trim = s_trim.clamp(0.0, 1.0)
            Z = s_trim.sum()
            if not torch.isfinite(Z) or Z <= 0:
                scores.append(torch.tensor(0.0, device=device))
                strengths.append(torch.tensor(0.0, device=device))
                continue

            w = s_trim / (Z + 1e-8)
            score_b = (w * m_trim).sum()
            strength_b = (w * s_trim).sum()

            scores.append(score_b)
            strengths.append(strength_b)

        return torch.stack(scores), torch.stack(strengths)

class SeparateFeaturesScorer(CompositeScorer):
    def __init__(self, config: CompositeConfig):
        self.last_diagnostics = {}

    def compute_composite(self, similarities, sentiments):
        s, m = prep_arrays(similarities, sentiments)
        n = s.size
        if n == 0:
            self.last_diagnostics = {
                "ess":0.0,"breadth_norm":0.0,"weight_entropy":0.0,"max_weight":0.0,
                "var_sent":0.0,"n":0,"strength_sim_mean":0.0,"strength_type":"weighted_similarity",
            }
            return 0.0, 0.0

        s = np.clip(s, 0.0, 1.0).astype(np.float64)
        Z = s.sum()
        if not np.isfinite(Z) or Z <= 0.0:
            self.last_diagnostics = {
                "ess":0.0,"breadth_norm":0.0,"weight_entropy":0.0,"max_weight":0.0,
                "var_sent":0.0,"n":n,"strength_sim_mean":float(s.mean()),
                "strength_type":"weighted_similarity",
            }
            return 0.0, 0.0

        w = s / Z
        sent_signal     = float((w * m).sum())   # [-1,1]
        strength_signal = float((w * s).sum())   # [0,1]

        w_safe  = np.clip(w, 1e-12, 1.0)
        entropy = float(-(w_safe * np.log(w_safe)).sum())
        ess = float(1.0 / np.sum(w**2))
        breadth_norm = float((ess - 1.0) / max(n - 1, 1))
        var_sent = float((w * (m - sent_signal) ** 2).sum())

        self.last_diagnostics = {
            "ess": ess, "breadth_norm": breadth_norm, "weight_entropy": entropy,
            "max_weight": float(w.max()), "var_sent": var_sent, "n": int(n),
            "strength_sim_mean": float(s.mean()),  # simple mean for reference only
            "strength_type":"weighted_similarity",
        }
        return sent_signal, strength_signal

    @torch.inference_mode()
    def compute_composite_gpu(self, sims_tensor, sentiment_tensor, mask, device):
        import torch
        S, B = sims_tensor.shape
        s = sims_tensor.clamp(0.0, 1.0)
        m = sentiment_tensor.view(S, 1).expand(S, B)

        if mask is not None:
            s_masked = torch.where(mask, s, torch.zeros_like(s))
        else:
            s_masked = s

        Z = s_masked.sum(dim=0, keepdim=True)              # (1,B)
        valid = Z.squeeze(0) > 0
        Z = torch.where(valid.unsqueeze(0), Z, torch.ones_like(Z))  # avoid div-by-0

        w = s_masked / Z                                   # (S,B)
        sent_signal     = (w * m).sum(dim=0)               # (B,)
        strength_signal = (w * s).sum(dim=0)               # (B,)

        # zero-out invalid columns
        if (~valid).any():
            sent_signal[~valid] = 0.0
            strength_signal[~valid] = 0.0

        # (optional) you can also compute and stash diagnostics like the CPU version
        return sent_signal, strength_signal
    
def _select_indices_for_composite(similarities: np.ndarray, keep_mask, comp_cfg) -> np.ndarray:
    mode = getattr(comp_cfg, "apply_over", "post_threshold")
    sims = np.asarray(similarities, float)

    if mode == "post_threshold":
        if keep_mask is None:
            # nothing was thresholded; use all (backward compatible)
            return np.arange(sims.size)
        return np.where(keep_mask)[0]

    elif mode == "loose_quantile":
        q = float(getattr(comp_cfg, "loose_quantile", 0.6))
        thr = np.quantile(sims[np.isfinite(sims)], q) if sims.size else 1.0
        return np.where(sims >= thr)[0]

    elif mode == "all":
        return np.arange(sims.size)

    else:
        # fallback
        return np.where(keep_mask)[0] if keep_mask is not None else np.arange(sims.size)
    
# ============================================================================
# Factory functions
# ============================================================================

# single registry-based factory
_SEGMENTER_REGISTRY = {
    "sentence":       SentenceSegmenter,
    "clause":         ClauseSegmenter,
    "paragraph":      ParagraphSegmenter,
    "sliding_window": SlidingWindowSegmenter,
    "semantic":       SemanticSegmenter,
}

def build_segmenter(config: SegmentationConfig, store=None, tokenizer=None):
    cls = _SEGMENTER_REGISTRY.get(config.method)
    if cls is None:
        raise ValueError(f"Unknown segmentation method: {config.method}")
    return cls(config, store=store, tokenizer=tokenizer)

# Keep backwards compatibility
create_segmenter = build_segmenter

def create_thresholder(config: ThresholdConfig) -> Thresholder:
    """Factory function to create appropriate thresholder"""
    method = config.method
    
    if method == 'hard':
        return HardThresholder(config)
    elif method == 'top_k':
        return TopKThresholder(config)
    elif method == 'soft_weight':
        return SoftWeightThresholder(config)
    elif method == 'adaptive':
        return AdaptiveThresholder(config)
    elif method == 'quantile':
        return QuantileThresholder(config)
    elif method == 'diversity_top_k':
        return DiversityTopKThresholder(config)
    elif method == 'topp':  # NEW
        return TopPThresholder(config)
    elif method == 'percent_of_max':  # NEW
        return PercentOfMaxThresholder(config)
    elif method == 'zscore':  # NEW
        return ZScoreThresholder(config)
    else:
        raise ValueError(f"Unknown threshold method: {method}")
    
def create_composite_scorer(cfg: CompositeConfig) -> CompositeScorer:
    if cfg.method == 'weighted_average':
        return WeightedAverageScorer(cfg)
    if cfg.method == 'attention_weight':
        return AttentionWeightScorer(cfg)
    if cfg.method == 'separate_features':
        return SeparateFeaturesScorer(cfg)
    if cfg.method == 'topk_attention':
        return TopKAttentionScorer(cfg)
    if cfg.method == 'power_attention':
        return PowerAttentionScorer(cfg)
    if cfg.method == 'trimmed_weighted':
        return TrimmedWeightedScorer(cfg)
    raise ValueError(f"Unknown composite method: {cfg.method}")