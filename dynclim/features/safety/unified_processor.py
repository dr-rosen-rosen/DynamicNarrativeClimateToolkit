"""
Unified Safety Feature Processor
=================================

A consolidated processor for extracting safety features from narrative reports.
Combines core processing with intelligent caching and GPU acceleration.

Key Features:
- Segment-level embedding cache for maximum reuse
- GPU/MPS acceleration support
- Batch optimization for multi-configuration processing
- Per-item segmentation and scoring
"""

import hashlib
import numpy as np
import pandas as pd
import torch
import multiprocessing as mp
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
# from concurrent.futures import ProcessPoolExecutor, as_completed
import time

from sentence_transformers import SentenceTransformer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class ItemFeatureResult:
    item_text: str

    # detailed per-report evidence
    selected_segments: List[str] = field(default_factory=list)
    segment_similarities: List[float] = field(default_factory=list)
    segment_sentiments: List[float] = field(default_factory=list)

    # core signals
    sent_signal: float = 0.0          # signed valence about this construct
    strength_signal: float = 0.0      # salience / focus of construct in this report
    final_score: float = 0.0          # sent_signal * strength_signal

    # extras
    n_segments: int = 0
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReportFeatureResult:
    """Results for a single report across all items"""
    report_id: str
    item_results: Dict[str, ItemFeatureResult]
    domain_scores: Dict[str, float]
    overall_final_score: float
    overall_sent: float
    overall_strength: float
    processing_time: float
    success: bool
    error_message: Optional[str] = None

# ============================================================================
# Helpers for hashing
# ============================================================================


def _as_text(seg):
    # Normalize any segment-ish object to a string for keys
    if isinstance(seg, str):
        return seg
    if isinstance(seg, dict):
        return seg.get("text") or str(seg)
    return str(seg)

def _item_key(item):
    # Normalize any item (str or dict) to a stable string key
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        return item.get("text") or item.get("name") or item.get("id") or str(item)
    return str(item)

def _extract_diag_from_configs(segments, sims_all, selected_segments, selected_similarities,
                               threshold_config, composite_config, thresholder=None, composer=None):
    """
    Build a diagnostics dict similar to the legacy per-item path.
    """
    n_total = len(segments)
    n_selected = len(selected_segments)
    sel_rate = (n_selected / n_total) if n_total else 0.0

    # Basic similarity summaries
    import numpy as np
    sims_all = np.asarray(sims_all, dtype=float)
    sims_sel = np.asarray(selected_similarities, dtype=float) if selected_similarities else np.array([], dtype=float)

    diag = {
        # selection-side facts
        "n_total": n_total,
        "n_selected": n_selected,
        "selection_rate": sel_rate,
        "selection_mode": getattr(composite_config, "apply_over", "post_threshold"),
        "threshold_method": getattr(threshold_config, "method", None),
        "threshold_value": getattr(threshold_config, "threshold", None),
        "k_used": getattr(threshold_config, "k", None),
        "q_used": getattr(threshold_config, "q", None),
        "lambda_used": getattr(threshold_config, "lambda_", None),

        # similarity summaries
        "sims_all_max": float(np.nanmax(sims_all)) if sims_all.size else None,
        "sims_all_mean": float(np.nanmean(sims_all)) if sims_all.size else None,
        "sims_sel_max": float(np.nanmax(sims_sel)) if sims_sel.size else None,
        "sims_sel_mean": float(np.nanmean(sims_sel)) if sims_sel.size else None,

        # scorer-side facts (namespaced with sc_)
        "scorer_method": getattr(composite_config, "method", None),
        "sc_tau": getattr(composite_config, "temperature", None),
        "sc_power": getattr(composite_config, "power", None),
        "sc_apply_over": getattr(composite_config, "apply_over", None),
    }

    # Pull optional diagnostics from thresholder/composer if they expose them
    th_diag = getattr(thresholder, "last_diagnostics", None)
    if isinstance(th_diag, dict) and th_diag:
        for k, v in th_diag.items():
            if k not in diag:  # Don't override standard fields
                diag[f"th_{k}"] = v

    sc_diag = getattr(composer, "last_diagnostics", None)
    if isinstance(sc_diag, dict) and sc_diag:
        for k, v in sc_diag.items():
            if k not in diag:  # Don't override standard fields
                diag[f"sc_{k}"] = v
    from dynclim.utils.diagnostics_schema import standardize_diagnostics
    diag = standardize_diagnostics(diag)
    return diag

# ============================================================================
# Main Processor Class
# ============================================================================

class Processor:
    """
    Unified processor for climate feature extraction with caching
    and acceleration.
    """
    
    def __init__(self, 
                 domain_definitions: Dict[str, List[str]],
                 embedding_model_name: str = 'all-MiniLM-L6-v2',
                 sentiment_method: str = 'vader',
                 # Caching options (default enabled for optimization)
                 cache_embeddings: bool = True,
                 cache_segments: bool = True,
                 cache_sentiments: bool = True,
                 normalize_segments: bool = True,
                 # Acceleration options
                 use_acceleration: bool = True,
                 device_pref: str = 'auto',
                 batch_size: int = 32,
                 store = None,
                 tok = None):
        """
        Initialize the processor.
        
        Args:
            domain_definitions: Dict mapping domain names to lists of scale items
            embedding_model_name: Name of sentence transformer model
            cache_embeddings: Enable segment-level embedding cache
            cache_segments: Enable text segmentation cache
            normalize_segments: Normalize segments before caching (improves hit rate)
            use_acceleration: Enable GPU/MPS acceleration
            device: Device for computation ('auto', 'cuda', 'mps', 'cpu')
            batch_size: Batch size for embedding computation
            sentiment_method: Method for sentiment analysis 
                ('vader', 'roberta', 'distilbert', 'finbert')
            cache_sentiments: Whether to cache sentiment scores
        """
        self.domain_definitions = domain_definitions
        self.embedding_model_name = embedding_model_name
        
        # Caching configuration
        self.cache_embeddings = cache_embeddings
        self.cache_segments = cache_segments
        self._segmenter_cache = {} 
        self.normalize_segments = normalize_segments
        # Acceleration configuration
        self.use_acceleration = use_acceleration
        self.batch_size = batch_size
        self.device_pref = device_pref
        self.device = self._setup_device(device_pref)
        # Sentiment
        self.sentiment_method = sentiment_method
        self.cache_sentiments = cache_sentiments
        self.sentiment_analyzer = self._init_sentiment_analyzer(sentiment_method, self.device)

        
        # Build item mappings
        self.all_items = []
        self.item_to_domain = {}
        self.item_to_column_name = {}
        for domain_name, items in domain_definitions.items():
            for idx, item in enumerate(items, 1):
                self.all_items.append(item)
                self.item_to_domain[item] = domain_name
                self.item_to_column_name[item] = f"{domain_name}_{idx}"
        
        # Initialize models
        self.embedding_model = SentenceTransformer(embedding_model_name, device=str(self.device))
        self.embedding_model.eval()
        # self.sentiment_analyzer = SentimentIntensityAnalyzer()
        # Pre-compute item embeddings
        self.item_embeddings = self.embedding_model.encode(self.all_items)
        self.item_to_embedding = {
            item: self.item_embeddings[i] 
            for i, item in enumerate(self.all_items)
        }
        # Normalize and keep an item matrix (numpy + torch) for fast matmul
        self._item_matrix_np = self.item_embeddings.astype(np.float32, copy=False)
        self._item_matrix_np /= (np.linalg.norm(self._item_matrix_np, axis=1, keepdims=True) + 1e-12)
        self._item_matrix_cpu = torch.as_tensor(self._item_matrix_np, dtype=torch.float32, device="cpu").contiguous()
        self._item_matrix_torch = None
        self._item_matrix_tT    = None   # (D, items) pre-transposed for fast block picks
        if self.use_acceleration:
            # self._item_matrix_torch = torch.from_numpy(self._item_matrix_np).to(self.device)
            dev = self.device  # already a torch.device
            self._item_matrix_torch = self._item_matrix_cpu.to(dev, non_blocking=True)
            self._item_matrix_tT    = self._item_matrix_torch.transpose(0, 1).contiguous()  # (D, items)
        # Initialize caches
        self._segment_embedding_cache = {}  # segment_hash -> embedding
        self._text_segment_cache = {}       # (text_hash, config_hash) -> segments
        self._similarity_cache = {}         # (segments_hash, item_hash) -> similarities
        self._unique_segments_seen = set()
        self._sentiment_cache = {}  # segment_hash -> sentiment_score
        
        # Cache statistics
        self.cache_stats = {
            'segment_embedding_hits': 0,
            'segment_embedding_misses': 0,
            'text_segment_hits': 0,
            'text_segment_misses': 0,
            'similarity_hits': 0,
            'similarity_misses': 0,
            'unique_segments': 0,
            'total_segments_processed': 0,
            'sentiment_hits': 0,
            'sentiment_misses': 0
        }
        self.use_gpu_scorer = True #getattr(self, "use_gpu_scorer", False)
        # optional: expose selector/scorer configs; fall back to your existing threshold/composite configs
        # self.selector_cfg = getattr(self, "selector_cfg", None)  # dict like {"type":"quantile","q":0.9}
        # self.scorer_cfg   = getattr(self, "scorer_cfg",   None)  # dict like {"type":"sim_weighted"}

        # sanity: ensure item matrix torch tensor exists if acceleration is on
        # if getattr(self, "use_acceleration", False) and getattr(self, "_item_matrix_torch", None) is None:
        #     # build from your existing numpy item matrix (I, D) float32
        #     # import torch
        #     # import numpy as np
        #     assert hasattr(self, "_item_matrix_np"), "Expect _item_matrix_np (I,D) float32 already built"
        #     self._item_matrix_torch = torch.from_numpy(self._item_matrix_np).to(self.device)
        # debugging
        self.debug_segmentation = True

        print(f"✅ Initialized Processor:")
        print(f"   Domains: {list(domain_definitions.keys())}")
        print(f"   Total items: {len(self.all_items)}")
        print(f"   Device: {self.device}")
        print(f"   Caching: embeddings={cache_embeddings}, segments={cache_segments}")
    
    # ========================================================================
    # Device Setup
    # ========================================================================
    
    def _setup_device(self, device: str = "auto") -> torch.device:
        """Select computation device for acceleration (CUDA, MPS, or CPU)."""
        if not getattr(self, "use_acceleration", True):
            print("💻 Acceleration disabled; using CPU.")
            return torch.device("cpu")

        # Automatic selection
        if device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                print("🚀 Using CUDA GPU acceleration")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
                print("🚀 Using Apple Silicon MPS acceleration")
            else:
                device = torch.device("cpu")
                print("💻 Using CPU (no GPU acceleration available)")
        else:
            # Trust user-specified device (e.g., "cuda", "mps", or "cpu")
            device = torch.device(device)
            print(f"🔧 Using manually specified device: {device}")

        return device
    
    # ============================================================================
    # GPU Diagnostics
    # ============================================================================
    def _build_gpu_diagnostics(
        self,
        sims_block,
        mask,
        threshold_config,
        composite_config
    ):
        import torch
        import numpy as np

        S, B = sims_block.shape

        # Count selected segments per item
        if mask is None:
            n_selected = torch.full((B,), S, dtype=torch.int32)
        else:
            n_selected = mask.sum(dim=0).to("cpu").numpy()

        sims_all_max  = sims_block.max(dim=0).values.to("cpu").numpy()
        sims_all_mean = sims_block.mean(dim=0).to("cpu").numpy()
        sims_all_std  = sims_block.std(dim=0).to("cpu").numpy()

        if mask is None:
            sims_sel_max  = sims_all_max
            sims_sel_mean = sims_all_mean
            sims_sel_std  = sims_all_std
        else:
            # max over selected only
            sims_masked = sims_block.clone()
            sims_masked[~mask] = -1e9
            sims_sel_max = sims_masked.max(dim=0).values.to("cpu").numpy()

            cnt = mask.sum(dim=0).float().clamp(min=1)
            sims_sel_mean = ((sims_block * mask).sum(dim=0) / cnt).to("cpu").numpy()
            # std of selected: optional for now
            sims_sel_std = np.zeros(B, dtype=np.float32)

        diag = {
            "n_total": np.full(B, S, dtype=np.int32),
            "n_selected": n_selected,
            "sims_all_max": sims_all_max,
            "sims_all_mean": sims_all_mean,
            "sims_all_std": sims_all_std,
            "sims_sel_max": sims_sel_max,
            "sims_sel_mean": sims_sel_mean,
            "sims_sel_std": sims_sel_std,

            # config info
            "selector_type": threshold_config.method,
            "scorer_method": composite_config.method,
        }

        return diag
    # @torch.inference_mode()
    # def _build_gpu_diagnostics(self, sims_block, mask, 
    #                         #    scores, strength,
    #                         # strength_type, entropy, 
    #                         threshold_config, composite_config):
        
    #     """
    #                         sims_block, mask, sent_signal_cpu, strength_signal_cpu, 
    #                 strength_type, entropy, threshold_config, composite_config
    #     Build standardized diagnostics from GPU computation.

    #     sims_block: torch.Tensor (S, B) [device]
    #     mask:       torch.BoolTensor (S, B) or None
    #     scores:     np.ndarray (B,)   # already on CPU
    #     strength:   np.ndarray (B,)   # already on CPU
    #     entropy:    torch.Tensor (B,) or None
    #     """
    #     import torch

    #     S, B = sims_block.shape
    #     device = sims_block.device

    #     # If no mask, treat as all-selected
    #     if mask is None:
    #         mask = torch.ones_like(sims_block, dtype=torch.bool, device=device)

    #     # ---- Compute per-item summaries ON DEVICE (vectorized) ----
    #     # counts
    #     sel_count = mask.sum(dim=0)                                  # (B,)

    #     # all-segment stats
    #     sims_all_max  = sims_block.max(dim=0).values                 # (B,)
    #     sims_all_mean = sims_block.mean(dim=0)                       # (B,)
    #     sims_all_std  = sims_block.std(dim=0, unbiased=False)        # (B,)

    #     # selected-only stats
    #     sims_masked   = sims_block.masked_fill(~mask, 0.0)           # (S,B)
    #     sel_cnt_safe  = sel_count.clamp_min(1)

    #     sims_sel_sum  = sims_masked.sum(dim=0)                       # (B,)
    #     sims_sel_mean = sims_sel_sum / sel_cnt_safe                  # (B,)

    #     sims_sq_sum   = (sims_masked * sims_masked).sum(dim=0)       # (B,)
    #     ex2           = sims_sq_sum / sel_cnt_safe                   # (B,)
    #     var_sel       = (ex2 - sims_sel_mean * sims_sel_mean).clamp_min(0.0)
    #     sims_sel_std  = var_sel.sqrt()                                # (B,)

    #     sims_sel_max  = sims_block.masked_fill(~mask, -1e9).max(dim=0).values

    #     # entropy may already be computed in scorer (preferred); normalize here
    #     # if entropy is not None and torch.is_tensor(entropy):
    #     #     entropy_vec = entropy
    #     # else:
    #     #     entropy_vec = None

    #     # ---- Move only the SMALL vectors once to CPU -> lists ----
    #     diag = {
    #         # counts
    #         "n_total": S,                                           # scalar (same for all items)
    #         "n_selected": sel_count.int().cpu().tolist(),           # list[int], len B

    #         # similarity stats (lists of length B)
    #         "sims_all_max":  sims_all_max.cpu().tolist(),
    #         "sims_all_mean": sims_all_mean.cpu().tolist(),
    #         "sims_all_std":  sims_all_std.cpu().tolist(),
    #         "sims_sel_max":  sims_sel_max.cpu().tolist(),
    #         "sims_sel_mean": sims_sel_mean.cpu().tolist(),
    #         "sims_sel_std":  sims_sel_std.cpu().tolist(),

    #         # strength & scorer meta
    #         # "strength_sim_mean": strength.tolist() if hasattr(strength, "tolist") else list(strength),
    #         # "strength_entropy": (entropy_vec.cpu().tolist() if entropy_vec is not None else None),
    #         # "strength_type": strength_type,

    #         # config info (strings/scalars)
    #         "selector_type": threshold_config.method,
    #         "scorer_method": composite_config.method,

    #         # method params (already resolved defaults in your __post_init__)
    #         "threshold_value": getattr(threshold_config, "threshold", None),
    #         "k_used": getattr(threshold_config, "k", None),
    #         "q_used": getattr(threshold_config, "q", None),
    #         "p_used": getattr(threshold_config, "p", None),
    #         "frac_used": getattr(threshold_config, "frac", None),
    #         "zmin_used": getattr(threshold_config, "zmin", None),
    #     }

    #     return diag
    
    # ============================================================================
    # GPU Scorers
    # ============================================================================

    @torch.inference_mode()
    def _gpu_matmul(self, seg_embs, j0, j1):
        import torch, torch.nn.functional as F

        # (S, D) – coerce to torch on target device
        seg_t = torch.as_tensor(seg_embs, dtype=torch.float32, device=self.device)

        # Pretransposed (D, I) created in __init__; fall back if missing
        if getattr(self, "_item_matrix_tT", None) is None:
            # build once if not present
            self._item_matrix_cpu = torch.as_tensor(self._item_matrix_np, dtype=torch.float32, device="cpu").contiguous()
            self._item_matrix_torch = self._item_matrix_cpu.to(self.device, non_blocking=True)
            self._item_matrix_tT    = self._item_matrix_torch.transpose(0, 1).contiguous()

        item_block = self._item_matrix_tT[:, j0:j1]  # (D, B); already on self.device

        # Normalize (cosine)
        seg_t = F.normalize(seg_t, p=2, dim=1)
        # items were normalized in __init__; fine to skip here

        sims = seg_t @ item_block  # (S, B)
        # If you standardize sims to [0,1], uncomment:
        # sims = sims.mul_(0.5).add_(0.5)

        # Final belt-and-suspenders: ensure device
        if sims.device.type != self.device.type:
            sims = sims.to(self.device, non_blocking=True)
        return sims

    # ========================================================================
    # Segmentation (with caching)
    # ========================================================================

    def segment_text(self, text: str, segmentation_config) -> List[str]:
        """
        Segment text with optional caching and a cached segmenter instance.
        """
        from dynclim.features.safety.toolkit import build_segmenter

        unit = getattr(segmentation_config, "min_unit", "chars")
        cfg_hash = self._get_config_hash(segmentation_config)
        seg_key = (cfg_hash, unit)

        # --- get or build the segmenter once ---
        segmenter = self._segmenter_cache.get(seg_key)
        if segmenter is None:
            segmenter = build_segmenter(
                segmentation_config,
                store=self.store,
                tokenizer=(self.tok if unit == "tokens" else None)
            )
            self._segmenter_cache[seg_key] = segmenter
            if getattr(self, "debug_segmentation", False):
                try:
                    backend = getattr(segmenter, "_backend", type(segmenter).__name__)
                    print(f"[seg] built segmenter {backend} for key={seg_key}")
                except Exception:
                    pass

        # --- segment-level caching? (your existing logic) ---
        if not getattr(self, "cache_segments", False):
            return segmenter.segment(text)

        text_hash = self._get_text_hash(text)
        cache_key = (text_hash, cfg_hash)

        if cache_key in self._text_segment_cache:
            self.cache_stats['text_segment_hits'] += 1
            return self._text_segment_cache[cache_key]

        self.cache_stats['text_segment_misses'] += 1
        segments = segmenter.segment(text)
        self._text_segment_cache[cache_key] = segments
        return segments  
    
    # ========================================================================
    # Sentiment (with caching)
    # ========================================================================
    
    def _init_sentiment_analyzer(self, method: str, device: str):
        """Initialize the appropriate sentiment analyzer."""
        if method == 'vader':
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            return SentimentIntensityAnalyzer()
        
        elif method in ['roberta', 'distilbert', 'finbert']:
            from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

            model_map = {
                'roberta': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
                'distilbert': 'distilbert-base-uncased-finetuned-sst-2-english',
                'finbert': 'ProsusAI/finbert'
            }
            tok = AutoTokenizer.from_pretrained(model_map[method])
            mdl = AutoModelForSequenceClassification.from_pretrained(model_map[method])
            mdl.to(self.device, dtype=torch.float32)
            # For CUDA: device=0; for MPS/CPU: device=-1
            # pipe_device = 0 if self.device.type == "cuda" else -1
            pipe_device = 0 if self.device.type == "cuda" else -1  # HF pipeline doesn’t accept 'mps' index
            sa = pipeline(
                "sentiment-analysis",
                model=mdl,
                tokenizer=tok,
                # model=model_map[method],
                device=pipe_device,
                # max_length=512,
                framework="pt",
                # truncation=True
            )
            try:
                sa.model.to(self.device)
                sa.device = self.device
            except Exception:
                pass
            return sa
        
        else:
            raise ValueError(f"Unknown sentiment method: {method}")
    
    def get_sentiment(self, segment: str) -> float:
        """
        Get sentiment score for a segment with caching.
        
        Returns:
            float: Sentiment score in range [-1, 1]
        """
        if not self.cache_sentiments:
            return self._compute_sentiment(segment)
        
        # Check cache
        segment_hash = self._get_segment_hash(segment)
        # Cache key includes method to allow switching methods
        cache_key = f"{segment_hash}_{self.sentiment_method}"
        
        if cache_key in self._sentiment_cache:
            self.cache_stats['sentiment_hits'] += 1
            return self._sentiment_cache[cache_key]
        
        # Cache miss - compute
        self.cache_stats['sentiment_misses'] += 1
        sentiment = self._compute_sentiment(segment)
        self._sentiment_cache[cache_key] = sentiment
        
        return sentiment
    
    def _compute_sentiment(self, segment: str) -> float:
        """Compute sentiment score using the configured method."""
        if self.sentiment_method == 'vader':
            scores = self.sentiment_analyzer.polarity_scores(segment)
            return scores['compound']
        
        else:  # Transformer-based
            # Truncate to max length (transformers have token limits)
            result = self.sentiment_analyzer(segment[:512])[0]
            
            # Convert to -1 to 1 scale
            label = result['label'].upper()
            score = result['score']
            
            if label in ['POSITIVE', 'POS', 'LABEL_2']:
                return score
            elif label in ['NEGATIVE', 'NEG', 'LABEL_0']:
                return -score
            else:  # NEUTRAL or LABEL_1
                return 0.0
    
    def get_sentiments_batch(self, segments: List[str]) -> List[float]:
        """
        Get sentiment scores for multiple segments with caching and batch processing.
        
        This is more efficient than calling get_sentiment repeatedly.
        """
        if not self.cache_sentiments:
            return self._compute_sentiments_batch(segments)
        
        sentiments = []
        segments_to_compute = []
        segment_indices = []
        
        # Check cache for each segment
        for i, segment in enumerate(segments):
            segment_hash = self._get_segment_hash(segment)
            cache_key = f"{segment_hash}_{self.sentiment_method}"
            
            if cache_key in self._sentiment_cache:
                sentiments.append(self._sentiment_cache[cache_key])
                self.cache_stats['sentiment_hits'] += 1
            else:
                segments_to_compute.append(segment)
                segment_indices.append(i)
                self.cache_stats['sentiment_misses'] += 1
        
        # Batch compute missing sentiments
        if segments_to_compute:
            new_sentiments = self._compute_sentiments_batch(segments_to_compute)
            
            # Store in cache and insert at correct positions
            for segment, sentiment, idx in zip(segments_to_compute, new_sentiments, segment_indices):
                segment_hash = self._get_segment_hash(segment)
                cache_key = f"{segment_hash}_{self.sentiment_method}"
                self._sentiment_cache[cache_key] = sentiment
                sentiments.insert(idx, sentiment)
        
        return sentiments
    
    def _compute_sentiments_batch(self, segments: List[str]) -> List[float]:
        """Compute sentiments for a batch of segments."""
        if self.sentiment_method == 'vader':
            # VADER is fast enough to process one-by-one
            return [self.sentiment_analyzer.polarity_scores(s)['compound'] 
                    for s in segments]
        
        else:  # Transformer-based - batch processing is crucial
            # Truncate all segments
            truncated = [s[:512] for s in segments]
            
            # Batch process (transformers library handles batching)
            results = self.sentiment_analyzer(truncated, batch_size=self.batch_size)
            
            # Convert to -1 to 1 scale
            sentiments = []
            for result in results:
                label = result['label'].upper()
                score = result['score']
                
                if label in ['POSITIVE', 'POS', 'LABEL_2']:
                    sentiments.append(score)
                elif label in ['NEGATIVE', 'NEG', 'LABEL_0']:
                    sentiments.append(-score)
                else:  # NEUTRAL
                    sentiments.append(0.0)
            
            return sentiments

    def switch_sentiment_method(self, new_method: str, external_cache: Optional[Dict] = None):
        """Switch to a different sentiment method."""
        if new_method == self.sentiment_method and external_cache is None:
            return
        
        # Reinitialize analyzer FIRST, then update method
        new_analyzer = self._init_sentiment_analyzer(new_method, str(self.device))
        print(f"Sentiment analyzer device: {self.sentiment_analyzer.device}")
        print(f"Torch device type: {self.device.type}")
        # Only update after successful initialization
        self.sentiment_analyzer = new_analyzer
        self.sentiment_method = new_method
        
        # Use external cache if provided
        if external_cache is not None:
            self._sentiment_cache = external_cache
            print(f"   Using external sentiment cache with {len(external_cache):,} entries")
        else:
            self._sentiment_cache.clear()
        
        # Reset sentiment-specific stats
        self.cache_stats['sentiment_hits'] = 0
        self.cache_stats['sentiment_misses'] = 0

    # ========================================================================
    # Embedding Computation (with segment-level caching)
    # ========================================================================
    
    def _all_item_similarities(self, seg_embs_np: np.ndarray, item_start: int = 0, item_end: Optional[int] = None) -> np.ndarray:
        """
        seg_embs_np: (S, D) float32. Returns (S, I_block) in [0,1] as in your accelerated path.
        Intent is to speed up embedding process
        """
        S, D = seg_embs_np.shape
        I = self._item_matrix_np.shape[0]
        if item_end is None or item_end > I:
            item_end = I
        if item_start < 0 or item_start >= item_end:
            raise ValueError("Invalid item block range.")

        # L2-normalize segments once
        seg = seg_embs_np.astype(np.float32, copy=False)
        seg /= (np.linalg.norm(seg, axis=1, keepdims=True) + 1e-12)

        # Torch fast path
        if self.use_acceleration and self._item_matrix_torch is not None:
            seg_t = torch.from_numpy(seg).to(self.device, non_blocking=True)
            sims_t = seg_t @ self._item_matrix_torch[item_start:item_end].T  # (S, I_block)
            sims = sims_t.clamp_(-1, 1).to("cpu").numpy()
        else:
            sims = seg @ self._item_matrix_np[item_start:item_end].T

        # match your 0..1 scaling
        np.clip(sims, -1.0, 1.0, out=sims)
        sims = 0.5 * (sims + 1.0)
        return sims
    
    def encode_segments(self, segments: List[str]) -> np.ndarray:
        """
        Encode segments with segment-level caching for maximum reuse.
        Uses GPU/MPS for the forward pass when available, then caches on CPU.
        """
        # Resolve embedding dim early if we have it
        if not segments:
            dim = getattr(self, "embedding_dim", 0)
            return np.zeros((0, dim), dtype=np.float32)

        # Prefer the resolved torch.device, not the preference string
        device_str = str(getattr(self, "device", "cpu"))  # 'cuda:0' | 'mps' | 'cpu'

        # Fast path: no caching — run entirely on device and return CPU numpy
        if not getattr(self, "cache_embeddings", False):
            with torch.inference_mode():
                embs_t = self.embedding_model.encode(
                    segments,
                    batch_size=getattr(self, "batch_size", 128),
                    show_progress_bar=False,
                    convert_to_tensor=True,
                    normalize_embeddings=True,
                    # device=device_str,  # optional if model already on device
                )
            embs = embs_t.float().cpu().numpy()
            # Track embedding_dim once
            if not hasattr(self, "embedding_dim"):
                self.embedding_dim = embs.shape[1] if embs.ndim == 2 else int(embs.shape[-1])
            return embs.astype(np.float32, copy=False)

        # -------- CACHED PATH ----------
        if not hasattr(self, "_segment_embedding_cache"):
            self._segment_embedding_cache = {}  # (model_name, seg_hash) -> np.ndarray
        if not hasattr(self, "cache_stats"):
            self.cache_stats = {
                "segment_embedding_hits": 0, "segment_embedding_misses": 0,
                "total_segments_processed": 0
            }
        if not hasattr(self, "_unique_segments_seen"):
            self._unique_segments_seen = set()

        model_name = getattr(self, "embedding_model_name", "default")
        batch_size = getattr(self, "batch_size", 128)

        n = len(segments)
        hits = {}
        to_encode, to_encode_idx = [], []

        for i, seg in enumerate(segments):
            seg_hash = self._get_segment_hash(seg)
            key = (model_name, seg_hash)
            cached = self._segment_embedding_cache.get(key)
            if cached is not None:
                hits[i] = cached
                self.cache_stats["segment_embedding_hits"] += 1
            else:
                to_encode.append(seg)
                to_encode_idx.append(i)
                self.cache_stats["segment_embedding_misses"] += 1
                self._unique_segments_seen.add(seg_hash)

        new_vecs_np: Optional[np.ndarray] = None
        if to_encode:
            with torch.inference_mode():
                embs_t = self.embedding_model.encode(
                    to_encode,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    convert_to_tensor=True,
                    normalize_embeddings=True,
                    # device=device_str,  # optional if model already on device
                )
            new_vecs_np = embs_t.float().cpu().numpy().astype(np.float32, copy=False)

            if not hasattr(self, "embedding_dim"):
                self.embedding_dim = new_vecs_np.shape[1] if new_vecs_np.ndim == 2 else int(new_vecs_np.shape[-1])

            # Write misses to cache
            for seg, vec in zip(to_encode, new_vecs_np):
                seg_hash = self._get_segment_hash(seg)
                key = (model_name, seg_hash)
                # store a compact float32 copy
                self._segment_embedding_cache[key] = vec

        # Assemble outputs in original order
        d = getattr(self, "embedding_dim", (new_vecs_np.shape[1] if new_vecs_np is not None else 0))
        out = np.empty((n, d), dtype=np.float32)

        for i, vec in hits.items():
            out[i] = vec
        if to_encode:
            for i, vec in zip(to_encode_idx, new_vecs_np):
                out[i] = vec

        self.cache_stats["total_segments_processed"] += n
        self.cache_stats["unique_segments"] = len(self._unique_segments_seen)

        return out
    # ========================================================================
    # Similarity Computation (with caching and acceleration)
    # ========================================================================
    
    def compute_similarities(self, 
                           segments: List[str],
                           segment_embeddings: np.ndarray,
                           item_embedding: np.ndarray,
                           item_text: str) -> np.ndarray:
        """
        Compute cosine similarities with caching and optional GPU acceleration.
        
        Args:
            segments: Original text segments
            segment_embeddings: Embeddings for segments
            item_embedding: Embedding for the safety item
            item_text: Text of the safety item
            
        Returns:
            Array of similarity scores
        """
        # Check similarity cache
        if self.cache_embeddings:
            segments_str = "|".join([self._normalize_segment(s) for s in segments])
            segments_hash = hashlib.md5(segments_str.encode()).hexdigest()
            item_hash = hashlib.md5(item_text.encode()).hexdigest()
            cache_key = (segments_hash, item_hash)
            
            if cache_key in self._similarity_cache:
                self.cache_stats['similarity_hits'] += 1
                return self._similarity_cache[cache_key]
            
            self.cache_stats['similarity_misses'] += 1
        
        # Compute similarities (with optional acceleration)
        if self.use_acceleration and self.device.type != 'cpu':
            similarities = self._compute_similarities_accelerated(
                segment_embeddings, item_embedding
            ).flatten()
        else:
            similarities = cosine_similarity(
                segment_embeddings,
                item_embedding.reshape(1, -1)
            ).flatten()
        
        # Cache result
        if self.cache_embeddings:
            self._similarity_cache[cache_key] = similarities
        
        return similarities
    
    def _compute_similarities_accelerated(self, 
                                         segment_embeddings: np.ndarray,
                                         item_embeddings: np.ndarray) -> np.ndarray:
        """Compute similarities using GPU/MPS acceleration."""
        try:
            # Convert to tensors and move to device
            segments_tensor = torch.tensor(segment_embeddings, dtype=torch.float32).to(self.device)
            items_tensor = torch.tensor(item_embeddings, dtype=torch.float32).to(self.device)
            
            if len(items_tensor.shape) == 1:
                items_tensor = items_tensor.unsqueeze(0)
            
            # Normalize and compute cosine similarity
            segments_norm = torch.nn.functional.normalize(segments_tensor, p=2, dim=1)
            items_norm = torch.nn.functional.normalize(items_tensor, p=2, dim=1)
            similarities = torch.mm(segments_norm, items_norm.t()).cpu().numpy()
            similarities = np.clip(similarities, -1.0, 1.0)
            similarities = 0.5 * (similarities + 1.0)

            return similarities
            
        except Exception as e:
            print(f"⚠️ Acceleration failed, falling back to CPU: {e}")
            if len(item_embeddings.shape) == 1:
                item_embeddings = item_embeddings.reshape(1, -1)
            return cosine_similarity(segment_embeddings, item_embeddings)
    
    # ========================================================================
    # Core Processing Pipeline
    # ========================================================================
    
    # def process_report_for_item(self,
    #                            text: str,
    #                            item_text: str,
    #                            item_embedding: np.ndarray,
    #                            segmentation_config,
    #                            threshold_config,
    #                            composite_config) -> ItemFeatureResult:
    #     """
    #     Process a single report for a single safety scale item.
        
    #     This performs per-item segmentation and thresholding, which is the
    #     key to accurate feature extraction.
        
    #     Args:
    #         text: Report text
    #         item_text: Safety scale item text
    #         item_embedding: Pre-computed embedding for the item
    #         segmentation_config: Segmentation configuration
    #         threshold_config: Threshold configuration
    #         composite_config: Composite scoring configuration
            
    #     Returns:
    #         ItemFeatureResult for this item
    #     """
    #     from dynclim.features.safety.toolkit import create_thresholder, create_composite_scorer, _select_indices_for_composite
        
    #     # Step 1: Segment the text
    #     segments = self.segment_text(text, segmentation_config)
        
    #     if not segments:
    #         return ItemFeatureResult(
    #             item_text=item_text,
    #             selected_segments=[],
    #             segment_similarities=[],
    #             segment_sentiments=[],
    #             sent_signal = 0.0          # signed valence about this construct
    #             strength_signal = 0.0      # salience / focus of construct in this report
    #             final_score = 0.0   
    #             #composite_score=0.0,
    #             #trength_score=0.0,
    #             n_segments=0
    #         )
        
    #     # Step 2: Compute embeddings (with caching)
    #     segment_embeddings = self.encode_segments(segments)
        
    #     # Step 3: Compute similarities (with caching and acceleration)
    #     similarities = self.compute_similarities(
    #         segments, segment_embeddings, item_embedding, item_text
    #     )


    #     # Step 3: Compute similarities
    #     similarities = self.compute_similarities(segments, segment_embeddings, item_embedding, item_text)

    #     comp_mode = getattr(composite_config, "apply_over", "post_threshold")

    #     # --- selection logic wrapper ---
    #     if comp_mode == "post_threshold":
    #         # Use your existing thresholder unchanged
    #         thresholder = create_thresholder(threshold_config)
    #         selected_segments, selected_similarities = thresholder.select_segments(
    #             segments, similarities.tolist()
    #         )
    #         idx = None  # not needed; you already have the reduced arrays

    #     elif comp_mode == "loose_quantile":
    #         # Skip the thresholder; keep the top loose_quantile by similarity
    #         sims = np.asarray(similarities, dtype=float)
    #         if sims.size == 0 or not np.isfinite(sims).any():
    #             return ItemFeatureResult(
    #                 item_text=item_text, selected_segments=[], segment_similarities=[],
    #                 segment_sentiments=[], composite_score=0.0, strength_score=0.0, n_segments=0
    #             )
    #         q = float(getattr(composite_config, "loose_quantile", 0.6))
    #         thr = np.quantile(sims[np.isfinite(sims)], q)
    #         idx = np.where(sims >= thr)[0]
    #         selected_segments = [segments[i] for i in idx]
    #         selected_similarities = sims[idx].astype(float).tolist()

    #     elif comp_mode == "all":
    #         # Skip the thresholder entirely; use all segments/similarities
    #         sims = np.asarray(similarities, dtype=float)
    #         idx = np.arange(sims.size)
    #         selected_segments = segments  # already a list
    #         selected_similarities = sims.astype(float).tolist()

    #     else:
    #         # Fallback: default to your original path
    #         thresholder = create_thresholder(threshold_config)
    #         selected_segments, selected_similarities = thresholder.select_segments(
    #             segments, similarities.tolist()
    #         )
    #         idx = None

    #     # Early exit if nothing selected
    #     if not selected_segments:
    #         return ItemFeatureResult(
    #             item_text=item_text,
    #             selected_segments=[],
    #             segment_similarities=[],
    #             segment_sentiments=[],
    #             composite_score=0.0,
    #             strength_score=0.0,
    #             n_segments=0
    #         )

    #     # Step 5: Compute sentiment ONLY for selected segments
    #     sentiments = self.get_sentiments_batch(selected_segments)

    #     # Step 6: Composite
    #     composer = create_composite_scorer(composite_config)
    #     composite_score, strength_score = composer.compute_composite(
    #         selected_similarities, sentiments
    #     )

    #     diag = {
    #         # selection-side facts
    #         "n_total": len(segments),                                 # all segs before selection
    #         "n_selected": len(selected_segments),
    #         "selection_mode": getattr(composite_config, "apply_over", "post_threshold"),
    #         "threshold_method": getattr(threshold_config, "method", None),
    #         "threshold_value": getattr(threshold_config, "threshold", None),
    #         "k_used": getattr(threshold_config, "k", None),
    #         "q_used": getattr(threshold_config, "q", None),
    #         "lambda_used": getattr(threshold_config, "lambda_", None),
    #         # scorer-side facts (namespaced with sc_)
    #         "scorer_method": composite_config.method,
    #         "sc_tau": getattr(composite_config, "temperature", None),
    #         "sc_power": getattr(composite_config, "power", None),
    #         "sc_apply_over": getattr(composite_config, "apply_over", None),
    #     }
    #     # pull scorer-provided diagnostics if present
    #     sc_diag = getattr(composer, "last_diagnostics", None)
    #     if isinstance(sc_diag, dict) and sc_diag:
    #         for k, v in sc_diag.items():
    #             diag[f"sc_{k}"] = v

    #     return ItemFeatureResult(
    #         item_text=item_text,
    #         selected_segments=selected_segments,
    #         segment_similarities=selected_similarities,
    #         segment_sentiments=sentiments,
    #         composite_score=composite_score,
    #         strength_score=strength_score,
    #         n_segments=len(selected_segments),
    #         diagnostics=diag
    #     )
    
    def process_single_report(self,
                             report_id: str,
                             text: str,
                             segmentation_config,
                             threshold_config,
                             composite_config) -> ReportFeatureResult:
        """
        Process a single report across all safety scale items.
        
        Args:
            report_id: Unique identifier for the report
            text: Report text
            segmentation_config: Segmentation configuration
            threshold_config: Threshold configuration
            composite_config: Composite scoring configuration
            
        Returns:
            ReportFeatureResult with scores for all items and domains
        """
        start_time = time.time()
        
        try:
            # Process report for each item separately

            item_results = self._process_report_all_items(
                report_text=text,
                segmentation_config=segmentation_config,
                threshold_config=threshold_config,
                composite_config=composite_config,
                item_block_size=256  # tune 128–512 if you have many items
            )

            # Calculate domain scores
            domain_scores = {}
            for domain_name, domain_items in self.domain_definitions.items():
                domain_item_scores = [
                    item_results[item].final_score
                    for item in domain_items 
                    if item in item_results
                ]
                domain_scores[domain_name] = np.mean(domain_item_scores) if domain_item_scores else 0.0
            
            # Calculate overall scores
            all_final_scores = [r.final_score for r in item_results.values()]
            all_sent_scores = [r.sent_signal for r in item_results.values()]
            all_strength_scores = [r.strength_signal for r in item_results.values()]
            
            overall_final = np.mean(all_final_scores) if all_final_scores else 0.0
            overall_sent = np.mean(all_sent_scores) if all_sent_scores else 0.0
            overall_strength = np.mean(all_strength_scores) if all_strength_scores else 0.0
            
            processing_time = time.time() - start_time
            
            return ReportFeatureResult(
                report_id=report_id,
                item_results=item_results,
                domain_scores=domain_scores,
                overall_final_score = overall_final,
                overall_sent = overall_sent,
                overall_strength = overall_strength,
                processing_time=processing_time,
                success=True
            )
            
        except Exception as e:
            import traceback
            processing_time = time.time() - start_time
            
            # Print full traceback for debugging
            print(f"\n❌ Error processing report {report_id}:")
            print(f"   Sentiment method: {self.sentiment_method}")
            print(f"   Analyzer type: {type(self.sentiment_analyzer)}")
            traceback.print_exc()
            
            return ReportFeatureResult(
                report_id=report_id,
                item_results={},
                domain_scores={domain: 0.0 for domain in self.domain_definitions},
                overall_final_score=0.0,
                overall_sent=0.0,
                overall_strength=0.0,
                processing_time=processing_time,
                success=False,
                error_message=f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            )
    
    def process_reports(self,
                       reports_data: List[Tuple[str, str]],
                       segmentation_config,
                       threshold_config,
                       composite_config,
                       use_parallel: bool = True,
                       max_workers: Optional[int] = None,
                       batch_size: int = 50) -> pd.DataFrame:
        """
        Process multiple reports and return results as DataFrame.
        
        Args:
            reports_data: List of (report_id, text) tuples, dicts, or text strings
            segmentation_config: Segmentation configuration
            threshold_config: Threshold configuration
            composite_config: Composite scoring configuration
            use_parallel: Whether to use parallel processing
            max_workers: Number of parallel workers
            batch_size: Batch size for parallel processing
            
        Returns:
            DataFrame with results
        """
        # Normalize input data
        normalized_data = self._normalize_input_data(reports_data)
        
        print(f"\n🚀 Starting processing...")
        print(f"   Reports: {len(normalized_data)}")
        print(f"   Parallel: {use_parallel}")
        
        start_time = time.time()

        # Process sequentially or in parallel
        if use_parallel and len(normalized_data) > 1:
            results = self._process_parallel(
                normalized_data, segmentation_config, threshold_config,
                composite_config, max_workers, batch_size
            )
        else:
            results = self._process_sequential(
                normalized_data, segmentation_config, threshold_config, composite_config
            )
        
        # Convert results to DataFrame
        df_data = []
        for result in results:
            row = {
                'report_id': result.report_id,
                'overall_final_score': result.overall_final_score,
                'overall_sent': result.overall_sent,
                'overall_strength':result.overall_strength,
                # 'overall_composite_score': result.overall_composite_score,
                # 'overall_strength_score': result.overall_strength_score,
                'processing_success': result.success,
                'processing_time': result.processing_time,
                'error_message': result.error_message
            }
            
            # Add item scores and diagnostics
            for item_text, item_result in result.item_results.items():
                col_name = self.item_to_column_name[item_text]
                row[f'{col_name}_final'] = item_result.final_score
                row[f'{col_name}_sent'] = item_result.sent_signal
                row[f'{col_name}_strength'] = item_result.strength_signal
                row[f'{col_name}_n_segments'] = item_result.n_segments
                # flatten diagnostics under the same col_name
                if getattr(item_result, "diagnostics", None):
                    for k, v in item_result.diagnostics.items():
                        row[f'diag_{col_name}_{k}'] = v
            # Add domain scores
            for domain, score in result.domain_scores.items():
                row[f'{domain}_domain_score'] = score
            
            df_data.append(row)
        
        results_df = pd.DataFrame(df_data)
        
        total_time = time.time() - start_time
        print(f"✅ Processing complete!")
        print(f"   Total time: {total_time:.2f} seconds")
        print(f"   Rate: {len(normalized_data)/total_time:.1f} reports/second")
        
        return results_df
    
    def process_reports_batch_optimized(self,
                                       reports_data: List[Tuple[str, str]],
                                       segmentation_config,
                                       threshold_config,
                                       composite_config,
                                       **kwargs):
        """
        Batch-optimized processing that pre-caches all unique segments.
        
        This is particularly effective when processing the same reports
        with multiple configurations, as it maximizes cache reuse.
        
        Args:
            reports_data: List of (report_id, text) tuples
            segmentation_config: Segmentation configuration
            threshold_config: Threshold configuration  
            composite_config: Composite scoring configuration
            **kwargs: Additional arguments passed to process_reports
            
        Returns:
            DataFrame with results
        """
        print("\n🚀 Using batch-optimized processing...")
        
        # Normalize input
        normalized_data = self._normalize_input_data(reports_data)
        
        # Step 1: Pre-segment all texts
        print("   Pre-segmenting all texts...")
        all_segments_by_report = {}
        unique_segments = set()
        
        for report_id, text in normalized_data:
            segments = self.segment_text(text, segmentation_config)
            all_segments_by_report[report_id] = segments
            unique_segments.update(segments)
        
        print(f"   Found {len(unique_segments)} unique segments")
        
        # Step 2: Pre-encode all unique segments
        if unique_segments and self.cache_embeddings:
            print("   Pre-encoding unique segments...")
            unique_segments_list = list(unique_segments)
            _ = self.encode_segments(unique_segments_list)
            print(f"   ✅ Pre-cached {len(unique_segments_list)} embeddings")
        
        # Step 3: Process normally (will hit cache)
        return self.process_reports(
            normalized_data,
            segmentation_config,
            threshold_config,
            composite_config,
            **kwargs
        )
    def _ensure_items_ready(self):
        """
        Make sure _item_keys and item matrices exist and align with self.all_items.
        Safe to call repeatedly.
        """
        if not hasattr(self, "all_items") or not self.all_items:
            raise ValueError("Processor has no 'all_items' defined.")

        # Keys aligned to all_items
        if not hasattr(self, "_item_keys") or self._item_keys is None or len(self._item_keys) != len(self.all_items):
            self._item_keys = [_item_key(obj) for obj in self.all_items]

        # Item embedding matrix (numpy)
        if not hasattr(self, "_item_matrix_np") or self._item_matrix_np is None \
        or self._item_matrix_np.shape[0] != len(self.all_items):
            # Build from item texts if not already available
            texts = [_as_text(obj) for obj in self.all_items]
            embs = self.embedding_model.encode(
                texts,
                batch_size=getattr(self, "batch_size", 128),
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True  # cosine-ready
            ).astype("float32")
            self._item_matrix_np = embs

        # GPU tensor if using GPU scorer
        if getattr(self, "use_gpu_scorer", False):
            import torch
            if not hasattr(self, "_item_matrix_torch") or self._item_matrix_torch is None \
            or self._item_matrix_torch.shape[0] != self._item_matrix_np.shape[0]:
                self._item_matrix_torch = torch.from_numpy(self._item_matrix_np).to(self.device)

    def _process_report_all_items(self, report_text, segmentation_config, threshold_config, composite_config, item_block_size: int = 256):
        """
        1) segment + encode once
        2) compute sims for all items in blocks
        3) run threshold + composite per item (GPU or CPU)
        4) reuse a per-report sentiment cache (CPU path) or full-vector (GPU path)
        """

        from dynclim.features.safety.toolkit import create_thresholder, create_composite_scorer
        self._ensure_items_ready()  # ensures keys + matrices exist and align
        I = self._item_matrix_np.shape[0]
        assert I == len(self._item_keys) == len(self.all_items), \
            f"Item alignment mismatch: I={I}, keys={len(self._item_keys)}, items={len(self.all_items)}"
        # 1) Segment & encode once
        segments_raw = self.segment_text(report_text, segmentation_config)
        if not segments_raw:
            return {}

        segments = [_as_text(s) for s in segments_raw]
        seg_embs = self.encode_segments(segments)  # (S, D), float32
        S = seg_embs.shape[0]

        # 1b) Sentiment ONCE for whole report (GPU path needs full vector)
        sentiments = self.get_sentiments_batch(segments)
        import numpy as np
        sentiment_np = np.asarray(sentiments, dtype=np.float32)  # (S,)

        # Prepare helpers (CPU and GPU path)
        thresholder = create_thresholder(threshold_config)
        composer = create_composite_scorer(composite_config)
        sent_cache = {}  # seg_text -> sentiment (for CPU selected segments)

        # Routing
        use_gpu = bool(getattr(self, "use_gpu_scorer", False) and 
                    getattr(self, "_item_matrix_torch", None) is not None)
        I = int(self._item_matrix_np.shape[0])
        results = {}
        block = int(getattr(self, "item_block_size", item_block_size))
        sent_t = torch.from_numpy(sentiment_np).to(self.device)

        j0 = 0
        while j0 < I:
            j1 = min(j0 + block, I)

            if use_gpu:
                # ----- GPU path: compute sims on device and score there -----
                with torch.inference_mode():

                    sims_block = self._gpu_matmul(seg_embs, j0, j1)  # (S, B) on device

                    # Segment selection on GPU — respect apply_over mode
                    apply_mode = getattr(composite_config, "apply_over", "post_threshold")
                    if apply_mode == "all":
                        mask = None          # scorer uses all segments
                    else:
                        mask = thresholder.select_segments_gpu(sims_block, self.device)

                    # Scoring on GPU
                    sent_signal, strength_signal = composer.compute_composite_gpu(
                        sims_block, 
                        sent_t, 
                        mask, 
                        self.device
                    )

                    # Move to CPU for post-processing
                    sent_signal_cpu     = sent_signal.detach().to("cpu").numpy()
                    strength_signal_cpu = strength_signal.detach().to("cpu").numpy()
                    final_score_cpu     = sent_signal_cpu * strength_signal_cpu

                    # Optionally, still within inference mode, build diagnostics 
                    # (safe if it internally calls .cpu(), which it should)
                    diag = self._build_gpu_diagnostics(
                        sims_block, mask,
                        threshold_config, composite_config
                    )

                # exit inference mode here: everything below is CPU / NumPy
                for off, j in enumerate(range(j0, j1)):
                    item_key = self._item_keys[j]

                    sent_val     = float(sent_signal_cpu[off])
                    strength_val = float(strength_signal_cpu[off])
                    final_val    = float(final_score_cpu[off])

                    results[item_key] = ItemFeatureResult(
                        item_text=item_key,
                        final_score=final_val,
                        sent_signal=sent_val,
                        strength_signal=strength_val,
                        n_segments=int(diag["n_selected"][off]),
                        diagnostics={**{k: (v[off] if isinstance(v, list) else v)
                                        for k, v in diag.items()
                                        if k not in ("n_total",)}, 
                                    "n_total": diag["n_total"]},
                    )
            else:
                # ----- CPU path: compute sims block on CPU, then per-item selection/composite -----
                sims_block_np = self._all_item_similarities(seg_embs, item_start=j0, item_end=j1)  # (S,B) CPU float32

                for off, j in enumerate(range(j0, j1)):
                    item_key = self._item_keys[j] 
                    sims_col = sims_block_np[:, off]  # (S,)

                    # Threshold selection (expects segments + sims list)
                    selected_segments, selected_similarities = thresholder.select_segments(
                        segments, sims_col.tolist()
                    )
                    selected_segments = [_as_text(s) for s in selected_segments]

                    # Sentiment for selected segments via cache
                    selected_sentiments = []
                    for seg_text in selected_segments:
                        val = sent_cache.get(seg_text)
                        if val is None:
                            # sentiment_np is already computed; reuse if you can map seg->index
                            # If you don't have index mapping, fall back to analyzer:
                            try:
                                # quick map once
                                if "_seg_index" not in sent_cache:
                                    sent_cache["_seg_index"] = {s: i for i, s in enumerate(segments)}
                                idx = sent_cache["_seg_index"].get(seg_text)
                                val = float(sentiment_np[idx]) if idx is not None else float(self.get_sentiment(seg_text))
                            except Exception:
                                val = float(self.get_sentiment(seg_text))
                            sent_cache[seg_text] = val
                        selected_sentiments.append(val)

                    sent_signal, strength_signal = composer.compute_composite(
                        selected_similarities, selected_sentiments
                    )

                    final_score = float(sent_signal) * float(strength_signal)
                    
                    # Diagnostics (legacy-style)
                    diag = _extract_diag_from_configs(
                        segments=segments,
                        sims_all=sims_col,
                        selected_segments=selected_segments,
                        selected_similarities=selected_similarities,
                        threshold_config=threshold_config,
                        composite_config=composite_config,
                        thresholder=thresholder,
                        composer=composer,
                    )
                    results[item_key] = ItemFeatureResult(
                        item_text=item_key,

                        # keep the raw detail for diagnostics / later analysis
                        selected_segments=selected_segments,
                        segment_similarities=selected_similarities,
                        segment_sentiments=selected_sentiments,

                        # rename conceptually:
                        sent_signal=float(sent_signal),                # was composite_score
                        strength_signal=float(strength_signal),        # was strength_score
                        final_score=final_score,                       # new
                        n_segments=len(selected_segments),
                        diagnostics=diag,
                    )

            j0 = j1
            # print(f"[dbg-items] built {len(results)} item results for this report.")
            # for k, v in list(results.items())[:5]:
            #     print("   ", k, "final_score=", getattr(v, "final_score", None),
            #                     "sent_signal=", getattr(v, "sent_signal", None),
            #                     "strength_signal=", getattr(v, "strength_signal", None))
        return results
    
    def _process_sequential(self, 
                           reports_data: List[Tuple[str, str]],
                           segmentation_config,
                           threshold_config,
                           composite_config) -> List[ReportFeatureResult]:
        """Sequential processing of reports."""
        results = []
        
        for i, (report_id, text) in enumerate(reports_data):
            
            result = self.process_single_report(
                report_id, text, segmentation_config, 
                threshold_config, composite_config
            )
            results.append(result)
        
        return results
    
    def _process_parallel(self,
                         reports_data: List[Tuple[str, str]],
                         segmentation_config,
                         threshold_config,
                         composite_config,
                         max_workers: Optional[int] = None,
                         batch_size: int = 10) -> List[ReportFeatureResult]:
        """Parallel processing of reports."""
        if max_workers is None:
            max_workers = min(mp.cpu_count() - 1, 4)
        
        print(f"   Workers: {max_workers}")
        print(f"   Batch size: {batch_size}")
        
        # Convert configs to dicts for serialization
        seg_dict = self._config_to_dict(segmentation_config)
        thresh_dict = self._config_to_dict(threshold_config)
        comp_dict = self._config_to_dict(composite_config)
        
        # Create batches
        batches = [
            reports_data[i:i + batch_size]
            for i in range(0, len(reports_data), batch_size)
        ]
        
        all_results = []
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit batches
            future_to_batch = {
                executor.submit(
                    _process_batch_worker,
                    batch,
                    self.domain_definitions,
                    self.embedding_model_name,
                    seg_dict,
                    thresh_dict,
                    comp_dict
                ): batch_idx
                for batch_idx, batch in enumerate(batches)
            }
            
            # Collect results
            completed = 0
            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                try:
                    batch_results = future.result()
                    all_results.extend(batch_results)
                    completed += 1
                    print(f"   Completed batch {completed}/{len(batches)}")
                except Exception as e:
                    print(f"   Error in batch {batch_idx}: {str(e)}")
        
        return all_results
    
    # ========================================================================
    # Cache Analysis and Statistics
    # ========================================================================
    
    def get_cache_stats(self) -> Dict:
        """Get detailed cache statistics."""
        def calc_rate(hits, misses):
            total = hits + misses
            return hits / total if total > 0 else 0
        
        stats = {
            'segment_embedding_hit_rate': calc_rate(
                self.cache_stats['segment_embedding_hits'],
                self.cache_stats['segment_embedding_misses']
            ),
            'sentiment_hit_rate': calc_rate(  # NEW
                self.cache_stats['sentiment_hits'],
                self.cache_stats['sentiment_misses']
            ),
            'text_segment_hit_rate': calc_rate(
                self.cache_stats['text_segment_hits'],
                self.cache_stats['text_segment_misses']
            ),
            'similarity_hit_rate': calc_rate(
                self.cache_stats['similarity_hits'],
                self.cache_stats['similarity_misses']
            ),
            'overall_hit_rate': calc_rate(
                sum(v for k, v in self.cache_stats.items() if 'hits' in k),
                sum(v for k, v in self.cache_stats.items() if 'misses' in k)
            ),
            'cache_sizes': {
                'segment_embeddings': len(self._segment_embedding_cache),
                'sentiments': len(self._sentiment_cache),
                'text_segments': len(self._text_segment_cache),
                'similarities': len(self._similarity_cache)
            },
            'unique_segments_seen': self.cache_stats['unique_segments'],
            'total_segments_processed': self.cache_stats['total_segments_processed'],
            'segment_reuse_factor': (
                self.cache_stats['total_segments_processed'] / 
                max(self.cache_stats['unique_segments'], 1)
            ),
            'raw_stats': self.cache_stats,
            'device': str(self.device)
        }
        
        return stats
    
    def print_cache_report(self):
        """Print detailed cache performance report."""
        stats = self.get_cache_stats()
        
        print("\n📊 Cache Performance Report")
        print("=" * 60)
        print(f"Device: {stats['device']}")
        print(f"\n🎯 Hit Rates:")
        print(f"   Segment Embeddings: {stats['segment_embedding_hit_rate']:.1%}")
        print(f"   Sentiments: {stats['sentiment_hit_rate']:.1%}") 
        print(f"   Text Segmentation: {stats['text_segment_hit_rate']:.1%}")
        print(f"   Similarities: {stats['similarity_hit_rate']:.1%}")
        print(f"   Overall: {stats['overall_hit_rate']:.1%}")
        
        print(f"\n📈 Reuse Statistics:")
        print(f"   Unique segments: {stats['unique_segments_seen']:,}")
        print(f"   Total processed: {stats['total_segments_processed']:,}")
        print(f"   Reuse factor: {stats['segment_reuse_factor']:.2f}x")
        
        print(f"\n💾 Cache Sizes:")
        for cache_type, size in stats['cache_sizes'].items():
            print(f"   {cache_type}: {size:,} entries")
    
    def clear_cache(self, cache_type: Optional[str] = None):
        """Clear specific or all caches."""
        if cache_type == 'embeddings' or cache_type is None:
            self._segment_embedding_cache.clear()
            self._unique_segments_seen.clear()
        if cache_type == 'segments' or cache_type is None:
            self._text_segment_cache.clear()
        if cache_type == 'similarities' or cache_type is None:
            self._similarity_cache.clear()
        
        # Reset relevant stats
        if cache_type is None:
            self.cache_stats['unique_segments'] = 0
            self._unique_segments_seen.clear()

    def analyze_cache_potential(self,
                               reports_data: List[Tuple[str, str]],
                               segmentation_config,
                               sample_size: int = 100) -> Dict:
        """
        Analyze potential for cache reuse in the given dataset.
        
        Args:
            reports_data: List of reports to analyze
            segmentation_config: Segmentation configuration
            sample_size: Number of reports to sample
            
        Returns:
            Dictionary with analysis results
        """
        print("\n🔍 Analyzing cache reuse potential...")
        
        normalized_data = self._normalize_input_data(reports_data)
        sample = normalized_data[:sample_size]
        
        all_segments = []
        segment_counts = defaultdict(int)
        
        for _, text in sample:
            segments = self.segment_text(text, segmentation_config)
            for segment in segments:
                normalized = self._normalize_segment(segment)
                segment_counts[normalized] += 1
                all_segments.append(normalized)
        
        # Calculate statistics
        unique_segments = len(segment_counts)
        total_segments = len(all_segments)
        reused_segments = sum(1 for count in segment_counts.values() if count > 1)
        
        # Find most common
        common_segments = sorted(
            segment_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]
        
        analysis = {
            'total_segments': total_segments,
            'unique_segments': unique_segments,
            'reused_segments': reused_segments,
            'reuse_ratio': reused_segments / unique_segments if unique_segments > 0 else 0,
            'avg_segment_frequency': total_segments / unique_segments if unique_segments > 0 else 0,
            'most_common_segments': common_segments
        }
        
        print(f"   Total segments: {total_segments:,}")
        print(f"   Unique segments: {unique_segments:,}")
        print(f"   Reused segments: {reused_segments:,} ({analysis['reuse_ratio']:.1%})")
        print(f"   Avg frequency: {analysis['avg_segment_frequency']:.2f}")
        print(f"\n   Top 5 most common:")
        for segment, count in common_segments[:5]:
            print(f"      '{segment[:50]}...': {count}x")
        
        return analysis
    
    # ========================================================================
    # Utility Methods
    # ========================================================================
    
    def get_domain_summary(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Get summary statistics for each domain."""
        domain_stats = []
        
        for domain_name in self.domain_definitions.keys():
            domain_score_col = f'{domain_name}_domain_score'
            
            if domain_score_col in results_df.columns:
                domain_stats.append({
                    'domain': domain_name,
                    'mean_score': results_df[domain_score_col].mean(),
                    'std_score': results_df[domain_score_col].std(),
                    'min_score': results_df[domain_score_col].min(),
                    'max_score': results_df[domain_score_col].max(),
                    'n_items': len(self.domain_definitions[domain_name]),
                    'n_reports': len(results_df)
                })
        
        return pd.DataFrame(domain_stats)
    
    def _normalize_input_data(self, reports_data) -> List[Tuple[str, str]]:
        """Normalize various input formats to (report_id, text) tuples."""
        normalized = []
        
        for i, item in enumerate(reports_data):
            if isinstance(item, tuple) and len(item) == 2:
                normalized.append(item)
            elif isinstance(item, dict):
                report_id = item.get('eid', item.get('report_id', item.get('id', f'REPORT_{i:03d}')))
                text = item.get('text', item.get('report_text', ''))
                normalized.append((report_id, text))
            elif isinstance(item, str):
                normalized.append((f'REPORT_{i:03d}', item))
            else:
                print(f"Warning: Skipping item {i} with unexpected format: {type(item)}")
        
        return normalized
    
    def _normalize_segment(self, segment: str) -> str:
        """Normalize segment for better cache matching."""
        if not self.normalize_segments:
            return segment
        
        # Lowercase, strip, and normalize whitespace
        normalized = segment.lower().strip()
        normalized = ' '.join(normalized.split())
        return normalized
    
    def _get_segment_hash(self, segment: str) -> str:
        """Get hash for a single segment."""
        normalized = self._normalize_segment(segment)
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _get_text_hash(self, text: str) -> str:
        """Generate hash for text content."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def _get_config_hash(self, config) -> str:
        """Generate hash for configuration objects."""
        config_dict = self._config_to_dict(config)
        config_str = str(sorted(config_dict.items()))
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def _config_to_dict(self, config):
        """Convert config objects to dictionaries for serialization."""
        if hasattr(config, '__dict__'):
            return {k: v for k, v in config.__dict__.items() if not k.startswith('_')}
        return config
    
# ============================================================================
# Parallel Processing Worker Function
# ============================================================================

def _process_batch_worker(batch, domain_definitions, embedding_model_name,
                          seg_config_dict, thresh_config_dict, comp_config_dict):
    """
    Worker function for parallel batch processing.
    
    This runs in a separate process, so it needs to reconstruct the processor
    and configuration objects.
    """
    from dynclim.features.safety.toolkit import (
        SegmentationConfig, ThresholdConfig, CompositeConfig
    )
    
    # Reconstruct config objects
    seg_config = SegmentationConfig(**seg_config_dict)
    thresh_config = ThresholdConfig(**thresh_config_dict)
    comp_config = CompositeConfig(**comp_config_dict)
    
    # Create processor (caching disabled in workers to avoid conflicts)
    processor = Processor(
        domain_definitions, 
        embedding_model_name,
        cache_embeddings=False,  # Disable in workers
        cache_segments=False
    )
    
    # Process batch
    results = []
    for report_id, text in batch:
        result = processor.process_single_report(
            report_id, text, seg_config, thresh_config, comp_config
        )
        results.append(result)
    
    return results


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    from dynclim.features.safety.toolkit import (
        SegmentationConfig, ThresholdConfig, CompositeConfig
    )
    
    # Define domains
    domain_definitions = {
        "LSC": [  # Leadership Safety Climate
            "Management acts quickly to correct safety problems",
            "Management provides adequate safety training"
        ],
        "COMM": [  # Communication
            "Safety procedures are clearly communicated",
            "Employees are encouraged to report safety concerns"
        ],
        "ENG": [  # Engagement
            "Team members look out for each other's safety",
            "Employees feel comfortable reporting near misses"
        ]
    }
    
    # Test data with repeated content to demonstrate caching
    reports_data = [
        ("R1", "Management acts quickly to fix issues. Communication is clear."),
        ("R2", "Management acts quickly on problems. Training is provided."),
        ("R3", "Communication is clear. Management provides good training."),
        ("R4", "Management acts quickly. Communication could be better."),
        ("R5", "Training is provided regularly. Communication is clear.")
    ]
    
    # Create processor with caching enabled
    processor = Processor(
        domain_definitions,
        cache_embeddings=True,
        cache_segments=True,
        normalize_segments=True,
        use_acceleration=True
    )
    
    # Analyze cache potential
    print("\n" + "="*60)
    print("CACHE POTENTIAL ANALYSIS")
    print("="*60)
    
    seg_config = SegmentationConfig(method='sentence', min_length=10)
    processor.analyze_cache_potential(reports_data, seg_config)
    
    # Test with multiple configurations to show cache reuse
    print("\n" + "="*60)
    print("TESTING CACHE REUSE ACROSS CONFIGURATIONS")
    print("="*60)
    
    configs = [
        (
            SegmentationConfig(method='sentence', min_length=10),
            ThresholdConfig(method='top_k', k=3),
            CompositeConfig(method='attention_weight')
        ),
        (
            SegmentationConfig(method='sentence', min_length=10),
            ThresholdConfig(method='top_k', k=5),
            CompositeConfig(method='attention_weight')
        ),
        (
            SegmentationConfig(method='sentence', min_length=10),
            ThresholdConfig(method='hard', threshold=0.5),
            CompositeConfig(method='weighted_average')
        )
    ]
    
    for i, (seg, thresh, comp) in enumerate(configs, 1):
        print(f"\n--- Configuration {i} ---")
        
        # Use batch-optimized processing
        results_df = processor.process_reports_batch_optimized(
            reports_data, seg, thresh, comp, use_parallel=False
        )
        
        # print(f"\nResults preview:")
        # print(results_df[['report_id', 'overall_composite_score']].round(3))
        
        processor.print_cache_report()
    
    # Show domain summary
    print("\n" + "="*60)
    print("DOMAIN SUMMARY")
    print("="*60)
    domain_summary = processor.get_domain_summary(results_df)
    print(domain_summary.round(3))
    
    print("\n✅ Notice how cache hit rates improve across configurations!")
    print("   This demonstrates the benefit of segment-level caching.")