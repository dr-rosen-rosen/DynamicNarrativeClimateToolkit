"""
Enhanced Cached Experiment Runner with Checkpointing and Performance Optimizations
================================================================================

CHANGES MADE:
1. Added configuration-level checkpointing to save progress incrementally
2. Added resume capability from checkpoints after interruption
3. Added optimized configuration ordering for better cache utilization
4. Added chunked processing for large datasets
5. Added progress monitoring with tqdm
6. Added memory management with periodic cache clearing
7. Added CSV/HDF5 export options for results
8. Added comparative experiment mode with performance metrics

NEW DEPENDENCIES:
- tqdm (for progress bars)
- h5py (optional, for HDF5 export)
"""

import pandas as pd
import numpy as np
import os, json, hashlib, itertools, tempfile, glob, re
from typing import List, Dict, Any, Tuple, Optional
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
import warnings
import pickle
from datetime import datetime
import time
from tqdm import tqdm
from dataclasses import is_dataclass, asdict
warnings.filterwarnings('ignore')

from dynclim.features.safety.unified_processor import Processor
from dynclim.features.safety.toolkit import SegmentationConfig, ThresholdConfig, CompositeConfig
from dynclim.features.safety.seg_core import SegmentationStore
from tokenizers import Tokenizer
# --- top-level helpers (module scope) ---

def _to_plain_dict(obj: Any) -> Dict[str, Any]:
    # tolerant “to dict” (works for dataclasses, pydantic-ish, or simple objects)
    from dataclasses import is_dataclass, asdict
    if obj is None:
        return {}
    try:
        if is_dataclass(obj):
            return asdict(obj)
    except Exception:
        pass
    if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
        return obj.to_dict()
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in vars(obj).items() if not k.startswith("_")}
    return dict(obj) if isinstance(obj, dict) else {"value": obj}

def _canon_payload(embedding_model, seg_config, thresh_config, comp_config, sent_method) -> Dict[str, Any]:
    return {
        "embedding_model": embedding_model,
        "seg":   _to_plain_dict(seg_config),
        "thresh":_to_plain_dict(thresh_config),
        "comp":  _to_plain_dict(comp_config),
        "sent_method": sent_method,
    }

def _combo_signature(payload: Dict[str, Any]) -> str:
    # stable, compact signature for deterministic ID assignment
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))

def _atomic_write_text(final_path: str, text: str) -> None:
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(suffix=".part", dir=os.path.dirname(final_path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp_path, final_path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

def _atomic_write_df_parquet(final_path: str, df: pd.DataFrame, **kwargs):
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".part", dir=os.path.dirname(final_path))
    os.close(fd)
    try:
        df.to_parquet(tmp, index=False, **kwargs)
        os.replace(tmp, final_path)
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass

def _atomic_write_df_gz(final_path: str, df: pd.DataFrame) -> None:
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(suffix=".part", dir=os.path.dirname(final_path))
    os.close(fd)
    try:
        df.to_csv(tmp_path, index=False, compression="gzip")
        os.replace(tmp_path, final_path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

def _fmt_val(v):
    if isinstance(v, float):
        return f"{v:.6g}"
    if isinstance(v, bool):
        return "1" if v else "0"
    return str(v)

def _sanitize(s: str) -> str:
    return (s.replace("/", "_")
             .replace("\\", "_")
             .replace(":", "-")
             .replace(" ", "")
             .replace("|", "-")
             .replace(",", "-")
             .replace("=", "-")
             .replace("__", "_"))

def _to_dict(obj):
    if obj is None:
        return {}
    try:
        if is_dataclass(obj):
            return asdict(obj)
    except Exception:
        pass
    if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
        return obj.to_dict()
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in vars(obj).items() if not k.startswith("_")}
    return dict(obj) if isinstance(obj, dict) else {"value": obj}

def _method_block(prefix: str, d: dict, order_hint: Optional[List[str]] = None) -> str:
    d = {k: v for k, v in d.items() if v is not None}
    method = d.pop("method", "unspecified")
    parts = [prefix, method]

    ordered_keys: List[str] = []
    if order_hint:
        ordered_keys.extend([k for k in order_hint if k in d])
    ordered_keys.extend(sorted(k for k in d.keys() if k not in (order_hint or [])))

    for k in ordered_keys:
        parts.append(f"{k}{_fmt_val(d[k])}")
    return _sanitize("_".join(parts))

class ExperimentRunner:
    """
    Experiment runner with integrated caching support and checkpointing for improved 
    performance and reliability.
    """
    
    def __init__(self, 
                 domain_definitions: Dict[str, List[str]],
                 use_caching: bool = True,
                 cache_embeddings: bool = True,
                 cache_segments: bool = True,
                 cache_results: bool = False,
                #  embedding_model_name: str = 'all-MiniLM-L6-v2',
                 embedding_model_names: List[str] = None,
                 checkpoint_dir: str = "checkpoints",
                 use_acceleration: bool = True,
                 device_pref: str = 'auto',
                 use_parallel: bool = False,
                 store: SegmentationStore = SegmentationStore("seg_cache.sqlite"),
                 tok: Optional[Tokenizer] = Tokenizer.from_pretrained("bert-base-uncased")
                 ):
        """
        Initialize experiment runner with optional caching and checkpointing.
        
        Args:
            domain_definitions: Dict mapping domain names to lists of scale items
            use_caching: Whether to use the cached processor
            cache_embeddings: Whether to cache text embeddings
            cache_segments: Whether to cache segmentation results
            cache_results: Whether to cache full processing results
            embedding_model_name: Name of sentence transformer model
            embedding_model_names: List of embedding model names to test.
                              If None, defaults to ['all-MiniLM-L6-v2']
            checkpoint_dir: Directory for saving checkpoints
            use_acceleration: Whether to use GPU/MPS acceleration
            device_pref: Device for acceleration ('auto', 'cuda', 'mps', 'cpu'
            use_parallel: Whether to use parallel processing (NOT recommended with caching)
                When caching is enabled, parallel processing actually slows things down
                because workers can't share caches. Default: False)
        """
        # Store embedding models to test
        if embedding_model_names is None:
            self.embedding_model_names = ['all-MiniLM-L6-v2']
        else:
            self.embedding_model_names = embedding_model_names
        
        # Store other config for processor creation
        self.domain_definitions = domain_definitions
        self.use_caching = use_caching
        self.cache_embeddings = cache_embeddings
        self.cache_segments = cache_segments
        self.cache_results = cache_results
        self.use_acceleration = use_acceleration
        self.device_pref = device_pref
        self.checkpoint_dir = checkpoint_dir
        self.use_parallel = use_parallel
        # shared, read-mostly cache on disk
        self.store = SegmentationStore("seg_cache.sqlite")
        # optional, only needed when you use token-based min_length
        self.tok = Tokenizer.from_pretrained("bert-base-uncased")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        if use_caching and use_parallel:
            print("WARNING: Parallel processing with caching is not recommended!")
            print("   Workers cannot share caches, which negates the main performance benefit.")
            print("   Consider setting use_parallel=False for better performance.")
        # Key: sentiment_method, Value: cache dict {segment_hash_method: score}
        self._sentiment_caches = {}
        self.processor = None  
        self.results = []
        self.experiment_times = []
    
    # ---- Registry + path helpers ----
    def _experiment_root(self, experiment_name: str) -> str:
        return os.path.join("checkpoints", experiment_name)

    def _registry_path(self, experiment_name: str) -> str:
        return os.path.join(self._experiment_root(experiment_name), "config_registry.csv")

    def _paths_for_config(self, experiment_name: str, config_id: int) -> Tuple[str, str, str, str]:
        """
        Returns (cfg_dir, results_path, meta_path, log_path)
        Layout:
          checkpoints/<experiment_name>/_cfg/<config_id>/{results.csv.gz, meta.json, run.log}
        """
        cfg_dir = os.path.join(self._experiment_root(experiment_name), "_cfg", str(int(config_id)))
        results = os.path.join(cfg_dir, "results")
        meta    = os.path.join(cfg_dir, "meta.json")
        # logp    = os.path.join(cfg_dir, "run.log")
        return cfg_dir, results, meta#, logp

    def _build_or_load_registry(self, experiment_name: str, config_combinations):
        """
        config_combinations: iterable of your dict combos with keys:
        'embedding_model', 'segmentation', 'threshold', 'composite', 'sentiment_method'
        Returns a DataFrame with columns ['config_id','signature', ...flattened params...]
        """
        os.makedirs(self._experiment_root(experiment_name), exist_ok=True)

        rows = []
        for combo in config_combinations:
            payload = _canon_payload(
                combo['embedding_model'],
                combo['segmentation'],
                combo['threshold'],
                combo['composite'],
                combo['sentiment_method'],
            )
            sig = _combo_signature(payload)
            row = {"signature": sig,
                "embedding_model": combo['embedding_model'],
                "sent_method": combo['sentiment_method']}
            seg_d = payload['seg']; th_d = payload['thresh']; comp_d = payload['comp']
            row.update({f"seg__{k}": seg_d.get(k) for k in sorted(seg_d.keys())})
            row.update({f"th__{k}": th_d.get(k) for k in sorted(th_d.keys())})
            row.update({f"comp__{k}": comp_d.get(k) for k in sorted(comp_d.keys())})
            rows.append(row)

        reg = pd.DataFrame(rows)
        if reg.empty:
            reg = pd.DataFrame(columns=["config_id","signature"])
        else:
            reg = reg.sort_values("signature").reset_index(drop=True)
            reg.insert(0, "config_id", reg.index.astype(int))

        # --- Drop zero-variation columns (preserve in sidecar) ---
        const_cols = [c for c in reg.columns
                      if c not in ('config_id', 'signature')
                      and reg[c].nunique(dropna=False) <= 1]
        if const_cols:
            constants = {}
            for c in const_cols:
                val = reg[c].iloc[0]
                # Convert numpy/pandas types to JSON-safe values
                if pd.isna(val):
                    constants[c] = None
                elif isinstance(val, (np.integer,)):
                    constants[c] = int(val)
                elif isinstance(val, (np.floating,)):
                    constants[c] = float(val)
                elif isinstance(val, (np.bool_,)):
                    constants[c] = bool(val)
                else:
                    constants[c] = str(val)
            meta_path = os.path.join(self._experiment_root(experiment_name),
                                     "registry_constants.json")
            os.makedirs(os.path.dirname(meta_path), exist_ok=True)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(constants, f, indent=2, sort_keys=True)
            reg = reg.drop(columns=const_cols)
            print(f"   Registry: dropped {len(const_cols)} constant columns → "
                  f"{len(reg.columns)} remaining (constants in registry_constants.json)")

        reg_path = self._registry_path(experiment_name)
        reg.to_csv(reg_path, index=False)
        return reg

    def _config_id_for_combo(self, registry_df: pd.DataFrame, combo: dict) -> int:
        payload = _canon_payload(
            combo['embedding_model'],
            combo['segmentation'],
            combo['threshold'],
            combo['composite'],
            combo['sentiment_method'],
        )
        sig = _combo_signature(payload)
        row = registry_df.loc[registry_df['signature'] == sig]
        if row.empty:
            raise RuntimeError("Config signature not found in registry; did the grid change?")
        return int(row.iloc[0]['config_id'])

    # # ---- Convenience: get config_id for a combo ----
    # def _config_id_for_combo(self, registry_df: pd.DataFrame, combo: Tuple[Any, Any, Any, Any, Any]) -> int:
    #     payload = _canon_payload(*combo)
    #     sig = _combo_signature(payload)
    #     row = registry_df.loc[registry_df["signature"] == sig]
    #     if row.empty:
    #         raise RuntimeError("Signature not found in registry (registry out of sync).")
    #     return int(row.iloc[0]["config_id"])
    
    def create_config_combinations(self, 
                                segmentation_methods: Optional[List[str]] = None,
                                threshold_methods: Optional[List[str]] = None,
                                composite_methods: Optional[List[str]] = None,
                                embedding_models: Optional[List[str]] = None,
                                sentiment_methods: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    
        """
        Create configuration combinations with optional filtering.
        
        Args:
            segmentation_methods: List of segmentation methods to include (None = all)
            threshold_methods: List of threshold methods to include (None = all)
            composite_methods: List of composite methods to include (None = all)
            embedding_models: List of embedding models to include (overrides __init__ list)
        
        Returns:
            List of configuration combinations
        """
            # Use provided embedding models or fall back to instance list
        if embedding_models is None:
            embedding_models = self.embedding_model_names
        # Use provided sentiment methods or default
        if sentiment_methods is None:
            sentiment_methods = ['vader']  # Default to just vader
        # Define all possible configurations
        all_segmentation_configs = [
            SegmentationConfig(method='sentence', min_length=5, backend='auto'),
            # SegmentationConfig(method='sentence', min_length=10, backend='auto'),
            # SegmentationConfig(method='clause', min_length=5),
            # SegmentationConfig(method='clause', min_length=10)
            # SegmentationConfig(method='sliding_window', window_size=50, overlap=25, min_length=10),
            # SegmentationConfig(method='sliding_window', window_size=100, overlap=50, min_length=10),
        ]
        
        all_threshold_configs = [
            # --- Top-K thresholders ---
            ThresholdConfig(method='top_k', k=3),
            ThresholdConfig(method='top_k', k=5),
            # ThresholdConfig(method='top_k', k=10),

            # --- Adaptive (median + λ·MAD) ---
            # ThresholdConfig(method='adaptive'),                       # uses defaults (λ=0.5)
            # ThresholdConfig(method='adaptive', lambda_=0.25),
            # ThresholdConfig(method='adaptive', lambda_=0.5)

            # --- Quantile (robust, scale-free) ---
            # ThresholdConfig(method='quantile', q=0.5),
            # ThresholdConfig(method='quantile', q=0.6),
            # ThresholdConfig(method='quantile', q=0.7),
            # ThresholdConfig(method='diversity_top_k', k=10, lambda_=0.7),
            
        ]
        all_composite_configs = [
            # --- Baseline weighting ---
            # CompositeConfig(method='weighted_average', similarity_weight=0.3, sentiment_weight=0.7, 
            #                 apply_over="post_threshold"),
            CompositeConfig(method='weighted_average', similarity_weight=0.5, sentiment_weight=0.5, 
                            apply_over="post_threshold"),
            # CompositeConfig(method='weighted_average', similarity_weight=0.7, sentiment_weight=0.3, 
            #                 apply_over="post_threshold"),
            # CompositeConfig(method='weighted_average', similarity_weight=0.3, sentiment_weight=0.7, 
            #                 apply_over="all"),
            CompositeConfig(method='weighted_average', similarity_weight=0.5, sentiment_weight=0.5, 
                            apply_over="all"),
            # CompositeConfig(method='weighted_average', similarity_weight=0.7, sentiment_weight=0.3, 
            #                 apply_over="all"),
            
            # CompositeConfig(method='separate_features'),

            # --- Softmax attention (temperature τ) ---
            CompositeConfig(method='attention_weight', temperature=0.15, apply_over="post_threshold"),
            # CompositeConfig(method='attention_weight', temperature=0.25, apply_over="post_threshold"),
            # CompositeConfig(method='attention_weight', temperature=0.50, apply_over="post_threshold"),
            CompositeConfig(method='attention_weight', temperature=1.00, apply_over="post_threshold"),
            CompositeConfig(method='attention_weight', temperature=0.15, apply_over="all"),
            # CompositeConfig(method='attention_weight', temperature=0.25, apply_over="all"),
            # CompositeConfig(method='attention_weight', temperature=0.50, apply_over="all"),
            CompositeConfig(method='attention_weight', temperature=1.00, apply_over="all"),

            # --- Power-law attention (exponent p) ---
            CompositeConfig(method='power_attention', power=0.5, apply_over="post_threshold"),
            # CompositeConfig(method='power_attention', power=1.0, apply_over="post_threshold"),
            # CompositeConfig(method='power_attention', power=1.5, apply_over="post_threshold"),
            CompositeConfig(method='power_attention', power=2.0, apply_over="post_threshold"),
            CompositeConfig(method='power_attention', power=0.5, apply_over="all"),
            # CompositeConfig(method='power_attention', power=1.0, apply_over="all"),
            # CompositeConfig(method='power_attention', power=1.5, apply_over="all"),
            CompositeConfig(method='power_attention', power=2.0, apply_over="all"),

            # --- Soft-over-all attention (for sensitivity runs) ---
            # CompositeConfig(method='attention_weight', temperature=0.25, apply_over="all"),
            # CompositeConfig(method='power_attention',  power=0.5,  apply_over="all"),
            

        ]
        
        # Filter configurations if specified
        segmentation_configs = [c for c in all_segmentation_configs 
                               if segmentation_methods is None or c.method in segmentation_methods]
        threshold_configs = [c for c in all_threshold_configs 
                            if threshold_methods is None or c.method in threshold_methods]
        composite_configs = [c for c in all_composite_configs 
                            if composite_methods is None or c.method in composite_methods]
        
        # Create combinations INCLUDING embedding models
        # Split composites by threshold dependency to avoid duplicate runs
        threshold_dependent   = [c for c in composite_configs 
                                 if getattr(c, 'apply_over', 'post_threshold') == 'post_threshold']
        threshold_independent = [c for c in composite_configs 
                                 if getattr(c, 'apply_over', 'post_threshold') != 'post_threshold']

        # Sentinel threshold for configs that ignore thresholding entirely.
        # top_k with k=99999 selects all segments, so the mask is a no-op.
        _SENTINEL_THRESHOLD = ThresholdConfig(method='top_k', k=99999)

        combinations = []

        # Full crossing for threshold-dependent composites
        for embedding_model, seg_config, thresh_config, comp_config, sent_method in itertools.product(
            embedding_models,
            segmentation_configs, 
            threshold_configs, 
            threshold_dependent,
            sentiment_methods 
        ):
            combinations.append({
                'embedding_model': embedding_model, 
                'segmentation': seg_config,
                'threshold': thresh_config,
                'composite': comp_config,
                'sentiment_method': sent_method,
                'config_id': self._generate_config_id(
                    embedding_model, seg_config, thresh_config, comp_config, sent_method
                )
            })

        # Single sentinel threshold for threshold-independent composites
        for embedding_model, seg_config, comp_config, sent_method in itertools.product(
            embedding_models,
            segmentation_configs,
            threshold_independent,
            sentiment_methods
        ):
            combinations.append({
                'embedding_model': embedding_model,
                'segmentation': seg_config,
                'threshold': _SENTINEL_THRESHOLD,
                'composite': comp_config,
                'sentiment_method': sent_method,
                'config_id': self._generate_config_id(
                    embedding_model, seg_config, _SENTINEL_THRESHOLD, comp_config, sent_method
                )
            })
        
        return combinations
    def _create_processor(self, embedding_model: str, sentiment_method: str) -> Processor:
        """
        Create a new processor with the specified models.
        Injects persistent sentiment cache to preserve sentiments across processor recreations.
        
        Args:
            embedding_model: Name of embedding model
            sentiment_method: Name of sentiment method
            
        Returns:
            Configured Processor instance
        """
        processor = Processor(
            domain_definitions=self.domain_definitions,
            embedding_model_name=embedding_model,
            sentiment_method=sentiment_method,
            cache_embeddings=self.cache_embeddings if self.use_caching else False,
            cache_segments=self.cache_segments if self.use_caching else False,
            cache_sentiments=True if self.use_caching else False,
            normalize_segments=True if self.use_caching else False,
            use_acceleration=self.use_acceleration,
            device_pref=self.device_pref,
            batch_size=64
        )
        
        processor.store = self.store
        processor.tok = self.tok

        # Inject persistent sentiment cache for this method
        if self.use_caching:
            if sentiment_method not in self._sentiment_caches:
                self._sentiment_caches[sentiment_method] = {}
            
            processor._sentiment_cache = self._sentiment_caches[sentiment_method]
            print(f"   Injected sentiment cache: {len(self._sentiment_caches[sentiment_method]):,} entries")
        
        return processor
    
    def optimize_config_order(self, config_combinations: List[Dict]) -> List[Dict]:
        """
        Order configurations to maximize cache reuse.
        GROUP BY EMBEDDING MODEL FIRST, then sentiment then others - critical for cache efficiency!
        """
        return sorted(config_combinations, key=lambda x: (
            x['embedding_model'],  
            x['sentiment_method'],
            x['segmentation'].method,
            x['segmentation'].min_length,
            getattr(x['segmentation'], 'window_size', 0),
            getattr(x['segmentation'], 'overlap', 0),
            x['threshold'].method,
            getattr(x['threshold'], 'threshold', 0),
            getattr(x['threshold'], 'k', 0),
            x['composite'].method
        ))
    
    def run_experiment(self, 
                      reports_data: List[Tuple[str, str]],
                      ground_truth: Dict[str, float] = None,
                      experiment_name: str = "safety_features_experiment",
                      max_workers: int = 4,
                      config_filter: Optional[Dict] = None,
                      show_progress: bool = True,
                      show_cache_stats: bool = True,
                      resume_from_checkpoint: bool = True,
                      save_interval: int = 1,
                      chunk_size: Optional[int] = None,
                      save_full_results: bool = True,
                      verbose_results: bool = False,
                      use_parallel: Optional[bool] = None) -> pd.DataFrame:
        """
        Run experiment across configuration combinations with checkpointing.
        
        Args:
            reports_data: List of (report_id, text) tuples
            ground_truth: Optional dict of report_id -> ground truth score
            experiment_name: Name for this experiment
            max_workers: Number of parallel workers
            config_filter: Dict with keys 'segmentation_methods', 'threshold_methods', 'composite_methods'
            show_progress: Whether to show progress updates
            show_cache_stats: Whether to show cache statistics during run
            resume_from_checkpoint: Whether to resume from existing checkpoint
            save_interval: Save checkpoint every N configurations
            chunk_size: Process reports in chunks of this size (None = process all at once)
            save_full_results: Whether to save full report-level results (not just summaries)
            verbose_results: If True, save all columns including segment-level diagnostics.
                           If False, save only item/domain scores and processing metadata.
                           Default False for compact output suitable for analysis runs.
        
        Returns:
            DataFrame with results for all configurations
        """
        use_parallel_setting = use_parallel if use_parallel is not None else self.use_parallel
        # Setup directories for full results
        self.full_results_dir = os.path.join(self._experiment_root(experiment_name), "_cfg")
        os.makedirs(self.full_results_dir, exist_ok=True)
        
        # Check for and load checkpoint
        checkpoint_file = os.path.join(self.checkpoint_dir, f"{experiment_name}_checkpoint.pkl")
        completed_configs = set()
        results = []  # This will store summary statistics
        
        if resume_from_checkpoint and os.path.exists(checkpoint_file):
            checkpoint_data = self._load_checkpoint(checkpoint_file)
            if checkpoint_data:
                completed_configs = checkpoint_data['completed_configs']
                results = checkpoint_data['results']
                print(f"📂 Resuming from checkpoint: {len(completed_configs)} configs already processed")
        
        # 1) Get configurations
        if config_filter:
            config_combinations = self.create_config_combinations(**config_filter)
        else:
            config_combinations = self.create_config_combinations()
        
        # Optimize configuration order for better cache usage
        config_combinations = self.optimize_config_order(config_combinations)
        registry_df = self._build_or_load_registry(experiment_name, config_combinations)
        for combo in config_combinations:
            combo['config_id'] = self._config_id_for_combo(registry_df, combo)

        # Filter out completed configurations
        pending_configs = [c for c in config_combinations if c['config_id'] not in completed_configs]
        
        print(f"\n🔬 Running experiment '{experiment_name}'")
        print(f"   Total configurations: {len(config_combinations)}")
        print(f"   Pending configurations: {len(pending_configs)}")
        print(f"   Reports: {len(reports_data)}")
        print(f"   Embedding models: {len(set(c['embedding_model'] for c in pending_configs))}")
        print(f"   Parallel processing: {use_parallel_setting}")
        if self.use_caching:
            print(f"   Caching: Enabled (embeddings={self.cache_embeddings}, segments={self.cache_segments})")

        experiment_start = time.time()
        config_iterator = tqdm(pending_configs, desc="Configurations", disable=not show_progress)
        # Track the current segmentation configuration so we only clear cheap caches
        # when the segmentation actually changes
        prev_seg_key = None
        current_embedding_model = None
        current_sentiment_method = None

        for i, config_combo in enumerate(config_iterator):
            try:
                cfg_dir, results_base, meta_path = self._paths_for_config(experiment_name, config_combo['config_id'])
                os.makedirs(cfg_dir, exist_ok=True)
                if resume_from_checkpoint and os.path.exists(results_base):
                    # mark done and continue
                    completed_configs.add(config_combo['config_id'])
                    continue

                embedding_changed = (
                    self.processor is None or 
                    config_combo['embedding_model'] != current_embedding_model
                )
                sentiment_changed = (
                    self.processor is not None and 
                    config_combo['sentiment_method'] != current_sentiment_method
                )
                
                # Handle model switches
                if embedding_changed:
                    # Embedding model changed - must recreate processor
                    if self.processor is None:
                        print(f"\n🔄 Initializing processor:")
                    else:
                        print(f"\n🔄 Switching embedding model: {current_embedding_model} → {config_combo['embedding_model']}")
                        
                        # Show cache stats before switching
                        if show_cache_stats and self.use_caching:
                            stats = self.processor.get_cache_stats()
                            print(f"   Previous model cache stats:")
                            print(f"      Embeddings: {len(self.processor._segment_embedding_cache):,} cached, "
                                f"{stats.get('segment_embedding_hit_rate', 0):.1%} hit rate")
                            print(f"      Sentiments: {len(self.processor._sentiment_cache):,} cached, "
                                f"{stats.get('sentiment_hit_rate', 0):.1%} hit rate")
                        
                        # Delete old processor
                        del self.processor
                    
                    print(f"   Embedding model: {config_combo['embedding_model']}")
                    print(f"   Sentiment method: {config_combo['sentiment_method']}")
                    
                    # Create new processor with persistent sentiment cache

                    self.processor = self._create_processor(
                        config_combo['embedding_model'],
                        config_combo['sentiment_method']
                    )
                    
                    current_embedding_model = config_combo['embedding_model']
                    current_sentiment_method = config_combo['sentiment_method']
                
                elif sentiment_changed:
                    # Only sentiment method changed - switch without recreating processor
                    print(f"\n🔄 Switching sentiment method: {current_sentiment_method} → {config_combo['sentiment_method']}")
                    
                    # Show current cache stats
                    if show_cache_stats and self.use_caching:
                        stats = self.processor.get_cache_stats()
                        print(f"   Current sentiment cache: {len(self.processor._sentiment_cache):,} cached, "
                            f"{stats.get('sentiment_hit_rate', 0):.1%} hit rate")
                    
                    # Ensure cache for new method exists
                    if config_combo['sentiment_method'] not in self._sentiment_caches:
                        self._sentiment_caches[config_combo['sentiment_method']] = {}
                    
                    # Switch sentiment method with persistent cache
                    self.processor.switch_sentiment_method(
                        config_combo['sentiment_method'],
                        external_cache=self._sentiment_caches[config_combo['sentiment_method']]
                    )
                    
                    current_sentiment_method = config_combo['sentiment_method']
                
                config_start = time.time()

                # Process reports (rest of the code stays the same)
                if chunk_size and len(reports_data) > chunk_size:
                    config_results_df = self._process_config_chunked(
                        reports_data, config_combo, chunk_size, max_workers
                    )
                else:
                    config_results_df = self.processor.process_reports_batch_optimized(
                        reports_data,
                        config_combo['segmentation'],
                        config_combo['threshold'],
                        config_combo['composite'],
                        use_parallel=self.use_parallel,
                        max_workers=max_workers if self.use_parallel else None
                    )
                config_time = time.time() - config_start
                
                # # Save full report-level results if requested
                #if save_full_results:
                #     self._save_full_config_results(config_combo['config_id'], config_results_df)
                # Ensure config_id in the per-report frame
                if 'config_id' not in config_results_df.columns:
                    config_results_df.insert(0, 'config_id', int(config_combo['config_id']))

                # --- Compact vs verbose results ---
                if verbose_results:
                    save_df = config_results_df
                else:
                    keep_patterns = [
                        r'^report_id$', r'^config_id$', r'^processing_',
                        r'_final$', r'_sent$', r'_strength$', r'_n_segments$',
                        r'_domain_score$', r'^overall_', r'^error',
                    ]
                    keep_re = re.compile('|'.join(keep_patterns))
                    compact_cols = [c for c in config_results_df.columns if keep_re.search(c)]
                    save_df = config_results_df[compact_cols].copy()

                # Downcast float64 → float32 for space savings
                f64_cols = save_df.select_dtypes(include=['float64']).columns
                if len(f64_cols):
                    if save_df is config_results_df:
                        save_df = save_df.copy()
                    save_df[f64_cols] = save_df[f64_cols].astype('float32')

                # Save per-config results (compressed parquet only)
                _atomic_write_df_parquet(
                    results_base + ".parquet", save_df,
                    compression='zstd'
                )

                # Save sidecar metadata for traceability
                meta_payload = {
                    "experiment_name": experiment_name,
                    "config_id": int(config_combo['config_id']),
                    "params": _canon_payload(
                        config_combo['embedding_model'],
                        config_combo['segmentation'],
                        config_combo['threshold'],
                        config_combo['composite'],
                        config_combo['sentiment_method'],
                    ),
                    "timing_sec": config_time,
                }
                os.makedirs(os.path.dirname(meta_path), exist_ok=True)
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta_payload, f, indent=2, sort_keys=True)
                # Compile summary results
                config_result = self._compile_results(
                    config_combo,
                    config_results_df,
                    ground_truth,
                    experiment_name,
                    config_time
                )
                results.append(config_result)
                completed_configs.add(config_combo['config_id'])
                
                # Update progress bar with cache stats
                if show_cache_stats and self.use_caching:
                    stats = self.processor.get_cache_stats()
                    config_iterator.set_postfix({
                        'cache_hit': f"{stats['overall_hit_rate']:.1%}",  # Changed from 'hit_rate'
                        'time': f"{config_time:.1f}s"
                    })
                
                # Save checkpoint periodically
                if (i + 1) % save_interval == 0:
                    self._save_checkpoint(checkpoint_file, completed_configs, results)
                
                # Memory management - clear caches periodically
                # Only clear cheap/derived caches when segmentation changes
                if self.use_caching:
                    seg = config_combo['segmentation']  # or config['segmentation'] if that's your variable
                    seg_key = (
                        getattr(seg, 'method', None),
                        getattr(seg, 'min_length', None),
                        getattr(seg, 'window_size', None),
                        getattr(seg, 'overlap', None),
                        getattr(seg, 'max_len', None),
                        getattr(seg, 'sentence_model', None),
                    )

                    if prev_seg_key is not None and seg_key != prev_seg_key:
                        # Segments will differ; embeddings are still expensive—keep them.
                        # Clear only cheap, derived caches that depend on segments/domains.
                        if hasattr(self.processor, 'clear_cache'):
                            # Clear similarities (cheap to recompute)
                            self.processor.clear_cache('similarities')
                            # If your sentiments depend on segments and are cheap for you, you *may* clear them.
                            # Leave commented unless you know you need it:
                            # self.processor.clear_cache('sentiments')

                    prev_seg_key = seg_key
                
            except Exception as e:
                print(f"\n❌ Error with config {config_combo['config_id']}: {str(e)}")
                self._save_checkpoint(checkpoint_file, completed_configs, results)
                continue
        
        # Final checkpoint save
        self._save_checkpoint(checkpoint_file, completed_configs, results)
        
        experiment_time = time.time() - experiment_start
        self.experiment_times.append(experiment_time)
        
        print(f"\n✅ Experiment complete!")
        print(f"   Total time: {experiment_time:.1f} seconds")
        print(f"   Average time per config: {experiment_time/len(pending_configs):.2f} seconds")
        
        if self.use_caching:
            print("\n📊 Final Cache Statistics:")
            if self.processor is not None:
                self.processor.print_cache_report()
        
        results_df = pd.DataFrame(results)
        self.results.append(results_df)
        
        # CHANGE: Option to export results
        self._export_results(results_df, experiment_name)
        
        # CHANGE: Print info about full results
        if save_full_results:
            print(f"\n📁 Full report-level results saved in: {self.full_results_dir}")
            print(f"   Each configuration's full results available as separate files")
        
        return results_df
    
    def _process_config_chunked(self, reports_data: List[Tuple[str, str]], 
                               config_combo: Dict, chunk_size: int, 
                               max_workers: int,
                               use_parallel: bool) -> pd.DataFrame:
        """
        Process large dataset in chunks for a single configuration.
        
        CHANGE: New method for memory-efficient processing
        """
        chunk_results = []
        n_chunks = (len(reports_data) + chunk_size - 1) // chunk_size
        
        chunk_iterator = tqdm(range(0, len(reports_data), chunk_size), 
                            desc="Chunks", leave=False, total=n_chunks)
        
        for chunk_start in chunk_iterator:
            chunk_end = min(chunk_start + chunk_size, len(reports_data))
            chunk = reports_data[chunk_start:chunk_end]
            
            # Process chunk
            chunk_df = self.processor.process_reports(
                chunk,
                config_combo['segmentation'],
                config_combo['threshold'],
                config_combo['composite'],
                use_parallel=use_parallel,
                max_workers=max_workers if use_parallel else None
            )
            chunk_results.append(chunk_df)
        
        # Combine chunk results
        return pd.concat(chunk_results, ignore_index=True)
    
    def _save_full_config_results(self, config_id: str, results_df: pd.DataFrame):
        """
        Save full report-level results for a configuration.
        
        CHANGE: New method to save complete report-level data
        """
        # Save as compressed CSV
        csv_file = os.path.join(self.full_results_dir, f"{config_id}_results.csv.gz")
        results_df["config_name"] = config_id
        results_df.to_csv(csv_file, index=False, compression='gzip')
        
        # Also save as HDF5 for faster loading if available
        # try:
        #     h5_file = os.path.join(self.full_results_dir, f"{config_id}_results.h5")
        #     results_df.to_hdf(h5_file, key='results', mode='w', complevel=5)
        # except:
        #     pass  # HDF5 optional
        # Also safe as parquet for faster loading, etc.
        try:
            pq_file = os.path.join(self.full_results_dir, f"{config_id}_results.parquet")
            results_df['report_id'] = results_df['report_id'].astype(str)
            results_df.to_parquet(pq_file)
        except:
            pass # parquet optional
    
    def load_full_config_results(self, config_id: str, experiment_name: str) -> Optional[pd.DataFrame]:
        """
        Load full report-level results for a specific configuration.
        
        CHANGE: New method to retrieve complete report-level data
        
        Args:
            config_id: Configuration ID
            experiment_name: Experiment name
            
        Returns:
            DataFrame with full report-level results or None if not found
        """
        full_results_dir = os.path.join(self.checkpoint_dir, f"{experiment_name}_full_results")
        
        # Try HDF5 first (faster)
        h5_file = os.path.join(full_results_dir, f"{config_id}_results.h5")
        if os.path.exists(h5_file):
            try:
                return pd.read_hdf(h5_file, key='results')
            except:
                pass
        
        # Try compressed CSV
        csv_file = os.path.join(full_results_dir, f"{config_id}_results.csv.gz")
        if os.path.exists(csv_file):
            return pd.read_csv(csv_file, compression='gzip')
        
        # Try regular CSV
        csv_file_plain = os.path.join(full_results_dir, f"{config_id}_results.csv")
        if os.path.exists(csv_file_plain):
            return pd.read_csv(csv_file_plain)
        
        return None
    
    def load_all_full_results(self, experiment_name: str) -> Dict[str, pd.DataFrame]:
        """
        Load all full report-level results for an experiment.
        
        CHANGE: New method to retrieve all report-level data
        
        Args:
            experiment_name: Experiment name
            
        Returns:
            Dict mapping config_id to DataFrame with full results
        """
        full_results_dir = os.path.join(self.checkpoint_dir, f"{experiment_name}_full_results")
        all_results = {}
        
        if not os.path.exists(full_results_dir):
            print(f"⚠️ Full results directory not found: {full_results_dir}")
            return all_results
        
        # Find all result files
        import glob
        h5_files = glob.glob(os.path.join(full_results_dir, "*_results.h5"))
        csv_files = glob.glob(os.path.join(full_results_dir, "*_results.csv*"))
        
        # Load from HDF5 files first
        for h5_file in h5_files:
            config_id = os.path.basename(h5_file).replace("_results.h5", "")
            try:
                all_results[config_id] = pd.read_hdf(h5_file, key='results')
            except:
                pass
        
        # Load from CSV files if not already loaded
        for csv_file in csv_files:
            config_id = os.path.basename(csv_file).replace("_results.csv.gz", "").replace("_results.csv", "")
            if config_id not in all_results:
                try:
                    if csv_file.endswith('.gz'):
                        all_results[config_id] = pd.read_csv(csv_file, compression='gzip')
                    else:
                        all_results[config_id] = pd.read_csv(csv_file)
                except:
                    pass
        
        print(f"📂 Loaded full results for {len(all_results)} configurations")
        return all_results
    
    def _save_checkpoint(self, checkpoint_file: str, completed_configs: set, results: list):
        """
        Save checkpoint data.
        
        CHANGE: Enhanced to include info about full results
        """
        checkpoint_data = {
            'completed_configs': completed_configs,
            'results': results,
            'timestamp': datetime.now().isoformat(),
            'n_completed': len(completed_configs),
            'has_full_results': hasattr(self, 'full_results_dir') and os.path.exists(self.full_results_dir)
        }
        
        try:
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            print(f"\n💾 Checkpoint saved: {len(completed_configs)} configurations")
            if checkpoint_data['has_full_results']:
                print(f"   Full results directory: {self.full_results_dir}")
        except Exception as e:
            print(f"\n⚠️ Failed to save checkpoint: {e}")
    
    def _load_checkpoint(self, checkpoint_file: str) -> Optional[Dict]:
        """
        Load checkpoint data.
        
        CHANGE: New method for checkpoint loading
        """
        try:
            with open(checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
            print(f"📂 Loaded checkpoint from {checkpoint_data['timestamp']}")
            return checkpoint_data
        except Exception as e:
            print(f"⚠️ Failed to load checkpoint: {e}")
            return None
    
    def _export_results(self, results_df: pd.DataFrame, experiment_name: str):
        """
        Export results to CSV and optionally HDF5.
        
        CHANGE: New method for result export
        """
        # Export to CSV
        csv_file = os.path.join(self.checkpoint_dir, f"{experiment_name}_results.csv")
        results_df.to_csv(csv_file, index=False)
        print(f"📄 Results exported to: {csv_file}")
        
        # Export to HDF5 if available
        try:
            import h5py
            h5_file = os.path.join(self.checkpoint_dir, f"{experiment_name}_results.h5")
            results_df.to_hdf(h5_file, key='results', mode='w')
            print(f"📄 Results exported to: {h5_file}")
        except ImportError:
            pass

    def _generate_config_id(
        self,
        embedding_model,
        seg_config,
        thresh_config,
        comp_config,
        sent_method,
        *,
        version: str = "v1",
        max_prefix_len: int = 120,   # safe length for directory names
        _to_dict=_to_dict,
        _method_block=_method_block,
        _sanitize=_sanitize
    ) -> str:
        seg_d   = _to_dict(seg_config)
        th_d    = _to_dict(thresh_config)
        comp_d  = _to_dict(comp_config)

        seg_id  = _method_block("seg",  seg_d,  ["method","min_length","window_size","overlap"])
        th_id   = _method_block("th",   th_d,   ["method","threshold","k","top_p","quantile"])
        comp_id = _method_block("comp", comp_d, ["method","similarity_weight","sentiment_weight","alpha","beta","gamma"])

        emb_id  = _sanitize(embedding_model.replace("-", "_"))
        sent_id = _sanitize(str(sent_method))

        payload = {
            "version": version,
            "embedding_model": embedding_model,
            "seg": seg_d,
            "thresh": th_d,
            "comp": comp_d,
            "sent_method": sent_method,
        }
        full_hash = hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()

        prefix = f"{version}__{emb_id}__{seg_id}__{th_id}__{comp_id}__sent_{sent_id}"
        if len(prefix) > max_prefix_len:
            prefix = prefix[:max_prefix_len].rstrip("_")

        return f"{prefix}__{full_hash[:8]}"
    
    def _compile_results(self, 
                        config_combo: Dict,
                        results_df: pd.DataFrame,
                        ground_truth: Dict[str, float],
                        experiment_name: str,
                        config_time: float) -> Dict:
        """Compile results for a single configuration."""
        
        # Basic statistics
        result = {
            'experiment_name': experiment_name,
            'config_id': config_combo['config_id'],
            'embedding_model': config_combo['embedding_model'],
            'sentiment_method': config_combo['sentiment_method'],
            'segmentation_method': config_combo['segmentation'].method,
            'threshold_method': config_combo['threshold'].method,
            'composite_method': config_combo['composite'].method,
            'n_documents': len(results_df),
            'n_successful': results_df['processing_success'].sum(),
            'success_rate': results_df['processing_success'].mean(),
            # 'mean_overall_composite': results_df['overall_composite_score'].mean(),
            # 'std_overall_composite': results_df['overall_composite_score'].std(),
            # 'mean_overall_strength': results_df['overall_strength_score'].mean(),
            # 'std_overall_strength': results_df['overall_strength_score'].std(),
            'mean_processing_time': results_df['processing_time'].mean(),
            'total_config_time': config_time,
        }
        
        # Add domain-level statistics
        for domain in self.domain_definitions.keys():
            domain_col = f'{domain}_domain_score'
            if domain_col in results_df.columns:
                result[f'mean_{domain}_score'] = results_df[domain_col].mean()
                result[f'std_{domain}_score'] = results_df[domain_col].std()
        
        # Add configuration details
        result.update({
            'seg_min_length': config_combo['segmentation'].min_length,
            'seg_window_size': getattr(config_combo['segmentation'], 'window_size', None),
            'seg_overlap': getattr(config_combo['segmentation'], 'overlap', None),
            'thresh_threshold': getattr(config_combo['threshold'], 'threshold', None),
            'thresh_k': getattr(config_combo['threshold'], 'k', None),
            'comp_similarity_weight': getattr(config_combo['composite'], 'similarity_weight', None),
            'comp_sentiment_weight': getattr(config_combo['composite'], 'sentiment_weight', None),
        })
        
        # Validation metrics if ground truth provided
        if ground_truth:
            valid_reports = results_df[results_df['report_id'].isin(ground_truth.keys())]
            
            if len(valid_reports) > 1:
                predicted_scores = []
                true_scores = []
                
                for _, row in valid_reports.iterrows():
                    report_id = row['report_id']
                    # predicted_scores.append(row['overall_composite_score'])
                    true_scores.append(ground_truth[report_id])
                
                if len(predicted_scores) > 1:
                    pearson_r, pearson_p = pearsonr(predicted_scores, true_scores)
                    spearman_r, spearman_p = spearmanr(predicted_scores, true_scores)
                    mse = mean_squared_error(true_scores, predicted_scores)
                    mae = mean_absolute_error(true_scores, predicted_scores)
                    
                    result.update({
                        'pearson_r': pearson_r,
                        'pearson_p': pearson_p,
                        'spearman_r': spearman_r,
                        'spearman_p': spearman_p,
                        'mse': mse,
                        'mae': mae,
                        'n_valid_samples': len(predicted_scores)
                    })
        
        return result
    
    def clear_cache(self, cache_type: Optional[str] = None):
        """Clear processor caches if using cached processor."""
        if self.use_caching and hasattr(self.processor, 'clear_cache'):
            self.processor.clear_cache(cache_type)
            print(f"Cache cleared: {cache_type or 'all'}")
        else:
            print("No cache to clear (not using cached processor)")
    
    def get_cache_stats(self) -> Optional[Dict]:
        """Get cache statistics if using cached processor."""
        if self.use_caching and hasattr(self.processor, 'get_cache_stats'):
            return self.processor.get_cache_stats()
        return None
    
    def visualize_results(self, results_df: pd.DataFrame, 
                         metric: str = 'mean_overall_composite',
                         save_path: Optional[str] = None):
        """
        Visualize experiment results.
        
        CHANGE: New method for result visualization
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Performance by segmentation method
        ax = axes[0, 0]
        seg_perf = results_df.groupby('segmentation_method')[metric].mean().sort_values()
        seg_perf.plot(kind='bar', ax=ax)
        ax.set_title(f'{metric} by Segmentation Method')
        ax.set_ylabel(metric)
        
        # Plot 2: Performance by threshold method
        ax = axes[0, 1]
        thresh_perf = results_df.groupby('threshold_method')[metric].mean().sort_values()
        thresh_perf.plot(kind='bar', ax=ax)
        ax.set_title(f'{metric} by Threshold Method')
        ax.set_ylabel(metric)
        
        # Plot 3: Performance by composite method
        ax = axes[1, 0]
        comp_perf = results_df.groupby('composite_method')[metric].mean().sort_values()
        comp_perf.plot(kind='bar', ax=ax)
        ax.set_title(f'{metric} by Composite Method')
        ax.set_ylabel(metric)
        
        # Plot 4: Processing time distribution
        ax = axes[1, 1]
        results_df['total_config_time'].hist(bins=30, ax=ax)
        ax.set_title('Configuration Processing Time Distribution')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Count')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"📊 Visualization saved to: {save_path}")
        else:
            plt.show()



# # Convenience functions
# def run_quick_experiment(domain_definitions: Dict[str, List[str]],
#                                reports_data: List[Tuple[str, str]],
#                                n_configs: int = 10,
#                                use_acceleration: bool = True) -> pd.DataFrame:
#     """
#     Run a quick experiment with a subset of configurations using caching.
    
#     CHANGE: Added acceleration parameter
    
#     Args:
#         domain_definitions: Domain definitions
#         reports_data: Report data
#         n_configs: Number of configurations to test
#         use_acceleration: Whether to use GPU/MPS acceleration
    
#     Returns:
#         Results DataFrame
#     """
#     runner = ExperimentRunner(
#         domain_definitions,
#         use_caching=True,
#         cache_embeddings=True,
#         cache_segments=True,
#         use_acceleration=use_acceleration
#     )
    
#     # Get subset of configurations
#     all_configs = runner.create_config_combinations()
#     test_configs = all_configs[:n_configs]
    
#     print(f"Running quick experiment with {n_configs} configurations...")
    
#     results = []
#     for config in tqdm(test_configs, desc="Configurations"):
#         result_df = runner.processor.process_reports(
#             reports_data,
#             config['segmentation'],
#             config['threshold'],
#             config['composite'],
#             use_parallel=False
#         )
#         results.append(result_df)
    
#     runner.processor.print_cache_report()
    
#     return pd.concat(results, ignore_index=True)


class LazyReportLoader:
    """
    Lazy loader for large report files.
    
    CHANGE: New class for memory-efficient data loading
    """
    def __init__(self, file_path: str, chunk_size: int = 10000):
        self.file_path = file_path
        self.chunk_size = chunk_size
    
    def __iter__(self):
        """Yield reports in chunks"""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            chunk = []
            for line in f:
                if '\t' in line:
                    parts = line.strip().split('\t', 1)
                    if len(parts) == 2:
                        report_id, text = parts
                        chunk.append((report_id, text))
                
                if len(chunk) >= self.chunk_size:
                    yield chunk
                    chunk = []
            
            if chunk:
                yield chunk
    
    def count_total(self) -> int:
        """Count total number of reports"""
        count = 0
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if '\t' in line:
                    count += 1
        return count


# Example usage
if __name__ == "__main__":
    # Define domains
    domain_definitions = {
        "LSC": [
            "Management acts quickly to correct safety problems",
            "Management provides adequate safety training",
            "Management listens to safety concerns"
        ],
        "COMM": [
            "Safety procedures are clearly communicated",
            "Employees are encouraged to report safety concerns",
            "Safety information is readily available"
        ]
    }
    
    # Sample data
    reports_data = [
        ("R001", "Management responds quickly to safety issues. Clear communication throughout."),
        ("R002", "Safety procedures need better communication. Training is adequate though."),
        ("R003", "Excellent safety leadership and open communication channels."),
        ("R004", "Management prioritizes production over safety concerns sometimes."),
        ("R005", "Regular safety meetings with good information sharing.")
    ]
    
    ground_truth = {
        "R001": 0.8,
        "R002": 0.4,
        "R003": 0.9,
        "R004": 0.2,
        "R005": 0.7
    }
    
    # Run experiment with checkpointing
    runner = ExperimentRunner(
        domain_definitions,
        use_acceleration=True,
        device='auto'
    )
    
    # Test with subset of configurations
    config_filter = {
        'segmentation_methods': ['sentence'],
        'threshold_methods': ['top_k', 'hard'],
        'composite_methods': ['attention_weight', 'weighted_average']
    }
    
    print("🔬 Running experiment with checkpointing and acceleration...")
    results = runner.run_experiment(
        reports_data, 
        ground_truth, 
        "test_experiment",
        config_filter=config_filter,
        resume_from_checkpoint=True,
        save_interval=2
    )
    
    print(f"\n✅ Experiment complete!")
    print(f"Results shape: {results.shape}")
    
    # Visualize results
    runner.visualize_results(
        results, 
        save_path=os.path.join(runner.checkpoint_dir, "experiment_results.png")
    )