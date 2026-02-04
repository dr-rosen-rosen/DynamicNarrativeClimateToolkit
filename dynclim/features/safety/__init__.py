# dynclim/features/safety/__init__.py
"""Safety climate feature extraction

This module provides tools for extracting safety-related features from text data
using validated safety climate scales.
"""

# Import the main processor (using the best version)
# from dynclim.features.safety.optimized_processor import OptimizedProcessor as SafetyProcessor
from dynclim.features.safety.unified_processor import Processor

# Import configuration classes
from dynclim.features.safety.toolkit import (
    SegmentationConfig,
    ThresholdConfig,
    CompositeConfig,
    # FeatureExtractor,
    # Individual segmenters if needed
    SentenceSegmenter,
    ClauseSegmenter,
    # Individual thresholders if needed  
    TopKThresholder,
    AdaptiveThresholder,
    SoftWeightThresholder,
    # Individual scorers if needed
    AttentionWeightScorer,
    WeightedAverageScorer
)

from dynclim.features.safety.seg_core import (
    SegmentationStore, 
    sha1, 
    char_offsets_from_sentences, 
    slice_by_offsets, 
    filter_min_length, 
    HAVE_BLING
)
__all__ = [
    # Main processor
    'Processor',
    
    # Configuration classes
    'SegmentationConfig',
    'ThresholdConfig', 
    'CompositeConfig',
    
    # Core extractor
    # 'FeatureExtractor',
    
    # Individual components (optional - only if users need direct access)
    'SentenceSegmenter',
    'ClauseSegmenter',
    'TopKThresholder',
    'SoftWeightThresholder',
    'AdaptiveThresholder',
    'AttentionWeightScorer',
    'WeightedAverageScorer',
    
    'SegmentationStore', 
    'sha1', 
    'char_offsets_from_sentences', 
    'slice_by_offsets', 
    'filter_min_length', 
    'HAVE_BLING'
]