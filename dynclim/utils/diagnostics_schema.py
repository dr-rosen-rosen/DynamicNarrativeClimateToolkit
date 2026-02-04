"""
Standardized Diagnostics Schema for CPU/GPU Parity
===================================================

This module defines the standard diagnostics structure that both CPU and GPU
scoring paths must emit. It provides validation and helper functions.

Usage:
    from diagnostics_schema import standardize_diagnostics, validate_diagnostics
    
    # After computing diagnostics in your scorer:
    raw_diag = {...}  # Your diagnostics dict
    clean_diag = standardize_diagnostics(raw_diag)
    
    # Optionally validate:
    if not validate_diagnostics(clean_diag):
        print("Warning: diagnostics incomplete")
"""

from typing import Dict, Any, Optional, List
import numpy as np


# =============================================================================
# Standard Field Definitions
# =============================================================================

STANDARD_FIELDS = {
    # -------------------------------------------------------------------------
    # Selection Metrics (counts and rates)
    # -------------------------------------------------------------------------
    "n_total": {
        "type": int,
        "description": "Total number of segments before selection",
        "required": True,
        "default": 0
    },
    "n_selected": {
        "type": int,
        "description": "Number of segments after selection",
        "required": True,
        "default": 0
    },
    "selection_rate": {
        "type": float,
        "description": "Fraction of segments selected (n_selected / n_total)",
        "required": False,
        "default": 0.0
    },
    
    # -------------------------------------------------------------------------
    # Selection Method Identification
    # -------------------------------------------------------------------------
    "selection_mode": {
        "type": str,
        "description": "How selection was applied: 'post_threshold' | 'all' | 'loose_quantile'",
        "required": True,
        "default": "post_threshold",
        "allowed_values": ["post_threshold", "all", "loose_quantile"]
    },
    "selector_type": {
        "type": str,
        "description": "Type of selector used: 'absolute' | 'topk' | 'quantile' | 'none' | etc.",
        "required": True,
        "default": "none",
        "allowed_values": ["absolute", "topk", "quantile", "topp", "percent_of_max", 
                          "zscore", "none", "soft_weight", "adaptive", "diversity_top_k"]
    },
    
    # -------------------------------------------------------------------------
    # Selector Parameters (only relevant one should be set)
    # -------------------------------------------------------------------------
    "threshold_value": {
        "type": float,
        "description": "Threshold value for 'absolute' selector",
        "required": False,
        "default": None
    },
    "k_used": {
        "type": int,
        "description": "K value for 'topk' selector",
        "required": False,
        "default": None
    },
    "q_used": {
        "type": float,
        "description": "Quantile value for 'quantile' selector (0-1)",
        "required": False,
        "default": None
    },
    "p_used": {
        "type": float,
        "description": "Cumulative probability for 'topp' selector (0-1)",
        "required": False,
        "default": None
    },
    "frac_used": {
        "type": float,
        "description": "Fraction of max for 'percent_of_max' selector (0-1)",
        "required": False,
        "default": None
    },
    "zmin_used": {
        "type": float,
        "description": "Minimum z-score for 'zscore' selector",
        "required": False,
        "default": None
    },
    "lambda_used": {
        "type": float,
        "description": "Lambda parameter for 'adaptive' or 'diversity' selectors",
        "required": False,
        "default": None
    },
    
    # -------------------------------------------------------------------------
    # Similarity Statistics - All Segments
    # -------------------------------------------------------------------------
    "sims_all_max": {
        "type": float,
        "description": "Maximum similarity across all segments",
        "required": False,
        "default": None
    },
    "sims_all_mean": {
        "type": float,
        "description": "Mean similarity across all segments",
        "required": False,
        "default": None
    },
    "sims_all_std": {
        "type": float,
        "description": "Standard deviation of similarities across all segments",
        "required": False,
        "default": None
    },
    
    # -------------------------------------------------------------------------
    # Similarity Statistics - Selected Segments
    # -------------------------------------------------------------------------
    "sims_sel_max": {
        "type": float,
        "description": "Maximum similarity across selected segments",
        "required": False,
        "default": None
    },
    "sims_sel_mean": {
        "type": float,
        "description": "Mean similarity across selected segments",
        "required": False,
        "default": None
    },
    "sims_sel_std": {
        "type": float,
        "description": "Standard deviation of similarities across selected segments",
        "required": False,
        "default": None
    },
    
    # -------------------------------------------------------------------------
    # Scorer Method Identification
    # -------------------------------------------------------------------------
    "scorer_method": {
        "type": str,
        "description": "Scorer type: 'weighted_average' | 'sim_weighted' | 'softmax' | etc.",
        "required": True,
        "default": "weighted_average",
        "allowed_values": ["weighted_average", "sim_weighted", "softmax", "uniform",
                          "sim_power", "attention_weight", "separate_features",
                          "topk_attention", "power_attention", "trimmed_weighted"]
    },
    
    # -------------------------------------------------------------------------
    # Scorer Parameters (prefixed with sc_)
    # -------------------------------------------------------------------------
    "sc_tau": {
        "type": float,
        "description": "Temperature parameter for softmax scorer",
        "required": False,
        "default": None
    },
    "sc_power": {
        "type": float,
        "description": "Power parameter for power attention scorer",
        "required": False,
        "default": None
    },
    "sc_sim_weight": {
        "type": float,
        "description": "Similarity weight for weighted_average scorer (0-1)",
        "required": False,
        "default": None
    },
    "sc_sent_weight": {
        "type": float,
        "description": "Sentiment weight for weighted_average scorer (0-1)",
        "required": False,
        "default": None
    },
    "sc_trim": {
        "type": float,
        "description": "Trim parameter for trimmed_weighted scorer",
        "required": False,
        "default": None
    },
    "sc_trim_on": {
        "type": str,
        "description": "What to trim on: 'similarity' or 'sentiment'",
        "required": False,
        "default": None
    },
    
    # -------------------------------------------------------------------------
    # Strength Metrics
    # -------------------------------------------------------------------------
    "strength_sim_mean": {
        "type": float,
        "description": "Mean similarity strength (always computed)",
        "required": False,
        "default": None
    },
    "strength_entropy": {
        "type": float,
        "description": "Negative entropy strength (only for attention scorers)",
        "required": False,
        "default": None
    },
    "strength_type": {
        "type": str,
        "description": "Which strength metric is primary: 'mean_sim' | 'neg_entropy'",
        "required": False,
        "default": "mean_sim",
        "allowed_values": ["mean_sim", "neg_entropy"]
    },
}

# Fields that are required in every diagnostics dict
REQUIRED_FIELDS = [k for k, v in STANDARD_FIELDS.items() if v.get("required", False)]


# =============================================================================
# Helper Functions
# =============================================================================

def standardize_diagnostics(raw_diag: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert raw diagnostics dict to standardized format.
    
    This function:
    1. Renames old field names to new standard names
    2. Fills in missing optional fields with defaults
    3. Computes derived fields (like selection_rate)
    4. Ensures correct types
    
    Args:
        raw_diag: Raw diagnostics dict from scorer/selector
        
    Returns:
        Standardized diagnostics dict with all standard fields
        
    Example:
        >>> raw = {
        ...     "threshold_method": "topk",  # Old name
        ...     "n_total": 10,
        ...     "n_selected": 5,
        ...     "k_used": 5
        ... }
        >>> std = standardize_diagnostics(raw)
        >>> std["selector_type"]  # New name
        'topk'
    """
    std_diag = {}
    
    # Field name mappings (old_name -> new_name)
    field_mappings = {
        "threshold_method": "selector_type",  # Old CPU field name
        "sc_apply_over": "selection_mode",     # Old GPU field name
    }
    
    # Apply field name mappings
    for old_name, new_name in field_mappings.items():
        if old_name in raw_diag and new_name not in raw_diag:
            raw_diag[new_name] = raw_diag[old_name]
    
    # Copy all standard fields from raw_diag
    for field_name, field_spec in STANDARD_FIELDS.items():
        if field_name in raw_diag:
            value = raw_diag[field_name]
            
            # Handle numpy types
            if isinstance(value, (np.integer, np.floating)):
                value = value.item()
            elif isinstance(value, np.ndarray):
                if value.size == 1:
                    value = value.item()
                else:
                    value = value.tolist()
            
            # Convert to expected type if possible
            expected_type = field_spec["type"]
            if value is not None:
                try:
                    if expected_type == int and not isinstance(value, bool):
                        value = int(value)
                    elif expected_type == float:
                        value = float(value)
                    elif expected_type == str:
                        value = str(value)
                except (ValueError, TypeError):
                    pass  # Keep original value if conversion fails
            
            std_diag[field_name] = value
        else:
            # Use default for missing fields
            std_diag[field_name] = field_spec.get("default")
    
    # Compute derived fields
    if "selection_rate" not in std_diag or std_diag["selection_rate"] is None:
        n_total = std_diag.get("n_total", 0)
        n_selected = std_diag.get("n_selected", 0)
        if n_total > 0:
            std_diag["selection_rate"] = float(n_selected) / float(n_total)
        else:
            std_diag["selection_rate"] = 0.0
    
    # Preserve any additional fields not in standard schema (with prefix)
    for key, value in raw_diag.items():
        if key not in std_diag and not key.startswith("_"):
            # Add non-standard fields with "extra_" prefix
            std_diag[f"extra_{key}"] = value
    
    return std_diag


def validate_diagnostics(diag: Dict[str, Any], strict: bool = False) -> bool:
    """
    Validate that diagnostics dict conforms to standard schema.
    
    Args:
        diag: Diagnostics dict to validate
        strict: If True, enforce allowed_values and types strictly
        
    Returns:
        True if valid, False otherwise
        
    Side effects:
        Prints warnings for validation failures
    """
    is_valid = True
    
    # Check required fields are present
    for field in REQUIRED_FIELDS:
        if field not in diag or diag[field] is None:
            print(f"⚠️  Missing required field: {field}")
            is_valid = False
    
    # Check field types and values
    for field_name, value in diag.items():
        if field_name.startswith("extra_"):
            continue  # Skip non-standard fields
        
        if field_name not in STANDARD_FIELDS:
            if not field_name.startswith("th_") and not field_name.startswith("sc_"):
                print(f"⚠️  Unknown field: {field_name}")
            continue
        
        field_spec = STANDARD_FIELDS[field_name]
        
        # Check type if value is not None
        if value is not None:
            expected_type = field_spec["type"]
            if not isinstance(value, expected_type):
                if strict:
                    print(f"⚠️  Field '{field_name}' has wrong type: "
                          f"expected {expected_type.__name__}, got {type(value).__name__}")
                    is_valid = False
        
        # Check allowed values if specified
        if strict and "allowed_values" in field_spec:
            if value is not None and value not in field_spec["allowed_values"]:
                print(f"⚠️  Field '{field_name}' has invalid value: {value} "
                      f"(allowed: {field_spec['allowed_values']})")
                is_valid = False
    
    return is_valid


def print_diagnostics(diag: Dict[str, Any], title: str = "Diagnostics"):
    """
    Pretty-print diagnostics in organized sections.
    
    Args:
        diag: Diagnostics dict
        title: Title to display
    """
    print("\n" + "=" * 70)
    print(f"{title}")
    print("=" * 70)
    
    # Section 1: Selection Info
    print("\n📊 SELECTION:")
    print(f"   selector_type:   {diag.get('selector_type', 'N/A')}")
    print(f"   selection_mode:  {diag.get('selection_mode', 'N/A')}")
    print(f"   n_total:         {diag.get('n_total', 0)}")
    print(f"   n_selected:      {diag.get('n_selected', 0)}")
    print(f"   selection_rate:  {diag.get('selection_rate', 0.0):.3f}")
    
    # Section 2: Selector Parameters
    params = []
    if diag.get('threshold_value') is not None:
        params.append(f"threshold={diag['threshold_value']:.3f}")
    if diag.get('k_used') is not None:
        params.append(f"k={diag['k_used']}")
    if diag.get('q_used') is not None:
        params.append(f"q={diag['q_used']:.3f}")
    if diag.get('p_used') is not None:
        params.append(f"p={diag['p_used']:.3f}")
    if diag.get('frac_used') is not None:
        params.append(f"frac={diag['frac_used']:.3f}")
    if diag.get('zmin_used') is not None:
        params.append(f"zmin={diag['zmin_used']:.3f}")
    if diag.get('lambda_used') is not None:
        params.append(f"lambda={diag['lambda_used']:.3f}")
    
    if params:
        print(f"   parameters:      {', '.join(params)}")
    
    # Section 3: Similarity Statistics
    print("\n📈 SIMILARITIES:")
    print(f"   All segments:")
    print(f"      max:  {diag.get('sims_all_max', 0.0):.4f}")
    print(f"      mean: {diag.get('sims_all_mean', 0.0):.4f}")
    if diag.get('sims_all_std') is not None:
        print(f"      std:  {diag.get('sims_all_std', 0.0):.4f}")
    
    print(f"   Selected segments:")
    print(f"      max:  {diag.get('sims_sel_max', 0.0):.4f}")
    print(f"      mean: {diag.get('sims_sel_mean', 0.0):.4f}")
    if diag.get('sims_sel_std') is not None:
        print(f"      std:  {diag.get('sims_sel_std', 0.0):.4f}")
    
    # Section 4: Scorer Info
    print("\n🎯 SCORING:")
    print(f"   scorer_method:   {diag.get('scorer_method', 'N/A')}")
    print(f"   strength_type:   {diag.get('strength_type', 'N/A')}")
    
    scorer_params = []
    if diag.get('sc_tau') is not None:
        scorer_params.append(f"tau={diag['sc_tau']:.3f}")
    if diag.get('sc_power') is not None:
        scorer_params.append(f"power={diag['sc_power']:.3f}")
    if diag.get('sc_sim_weight') is not None:
        scorer_params.append(f"sim_w={diag['sc_sim_weight']:.3f}")
    if diag.get('sc_sent_weight') is not None:
        scorer_params.append(f"sent_w={diag['sc_sent_weight']:.3f}")
    if diag.get('sc_trim') is not None:
        scorer_params.append(f"trim={diag['sc_trim']:.3f}")
    
    if scorer_params:
        print(f"   parameters:      {', '.join(scorer_params)}")
    
    # Section 5: Extra fields
    extra_fields = {k: v for k, v in diag.items() 
                   if k not in STANDARD_FIELDS and not k.startswith("_")}
    if extra_fields:
        print("\n📝 ADDITIONAL:")
        for key, value in sorted(extra_fields.items()):
            if isinstance(value, float):
                print(f"   {key}: {value:.4f}")
            else:
                print(f"   {key}: {value}")


def compare_diagnostics(diag1: Dict[str, Any], diag2: Dict[str, Any], 
                       label1: str = "CPU", label2: str = "GPU") -> bool:
    """
    Compare two diagnostics dicts and report differences.
    
    Args:
        diag1: First diagnostics dict
        diag2: Second diagnostics dict
        label1: Label for first dict (e.g., "CPU")
        label2: Label for second dict (e.g., "GPU")
        
    Returns:
        True if diagnostics match (within tolerance), False otherwise
    """
    print("\n" + "=" * 70)
    print(f"Comparing {label1} vs {label2} Diagnostics")
    print("=" * 70)
    
    all_fields = set(diag1.keys()) | set(diag2.keys())
    all_fields = sorted(f for f in all_fields if not f.startswith("_"))
    
    matches = True
    tolerance = 1e-5
    
    for field in all_fields:
        val1 = diag1.get(field)
        val2 = diag2.get(field)
        
        if val1 is None and val2 is None:
            continue
        
        if val1 is None or val2 is None:
            print(f"⚠️  {field}: {label1}={val1}, {label2}={val2}")
            matches = False
            continue
        
        # Compare based on type
        if isinstance(val1, (int, np.integer)) and isinstance(val2, (int, np.integer)):
            if int(val1) != int(val2):
                print(f"❌ {field}: {label1}={val1}, {label2}={val2}")
                matches = False
            else:
                print(f"✅ {field}: {val1}")
        
        elif isinstance(val1, (float, np.floating)) and isinstance(val2, (float, np.floating)):
            diff = abs(float(val1) - float(val2))
            if diff > tolerance:
                print(f"❌ {field}: {label1}={val1:.6f}, {label2}={val2:.6f} (diff={diff:.9f})")
                matches = False
            else:
                print(f"✅ {field}: {val1:.6f}")
        
        elif isinstance(val1, str) and isinstance(val2, str):
            if val1 != val2:
                print(f"❌ {field}: {label1}='{val1}', {label2}='{val2}'")
                matches = False
            else:
                print(f"✅ {field}: '{val1}'")
        
        else:
            if val1 != val2:
                print(f"⚠️  {field}: {label1}={val1}, {label2}={val2}")
                matches = False
    
    print("\n" + "=" * 70)
    if matches:
        print("✅ DIAGNOSTICS MATCH")
    else:
        print("❌ DIAGNOSTICS DIFFER")
    print("=" * 70)
    
    return matches


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Example 1: Standardize old-format diagnostics
    print("Example 1: Standardizing old-format diagnostics")
    print("-" * 70)
    
    old_cpu_diag = {
        "threshold_method": "topk",  # Old field name
        "n_total": 10,
        "n_selected": 5,
        "k_used": 5,
        "sims_all_mean": 0.65,
        "sims_sel_mean": 0.82,
        "scorer_method": "weighted_average"
    }
    
    std_diag = standardize_diagnostics(old_cpu_diag)
    print_diagnostics(std_diag, "Standardized CPU Diagnostics")
    
    # Example 2: Validate diagnostics
    print("\n\nExample 2: Validating diagnostics")
    print("-" * 70)
    
    is_valid = validate_diagnostics(std_diag)
    print(f"\nValidation result: {'✅ VALID' if is_valid else '❌ INVALID'}")
    
    # Example 3: Compare CPU and GPU diagnostics
    print("\n\nExample 3: Comparing CPU and GPU diagnostics")
    print("-" * 70)
    
    gpu_diag = {
        "selector_type": "topk",  # New field name
        "n_total": 10,
        "n_selected": 5,
        "k_used": 5,
        "sims_all_mean": 0.65,
        "sims_sel_mean": 0.82001,  # Slight difference
        "scorer_method": "weighted_average",
        "selection_mode": "post_threshold"
    }
    
    compare_diagnostics(std_diag, gpu_diag, "CPU", "GPU")
