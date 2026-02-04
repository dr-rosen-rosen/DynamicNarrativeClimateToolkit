# ============================================================================
# Functions for diagnostics
# ============================================================================
import numpy as np

def prep_arrays(similarities, sentiments):
    # accept torch tensors or nested lists
    try:
        s = np.asarray(similarities, dtype=float).reshape(-1)
        m = np.asarray(sentiments,   dtype=float).reshape(-1)
    except Exception:
        s = np.array(similarities, dtype=float).reshape(-1)
        m = np.array(sentiments,   dtype=float).reshape(-1)

    mask = np.isfinite(s) & np.isfinite(m)
    s, m = s[mask], m[mask]
    # sims should already be in [0,1]; keep a safe clip
    s = np.clip(s, 0.0, 1.0)
    return s, m

def softmax_weights(s, tau=0.25, eps=1e-8):
    x = s / max(tau, eps)
    x -= x.max()  # numeric stability
    w = np.exp(x)
    z = w.sum() + eps
    return w / z

def weight_entropy(w, eps=1e-12):
    w = np.asarray(w, float)
    w = w / (w.sum() + eps)
    return float(-(w * np.log(w + eps)).sum())

def effective_n(w, eps=1e-12):
    w = np.asarray(w, float)
    s1 = w.sum()
    s2 = (w**2).sum()
    return float((s1*s1) / (s2 + eps))

def diagnostics(weights, sentiments):
    w = np.asarray(weights, float)
    m = np.asarray(sentiments, float)
    wsum = w.sum()
    mu = float((w*m).sum() / max(wsum, 1e-8)) if w.size else 0.0
    var = float((w*((m-mu)**2)).sum() / max(wsum, 1e-8)) if w.size else 0.0
    return {
        "ess": effective_n(w),
        "weight_entropy": weight_entropy(w),
        "mu_w": mu,
        "sd_w": np.sqrt(var),
        "n": int(w.size),
        "max_weight": float(w.max()) if w.size else 0.0,
    }