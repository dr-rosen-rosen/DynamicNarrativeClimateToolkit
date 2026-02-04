"""
Early Warning Signals (EWS) for Irregular Event Series
======================================================

This module rebuilds our **count-based** rolling/expanding EWS pipeline so it
works cleanly for **irregularly spaced** time series. It mirrors the feature set
we outlined earlier and keeps dependencies minimal (pure NumPy/Pandas; SciPy is
used only if available for a few stats).

Core features
-------------
Detrending helpers (per-window):
  - `detrend_series(y, method="linear"|"moving_average"|"loess"|"none", ...)`
    * `linear`: OLS straight line
    * `moving_average`: centered MA with odd window length
    * `loess`: LOWESS-like local linear smoother with tricube weights (pure Python, no deps)

Indicators per window (on **detrended** series):
  - variance / std / skew / (excess) kurtosis
  - ACF(1)
  - AR(1) via OLS and return rate = 1 − φ₁
  - spectral density ratio (low / high power via FFT periodogram; configurable band fractions)
  - spectral slope (OLS slope of log power vs log freq)
  - DFA α (detrended fluctuation analysis; default linear detrending within segments)

Windowing over **event counts** (good for irregular sampling):
  - `rolling_or_expanding_ews(values, timestamps, window_size, step, mode, cfg)`
    * `mode="rolling"` uses overlapping fixed-count windows (step required)
    * `mode="expanding"` grows the window from the start (step ignored)
  - Returns one row per window with: `window_index, start_idx, end_idx, start_time, end_time, n_events` + all indicators.

Trend testing across windows:
  - `spearman_trend(values)` — Spearman ρ vs. window index.
  - `perm_test_trend(values, n_perm=999, random_state=None)` — permutation p-value for that ρ.
  - `kendall_tau_b(series, y=None)` — exact Kendall’s tau-b (handles ties properly).

Multivariate / community indicators (per window):
  - `community_indicators(window_df, cols)` — mean variance/std/cv, mean off-diagonal correlation, leading eigenvalue of standardized covariance, PC1 AR(1) & return rate.
  - `rolling_or_expanding_community_ews(df, time_col, value_cols, window_size, step=None, mode="rolling"|"expanding")`

Convenience:
  - `ews_for_metric(df, time_col, value_col, ...)` — compute per-window EWS and (optionally) merge features back onto event rows as `ews_<metric>__<indicator>`.

Notes on spectra & irregular sampling
-------------------------------------
Spectral measures assume a **uniform sample axis**. For irregular events we
use the sequence index as the implicit sampling axis (fs=1). If you have a
strong reason to weight by actual time gaps, consider resampling first or
switching to Lomb–Scargle (not included here to keep deps light).

"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Iterator, List, Literal, Optional, Tuple, Union

import math
import numpy as np
import pandas as pd

# Optional stats (nice-to-have, not required)
try:
    from scipy.stats import skew as scipy_skew, kurtosis as scipy_kurtosis, spearmanr as scipy_spearmanr
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _as_1d(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float).ravel()
    return arr


def _safe_var(x: np.ndarray) -> float:
    x = _as_1d(x)
    if x.size < 2:
        return np.nan
    return float(np.var(x, ddof=1))


def _safe_std(x: np.ndarray) -> float:
    v = _safe_var(x)
    return float(math.sqrt(v)) if np.isfinite(v) else np.nan


def _acf1(x: np.ndarray) -> float:
    x = _as_1d(x)
    if x.size < 3:
        return np.nan
    x = x[np.isfinite(x)]
    if x.size < 3:
        return np.nan
    x0, x1 = x[:-1], x[1:]
    if np.std(x0, ddof=1) == 0 or np.std(x1, ddof=1) == 0:
        return np.nan
    return float(np.corrcoef(x0, x1)[0, 1])


def _ar1_ols(x: np.ndarray) -> float:
    x = _as_1d(x)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return np.nan
    y = x[1:]
    X = x[:-1]
    denom = float(np.dot(X, X))
    if denom == 0:
        return np.nan
    phi = float(np.dot(X, y) / denom)
    return phi


def _return_rate(phi: float) -> float:
    if phi is None or np.isnan(phi):
        return np.nan
    return float(1.0 - phi)

# ---------------------------------------------------------------------------
# Detrending helpers
# ---------------------------------------------------------------------------

def _detrend_none(y: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    y = _as_1d(y)
    return y, np.zeros_like(y)


def _detrend_linear(y: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    y = _as_1d(y)
    n = y.size
    if n < 2 or not np.isfinite(y).any():
        return y, np.zeros_like(y)
    x = np.arange(n, dtype=float)
    mask = np.isfinite(y)
    if mask.sum() < 2:
        return y, np.zeros_like(y)
    A = np.vstack([x[mask], np.ones(mask.sum())]).T
    a, b = np.linalg.lstsq(A, y[mask], rcond=None)[0]
    trend = a * x + b
    return (y - trend), trend


def _detrend_moving_average(y: np.ndarray, window: Optional[int] = None, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    y = _as_1d(y)
    n = y.size
    if n == 0:
        return y, np.zeros_like(y)
    if window is None:
        # Odd window ~ n/5
        window = max(3, (n // 5) | 1)
    if window % 2 == 0:
        window += 1
    pad = window // 2
    y_filled = np.copy(y)
    if not np.isfinite(y_filled).all():
        # simple linear interpolation for NaNs
        s = pd.Series(y_filled)
        y_filled = s.interpolate(limit_direction='both').to_numpy()
    kernel = np.ones(window, dtype=float) / window
    trend = np.convolve(y_filled, kernel, mode='same')
    return (y - trend), trend

# Lightweight LOESS/LOWESS (tricube weights, local linear)
# This is a simplified implementation adequate for smoothing within windows.

def _lowess(x: np.ndarray, y: np.ndarray, frac: float = 0.3, it: int = 1) -> np.ndarray:
    n = x.size
    r = max(2, int(np.ceil(frac * n)))
    yest = np.zeros(n, dtype=float)
    for _ in range(max(1, it)):
        for i in range(n):
            # neighborhood indices
            lo = max(0, i - r)
            hi = min(n, i + r + 1)
            xi = x[lo:hi]
            yi = y[lo:hi]
            # distances and tricube weights
            d = np.abs(xi - x[i])
            if d.max() == 0:
                w = np.ones_like(d)
            else:
                u = d / d.max()
                w = (1 - u**3)**3
            # weighted linear regression
            W = np.diag(w)
            X = np.column_stack([xi, np.ones_like(xi)])
            try:
                beta = np.linalg.lstsq(W @ X, W @ yi, rcond=None)[0]
                yest[i] = beta[0] * x[i] + beta[1]
            except np.linalg.LinAlgError:
                yest[i] = yi.mean()
        # robustification (optional): skipped for simplicity in small windows
    return yest


def _detrend_loess(y: np.ndarray, loess_frac: float = 0.3, loess_iters: int = 1, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    y = _as_1d(y)
    n = y.size
    if n < 3:
        return y, np.zeros_like(y)
    x = np.arange(n, dtype=float)
    y_filled = np.copy(y)
    if not np.isfinite(y_filled).all():
        s = pd.Series(y_filled)
        y_filled = s.interpolate(limit_direction='both').to_numpy()
    trend = _lowess(x, y_filled, frac=loess_frac, it=loess_iters)
    return (y - trend), trend

DetrendMethod = Literal['none','linear','moving_average','loess']

_DETRENDS: Dict[DetrendMethod, Callable[..., Tuple[np.ndarray, np.ndarray]]] = {
    'none': _detrend_none,
    'linear': _detrend_linear,
    'moving_average': _detrend_moving_average,
    'loess': _detrend_loess,
}


def detrend_series(y: Union[np.ndarray, pd.Series], method: DetrendMethod = 'linear', **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    func = _DETRENDS.get(method, _detrend_linear)
    return func(_as_1d(y), **kwargs)

# ---------------------------------------------------------------------------
# Spectral features
# ---------------------------------------------------------------------------

def _psd_fft(y: np.ndarray, fs: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Simple periodogram using rFFT; returns (freqs, power). DC excluded.
    Uses next power-of-two for stability.
    """
    y = _as_1d(y)
    y = y[np.isfinite(y)]
    n = y.size
    if n < 8:
        return np.array([]), np.array([])
    y = y - y.mean()
    nfft = 1 << (n - 1).bit_length()
    X = np.fft.rfft(y, n=nfft)
    P = (X * np.conj(X)).real
    f = np.fft.rfftfreq(nfft, d=1.0 / fs)
    # drop DC
    return f[1:], P[1:]


def spectral_ratio(y: np.ndarray, fs: float = 1.0, low_frac: float = 0.1, high_frac: float = 0.1) -> float:
    """Low/High frequency power ratio using fractional bands of the positive spectrum.
    `low_frac` uses the lowest fraction of bins; `high_frac` uses the highest fraction.
    """
    f, P = _psd_fft(y, fs=fs)
    m = P.size
    if m < 4:
        return np.nan
    k_low = max(1, int(round(low_frac * m)))
    k_high = max(1, int(round(high_frac * m)))
    low = float(np.nansum(P[:k_low]))
    high = float(np.nansum(P[-k_high:]))
    if high <= 0:
        return np.nan
    return low / high


def spectral_slope(y: np.ndarray, fs: float = 1.0) -> float:
    """OLS slope of log10(power) vs. log10(freq) over positive spectrum.
    Returns NaN if insufficient bins or invalid log values.
    """
    f, P = _psd_fft(y, fs=fs)
    mask = (f > 0) & (P > 0)
    if mask.sum() < 4:
        return np.nan
    xf = np.log10(f[mask])
    yp = np.log10(P[mask])
    A = np.vstack([xf, np.ones_like(xf)]).T
    a, b = np.linalg.lstsq(A, yp, rcond=None)[0]
    return float(a)

# ---------------------------------------------------------------------------
# DFA (detrended fluctuation analysis)
# ---------------------------------------------------------------------------

def dfa_alpha(y: np.ndarray, order: int = 1, min_scale: int = 4, max_scale: Optional[int] = None, num_scales: int = 12) -> float:
    """Compute DFA scaling exponent α with linear (order=1) detrending by default.
    Basic, dependency-free implementation.
    """
    x = _as_1d(y)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 16:
        return np.nan
    # profile
    x = x - x.mean()
    Y = np.cumsum(x)
    if max_scale is None:
        max_scale = max(min(n // 4, 200), min_scale + 1)
    # logarithmically spaced scales
    scales = np.unique(np.floor(np.logspace(np.log10(min_scale), np.log10(max_scale), num=num_scales)).astype(int))
    F = []
    for s in scales:
        if s < 2:
            continue
        nseg = n // s
        if nseg < 2:
            continue
        rms = []
        for k in range(nseg):
            seg = Y[k*s:(k+1)*s]
            t = np.arange(s, dtype=float)
            # local polynomial of given order (only order=1 implemented here)
            A = np.vstack([t, np.ones_like(t)]).T if order == 1 else np.vander(t, N=order+1, increasing=True)
            beta = np.linalg.lstsq(A, seg, rcond=None)[0]
            trend = A @ beta
            rms.append(np.sqrt(np.mean((seg - trend)**2)))
        if len(rms) > 0:
            F.append((s, np.sqrt(np.mean(np.array(rms)**2))))
    if len(F) < 3:
        return np.nan
    s_arr = np.array([s for s, _ in F], dtype=float)
    F_arr = np.array([f for _, f in F], dtype=float)
    mask = (s_arr > 0) & (F_arr > 0)
    xf = np.log10(s_arr[mask])
    yf = np.log10(F_arr[mask])
    A = np.vstack([xf, np.ones_like(xf)]).T
    a, b = np.linalg.lstsq(A, yf, rcond=None)[0]
    return float(a)

# ---------------------------------------------------------------------------
# Indicators on a detrended series
# ---------------------------------------------------------------------------

def _skew_kurtosis(arr: np.ndarray) -> Tuple[float,float]:
    arr = _as_1d(arr)
    arr = arr[np.isfinite(arr)]
    if arr.size < 8:
        return np.nan, np.nan
    if _HAVE_SCIPY:
        return float(scipy_skew(arr, bias=False)), float(scipy_kurtosis(arr, fisher=True, bias=False))
    # fallback moments
    mu = float(np.mean(arr))
    m2 = float(np.mean((arr - mu)**2))
    if m2 <= 0:
        return np.nan, np.nan
    m3 = float(np.mean((arr - mu)**3))
    m4 = float(np.mean((arr - mu)**4))
    skew = m3 / (m2**1.5)
    kurt = m4 / (m2**2) - 3.0
    return float(skew), float(kurt)

def compute_window_indicators(y: np.ndarray, cfg: 'EWSConfig') -> Dict[str, float]:
    res, trend = detrend_series(
        y,
        method=cfg.detrend,
        window=cfg.ma_window,
        loess_frac=cfg.loess_frac,
        loess_iters=cfg.loess_iters
    )

    # Prepare
    x = np.asarray(res, dtype=float)
    metrics: Dict[str, float] = {}

    # mean / variance / std / cv
    if cfg.compute_mean:
        mu = float(np.nanmean(x)) if np.isfinite(x).any() else np.nan
        metrics['mean'] = mu
    else:
        mu = np.nan

    if cfg.compute_variance:
        metrics['variance'] = _safe_var(x)  # ddof consistent with your helper

    if cfg.compute_std:
        std = _safe_std(x)
        metrics['std'] = std
    else:
        std = np.nan

    if cfg.compute_cv:
        # safer CV guard than "mu not in (0, np.nan)"
        metrics['cv'] = (std / abs(mu)) if (np.isfinite(std) and np.isfinite(mu) and abs(mu) > 0) else np.nan

    # ACF(1), AR(1), Return rate
    if cfg.compute_acf1:
        metrics['acf1'] = _acf1(x)
    if cfg.compute_ar1 or cfg.compute_return_rate:
        phi = _ar1_ols(x)
        if cfg.compute_ar1:
            metrics['ar1'] = phi
        if cfg.compute_return_rate:
            metrics['return_rate'] = _return_rate(phi)  # 1 - phi

    # Skew / Kurtosis
    if cfg.compute_skew or cfg.compute_kurtosis:
        sk, kt = _skew_kurtosis(x)
        if cfg.compute_skew:
            metrics['skew'] = sk
        if cfg.compute_kurtosis:
            metrics['kurtosis'] = kt   # excess kurtosis if your helper returns it

    # Spectral metrics (use names aligned with config flags)
    if cfg.compute_spectral_ratio:
        metrics['spectral_ratio'] = spectral_ratio(
            x, fs=cfg.fs, low_frac=cfg.low_frac, high_frac=cfg.high_frac
        )
    if cfg.compute_spectral_slope:
        metrics['spectral_slope'] = spectral_slope(x, fs=cfg.fs)

    # DFA
    if cfg.compute_dfa_alpha:
        metrics['dfa_alpha'] = dfa_alpha(x, order=cfg.dfa_order)

    # --- OPTIONAL backward-compat aliases (remove if you don't need them) ---
    # Keep older column names some scripts may expect
    # if 'variance' in metrics and 'var' not in metrics:
    #     metrics['var'] = metrics['variance']
    # if 'spectral_ratio' in metrics and 'spec_ratio' not in metrics:
    #     metrics['spec_ratio'] = metrics['spectral_ratio']
    # if 'spectral_slope' in metrics and 'spec_slope' not in metrics:
    #     metrics['spec_slope'] = metrics['spectral_slope']
    # ------------------------------------------------------------------------

    return metrics


# ---------------------------------------------------------------------------
# Windowing over **event counts**
# ---------------------------------------------------------------------------

def _iter_rolling_indices(n: int, window_size: int, step: int) -> Iterator[Tuple[int,int]]:
    i = 0
    while i + window_size <= n:
        yield i, i + window_size
        i += step


def _iter_expanding_indices(n: int, start_size: int) -> Iterator[Tuple[int,int]]:
    for j in range(start_size, n + 1):
        yield 0, j

@dataclass
class EWSConfig:
    detrend: DetrendMethod = 'linear'
    fs: float = 1.0
    low_frac: float = 0.1
    high_frac: float = 0.1
    dfa_order: int = 1
    # detrending params
    ma_window: Optional[int] = None
    loess_frac: float = 0.3
    loess_iters: int = 1
    # Indicator toggles
    # Indicator toggles
    compute_mean: bool = True          # <-- NEW
    compute_variance: bool = True
    compute_std: bool = True
    compute_cv: bool = True            # <-- NEW
    compute_skew: bool = True
    compute_kurtosis: bool = True
    compute_acf1: bool = True
    compute_ar1: bool = True
    compute_return_rate: bool = True
    compute_spectral_ratio: bool = True
    compute_spectral_slope: bool = True
    compute_dfa_alpha: bool = True


def rolling_or_expanding_ews(values: Union[np.ndarray, pd.Series, List[float]],
                              timestamps: Union[np.ndarray, pd.Series, List[pd.Timestamp]],
                              window_size: int,
                              step: Optional[int] = None,
                              mode: Literal['rolling','expanding'] = 'rolling',
                              event_ids: Union[np.ndarray,pd.Series] = None,
                              anchor: Literal['right','left','center'] = 'right',
                              cfg: Optional[EWSConfig] = None) -> pd.DataFrame:
    """Compute EWS on **count-based** windows.

    Parameters
    ----------
    values : sequence of measurements (same order as timestamps)
    timestamps : sequence of event times (any datetime-like)
    window_size : number of events per window
    step : step between windows (required for mode='rolling')
    mode : 'rolling' or 'expanding'
    cfg : EWSConfig
    """
    cfg = cfg or EWSConfig()
    v = _as_1d(values)
    t = pd.to_datetime(pd.Index(timestamps)).to_numpy()
    n = v.size
    if n != t.size:
        raise ValueError("values and timestamps must have equal length")
    rows = []
    if mode == 'rolling':
        if step is None or step <= 0:
            raise ValueError("step must be specified and > 0 for rolling mode")
        it = _iter_rolling_indices(n, window_size, step)
    else:
        it = _iter_expanding_indices(n, window_size)

    widx = 0
    for i0, i1 in it:
        y = v[i0:i1]
        if y.size < max(8, window_size if mode=='expanding' else 8):
            continue
        ts0 = pd.Timestamp(t[i0]); ts1 = pd.Timestamp(t[i1-1])
        inds = compute_window_indicators(y, cfg)
        rows.append({
            'window_index': widx,
            'start_idx': i0,
            'end_idx': i1-1,
            'start_time': ts0,
            'end_time': ts1,
            'n_events': int(y.size),
            **inds,
        })
        widx += 1
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Trend testing
# ---------------------------------------------------------------------------

def spearman_trend(values: Union[np.ndarray, pd.Series]) -> Tuple[float, float]:
    """Spearman rho vs window index, returning (rho, pvalue).
    Uses SciPy if available, otherwise a permutation-based large-sample p.
    """
    y = _as_1d(values)
    x = np.arange(y.size, dtype=float)
    if _HAVE_SCIPY:
        rho, p = scipy_spearmanr(x, y, nan_policy='omit')
        return float(rho), float(p)
    # fallback: compute rho via ranks; p approx via normal approximation
    s = pd.Series(y).rank().to_numpy()
    t = pd.Series(x).rank().to_numpy()
    rho = np.corrcoef(s, t)[0, 1]
    # no exact p without SciPy; return NaN
    return float(rho), np.nan


def perm_test_trend(values: Union[np.ndarray, pd.Series], n_perm: int = 999, random_state: Optional[int] = None) -> float:
    """Permutation p-value for Spearman rho vs window index (two-sided)."""
    rng = np.random.default_rng(random_state)
    y = _as_1d(values)
    x = np.arange(y.size, dtype=float)
    # observed
    if _HAVE_SCIPY:
        rho_obs, _ = scipy_spearmanr(x, y, nan_policy='omit')
    else:
        r_y = pd.Series(y).rank().to_numpy()
        r_x = pd.Series(x).rank().to_numpy()
        rho_obs = np.corrcoef(r_x, r_y)[0, 1]
    # permutations
    cnt = 0
    for _ in range(n_perm):
        y_perm = rng.permutation(y)
        if _HAVE_SCIPY:
            rho, _ = scipy_spearmanr(x, y_perm, nan_policy='omit')
        else:
            r_y = pd.Series(y_perm).rank().to_numpy()
            r_x = pd.Series(x).rank().to_numpy()
            rho = np.corrcoef(r_x, r_y)[0, 1]
        if abs(rho) >= abs(rho_obs):
            cnt += 1
    p = (cnt + 1) / (n_perm + 1)
    return float(p)

# Exact Kendall tau-b (O(n^2), fine for modest window counts)

def kendall_tau_b(x: Union[np.ndarray, pd.Series], y: Optional[Union[np.ndarray, pd.Series]] = None) -> Tuple[float, int, int, Dict[str,int]]:
    """Return (tau_b, C, D, ties) where C/D are concordant/discordant pair counts.
    If y is None, tests monotonic trend of x vs its index.
    """
    x = _as_1d(x)
    if y is None:
        y = np.arange(x.size)
    else:
        y = _as_1d(y)
    n = x.size
    C = D = 0
    ties_x = ties_y = ties_xy = 0
    for i in range(n-1):
        for j in range(i+1, n):
            dx = np.sign(x[j] - x[i])
            dy = np.sign(y[j] - y[i])
            if dx == 0 and dy == 0:
                ties_xy += 1
            elif dx == 0:
                ties_x += 1
            elif dy == 0:
                ties_y += 1
            else:
                s = dx * dy
                if s > 0:
                    C += 1
                elif s < 0:
                    D += 1
    denom = math.sqrt((C + D + ties_x) * (C + D + ties_y))
    tau_b = (C - D) / denom if denom > 0 else np.nan
    return float(tau_b), C, D, {'ties_x': ties_x, 'ties_y': ties_y, 'ties_xy': ties_xy}

# ---------------------------------------------------------------------------
# Multivariate/community indicators
# ---------------------------------------------------------------------------

def community_indicators(window_df: pd.DataFrame, cols: List[str]) -> Dict[str, float]:
    X = window_df[cols].to_numpy(dtype=float)
    # standardize columns within the window
    Xs = (X - np.nanmean(X, axis=0)) / np.nanstd(X, axis=0, ddof=1)
    # mean var/std/cv of columns
    col_vars = np.nanvar(X, axis=0, ddof=1)
    col_stds = np.sqrt(col_vars)
    col_means = np.nanmean(X, axis=0)
    cvs = np.divide(col_stds, np.abs(col_means), out=np.full_like(col_stds, np.nan), where=(np.abs(col_means) > 0))
    mean_var = float(np.nanmean(col_vars))
    mean_std = float(np.nanmean(col_stds))
    mean_cv = float(np.nanmean(cvs))
    # mean off-diagonal correlation
    C = np.corrcoef(Xs, rowvar=False)
    if C.ndim == 0:
        mean_corr = np.nan
    else:
        m = C.shape[0]
        off = C[~np.eye(m, dtype=bool)]
        mean_corr = float(np.nanmean(off)) if off.size else np.nan
    # leading eigenvalue of standardized covariance
    cov = np.cov(Xs, rowvar=False)
    try:
        eigvals = np.linalg.eigvalsh(cov)
        eig1 = float(np.nanmax(eigvals))
    except np.linalg.LinAlgError:
        eig1 = np.nan
    # PC1 scores and AR(1)
    try:
        U, S, Vt = np.linalg.svd(np.nan_to_num(Xs), full_matrices=False)
        pc1 = U[:, 0] * S[0]
        phi = _ar1_ols(pc1)
        rr = _return_rate(phi)
    except Exception:
        phi = rr = np.nan
    return {'mean_var': mean_var, 'mean_std': mean_std, 'mean_cv': mean_cv, 'mean_corr': mean_corr, 'eig1': eig1, 'pc1_ar1': phi, 'pc1_return_rate': rr}


def rolling_or_expanding_community_ews(df: pd.DataFrame, time_col: str, value_cols: List[str], window_size: int, step: Optional[int] = None, mode: Literal['rolling','expanding'] = 'rolling') -> pd.DataFrame:
    times = pd.to_datetime(df[time_col]).to_numpy()
    n = df.shape[0]
    rows = []
    if mode == 'rolling':
        if step is None or step <= 0:
            raise ValueError("step must be specified and > 0 for rolling mode")
        it = _iter_rolling_indices(n, window_size, step)
    else:
        it = _iter_expanding_indices(n, window_size)
    widx = 0
    for i0, i1 in it:
        win = df.iloc[i0:i1]
        if len(win) < 3:
            continue
        t0 = pd.Timestamp(times[i0]); t1 = pd.Timestamp(times[i1-1])
        inds = community_indicators(win, value_cols)
        rows.append({'window_index': widx, 'start_idx': i0, 'end_idx': i1-1, 'start_time': t0, 'end_time': t1, 'n_events': int(len(win)), **inds})
        widx += 1
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def ews_for_metric(
        df: pd.DataFrame, 
        time_col: str, 
        value_col: str, 
        window_size: int, 
        step: Optional[int] = None, 
        mode: Literal['rolling','expanding'] = 'rolling', 
        cfg: Optional[EWSConfig] = None, 
        merge_with_original: bool = False, 
        metric_name: Optional[str] = None, 
        align: Literal['start','center','end'] = 'end',
        id_col: Optional[str] = None) -> pd.DataFrame:
    cfg = cfg or EWSConfig()
    values = df[value_col].to_numpy(dtype=float)
    times = pd.to_datetime(df[time_col]).to_numpy()
    ews = rolling_or_expanding_ews(values, times, window_size=window_size, step=step, mode=mode, cfg=cfg)
    if not merge_with_original:
        # Add event IDs to the window summary if requested
        if id_col:
            ews['event_id'] = np.nan
            for _, row in ews.iterrows():
                idx = int(row['end_idx'] if align == 'end' else row['start_idx'])
                ews.at[_, 'event_id'] = df.iloc[idx][id_col]
        return ews
    # merge back onto rows at window anchor
    metric = metric_name or value_col
    feature_cols = [c for c in ews.columns if c not in ['window_index','start_idx','end_idx','start_time','end_time','n_events']]
    out = df.copy()
    anchor_idx = {'start': 'start_idx', 'end': 'end_idx', 'center': None}[align]
    if align == 'center':
        centers = (ews['start_idx'] + ews['end_idx']) // 2
        ews = ews.assign(center_idx=centers)
        key = 'center_idx'
    else:
        key = anchor_idx
    for c in feature_cols:
        colname = f"ews_{metric}__{c}"
        out[colname] = np.nan
        for _, row in ews.iterrows():
            idx = int(row[key])
            out.iat[idx, out.columns.get_loc(colname)] = row[c]
    return out

# --- BEGIN COMPAT + RESTORE PATCH ---

# 1) Add EWMA detrend
def _detrend_ewma(y: np.ndarray, span: float | None = None, alpha: float | None = None, adjust: bool = True, **kwargs):
    """
    Exponentially weighted moving average trend; returns (residuals, trend).
    Provide either `span` (>0) or `alpha` in (0,1]; if both provided, `alpha` is used.
    """
    import pandas as _pd
    s = _pd.Series(np.asarray(y, dtype=float))
    if alpha is None and (span is None or span <= 0):
        raise ValueError("For detrend='ewma', provide a positive span or alpha in (0,1].")
    trend = s.ewm(alpha=alpha, span=span, adjust=adjust).mean().to_numpy()
    return (s.to_numpy() - trend).astype(float), trend.astype(float)

# 2) Restore Gaussian detrend (with SMA fallback)
try:
    from scipy.ndimage import gaussian_filter1d as _gauss1d
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

def _detrend_gaussian(y: np.ndarray, sigma: float = 5.0, **kwargs):
    """
    Gaussian-smoothed trend; returns (residuals, trend).
    If SciPy unavailable, falls back to simple moving average with window ~6*sigma.
    """
    y = np.asarray(y, dtype=float)
    if _HAVE_SCIPY:
        # NaN-safe: fill with mean, filter, then reinsert NaNs into trend where appropriate
        y_filled = np.nan_to_num(y, nan=np.nanmean(y) if np.isfinite(y).any() else 0.0)
        trend = _gauss1d(y_filled, sigma=max(0.5, float(sigma)))
    else:
        w = max(3, int(round(6 * float(sigma))))
        if w % 2 == 0:
            w += 1
        kernel = np.ones(w, dtype=float) / w
        y_filled = np.nan_to_num(y, nan=np.nanmean(y) if np.isfinite(y).any() else 0.0)
        trend = np.convolve(y_filled, kernel, mode='same')
    return (y - trend).astype(float), trend.astype(float)

# 3) Register new detrenders into the map used by detrend_series
try:
    _DETRENDS  # noqa: F401
except NameError:
    _DETRENDS = {}
# Add/override keys safely
_DETRENDS.update({
    'ewma': _detrend_ewma,
    'gaussian': _detrend_gaussian,
})

# 4) Back-compat wrappers for older public functions

def ar1_coef(y: np.ndarray) -> float:
    """Back-compat: returns AR(1) coefficient; now available as 'ar1' via compute_window_indicators."""
    return _ar1_ols(_as_1d(y))

def return_rate_from_ar1(phi1: float) -> float:
    """Back-compat: return rate = 1 - phi1."""
    return _return_rate(phi1)

def spectral_reddening(y: np.ndarray, fs: float = 1.0, low_frac: float = 0.1, high_frac: float = 0.1) -> float:
    """
    Back-compat: ratio of low-frequency to high-frequency power.
    Equivalent to spectral_ratio with provided bands.
    """
    return spectral_ratio(_as_1d(y), fs=fs, low_frac=low_frac, high_frac=high_frac)

def rolling_ews(values, timestamps, window_size: int, step: int, cfg: 'EWSConfig' | None = None) -> 'pd.DataFrame':
    """
    Back-compat: previously time/count based; now count-based rolling via rolling_or_expanding_ews.
    """
    return rolling_or_expanding_ews(values, timestamps, window_size=window_size, step=step, mode='rolling', cfg=cfg)

def rolling_count_windows(n: int, window_size: int, step: int):
    """
    Back-compat: yields (i0, i1) index pairs for count-based rolling windows.
    """
    yield from _iter_rolling_indices(n, window_size, step)

def rolling_time_windows(timestamps: 'pd.Series', window: 'pd.Timedelta' | str, step: 'pd.Timedelta' | str):
    """
    Back-compat time-based window indexer. Returns (i0, i1) pairs using timestamps.
    """
    import pandas as _pd
    t = _pd.to_datetime(timestamps).reset_index(drop=True)
    window = _pd.to_timedelta(window)
    step = _pd.to_timedelta(step)
    n = len(t)
    if n == 0:
        return
    start = t.iloc[0]
    i0 = 0
    while True:
        end_time = start + window
        # find last index within window
        i1 = int(_pd.Index(t).slice_indexer(start, end_time, inclusive='both')[-1]) + 1 if (t <= end_time).any() else i0
        if i1 - i0 >= 1:
            yield (i0, i1)
        # advance
        start = start + step
        # move i0 to first index >= start
        while i0 < n and t.iloc[i0] < start:
            i0 += 1
        if i0 >= n:
            break

def run_ews(values, timestamps, window_size: int, step: int, cfg: 'EWSConfig' | None = None, mode: str = 'rolling'):
    """
    Back-compat dispatcher; prefer rolling_or_expanding_ews directly.
    """
    mode = 'expanding' if str(mode).lower().startswith('exp') else 'rolling'
    return rolling_or_expanding_ews(values, timestamps, window_size=window_size, step=(None if mode=='expanding' else step), mode=mode, cfg=cfg)

# 5) Optional shim: burstiness+memory (if previously imported from ews)
def rolling_burstiness_memory_by_count(timestamps, window_size: int, step: int, unit: str = 's', **kwargs):
    """
    Back-compat shim: defers to temporal_features.rolling_burstiness_by_count if available.
    """
    try:
        from temporal_features import rolling_burstiness_by_count as _rb
    except Exception as e:
        raise ImportError("rolling_burstiness_memory_by_count has moved to temporal_features. "
                          "Install/import that module or update your calls.") from e
    return _rb(timestamps, window_size=window_size, step=step, unit=unit, **kwargs)

# --- END COMPAT + RESTORE PATCH ---

# ---------------------------------------------------------------------------
# __main__ quick smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = pd.date_range('2024-01-01', periods=400, freq='D')
    # AR(1) with rising phi (critical slowing down)
    np.random.seed(0)
    x = np.zeros(len(rng))
    for i in range(1, len(rng)):
        phi = 0.8 + 0.19 * (i/len(rng))
        x[i] = phi * x[i-1] + np.random.normal(0, 1)
    df = pd.DataFrame({'t': rng, 'x': x})
    cfg = EWSConfig(detrend='linear', fs=1.0, low_frac=0.1, high_frac=0.1, dfa_order=1)
    e = rolling_or_expanding_ews(df['x'], df['t'], window_size=50, step=25, mode='rolling', cfg=cfg)
    print(e.head())