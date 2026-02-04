
"""
temporal_features.py
---------------------
Tools for computing inter-event durations, burstiness metrics (original and finite-size corrected),
memory coefficients, and rolling-by-count window summaries for timestamped event sequences.

References
----------
- Goh, K.-I., & Barabási, A.-L. (2008). Burstiness and memory in complex systems. *EPL*, 81(48002).
  Original burstiness B = (CV - 1) / (CV + 1) and memory coefficient M (lag-1 correlation).
- Kim, E.-K., & Jo, H.-H. (2016). Measuring burstiness for finite event sequences. *Phys. Rev. E*, 94, 032311.
  Finite-size corrected burstiness A_n(r) for periodic (Eq. 22) and open boundaries (App. A, Eq. A17);
  and extensions controlling for minimum inter-event time ỹ (Eq. 28 for periodic; open-boundary analog in App. A).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

TimeUnit = Literal["s", "ms", "us", "ns", "m", "h", "d"]
Method = Literal["goh", "kimjo2016_periodic", "kimjo2016_open"]
# For minimum-interevent variants we expose helpers below.

_UNIT_DIVISORS = {
    "ns": 1e-9,
    "us": 1e-6,
    "ms": 1e-3,
    "s": 1.0,
    "m": 60.0,
    "h": 3600.0,
    "d": 86400.0,
}

def _as_unit(seconds: np.ndarray, unit: TimeUnit) -> np.ndarray:
    if unit not in _UNIT_DIVISORS:
        raise ValueError(f"unit must be one of {list(_UNIT_DIVISORS)}, got {unit!r}")
    return seconds / _UNIT_DIVISORS[unit]

# ---------- Core: inter-event durations ----------

def inter_event_durations(
    timestamps: Union[Sequence[pd.Timestamp], pd.Series, Sequence[str]],
    sort: bool = False, # this should be sorted before calling in most cases, otherwise may be difficult to link back to other data
    unit: TimeUnit = "s",
) -> np.ndarray:
    ts = pd.to_datetime(pd.Series(timestamps).dropna())
    if sort:
        ts = ts.sort_values(kind="mergesort").reset_index(drop=True)
    diffs = ts.diff().iloc[1:]  # first is NaT
    seconds = diffs.dt.total_seconds().to_numpy(dtype=float)
    return _as_unit(seconds, unit)

# ---------- Burstiness ----------

@dataclass
class BurstinessResult:
    method: str
    n: int        # number of intervals used
    mean_dt: float
    sd_dt: float
    cv: float
    b: float      # burstiness value (B or A_n depending on method)

def _cv(x: np.ndarray, ddof: int = 0) -> Tuple[int, float, float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2:
        return n, np.nan, np.nan, np.nan
    mean_dt = float(np.mean(x))
    sd_dt = float(np.std(x, ddof=ddof))
    if mean_dt <= 0:
        return n, mean_dt, sd_dt, np.nan
    return n, mean_dt, sd_dt, sd_dt / mean_dt

def burstiness(
    inter_event_durs: Union[Sequence[float], np.ndarray],
    method: Method = "goh",
    ddof: int = 0,
) -> BurstinessResult:
    """
    method:
      - 'goh': B = (CV - 1) / (CV + 1)
      - 'kimjo2016_periodic': A_n(r) = (sqrt(n+1) r - sqrt(n-1)) / ((sqrt(n+1)-2) r + sqrt(n-1))
      - 'kimjo2016_open':     A_n(r) = (sqrt(n+2) r - sqrt(n))   / ((sqrt(n+2)-2) r + sqrt(n))
    """
    n, mean_dt, sd_dt, cv = _cv(np.asarray(inter_event_durs, dtype=float), ddof=ddof)
    if not np.isfinite(cv):
        return BurstinessResult(method=method, n=n, mean_dt=mean_dt, sd_dt=sd_dt, cv=cv, b=float("nan"))
    if method == "goh":
        b = (cv - 1.0) / (cv + 1.0)
    elif method == "kimjo2016_periodic":
        # n is number of intervals
        if n < 2:
            b = float("nan")
        else:
            num = np.sqrt(n + 1.0) * cv - np.sqrt(n - 1.0)
            den = (np.sqrt(n + 1.0) - 2.0) * cv + np.sqrt(n - 1.0)
            b = num / den if den != 0 else float("nan")
    elif method == "kimjo2016_open":
        if n < 1:
            b = float("nan")
        else:
            num = np.sqrt(n + 2.0) * cv - np.sqrt(n * 1.0)
            den = (np.sqrt(n + 2.0) - 2.0) * cv + np.sqrt(n * 1.0)
            b = num / den if den != 0 else float("nan")
    else:
        raise ValueError("method must be 'goh', 'kimjo2016_periodic', or 'kimjo2016_open'")
    return BurstinessResult(method=method, n=n, mean_dt=mean_dt, sd_dt=sd_dt, cv=cv, b=float(b))

# ---------- Minimum inter-event time variants (Kim & Jo 2016, Eq. (28) and open-boundary analog) ----------

def finite_size_burstiness_with_min_iet(
    inter_event_durs: Union[Sequence[float], np.ndarray],
    method: Literal["periodic", "open"] = "periodic",
    y_tilde: float = 0.0,
    ddof: int = 0,
) -> BurstinessResult:
    """
    Finite-size corrected burstiness controlling for minimum inter-event time fraction y~ = tau_min / T.

    Parameters
    ----------
    inter_event_durs : array-like of float
        Inter-event durations (any consistent unit).
    method : {'periodic','open'}
        Use Kim & Jo periodic Eq. (28) or open-boundary analog (Appendix A; constructed via their B2 formula).
    y_tilde : float
        Minimum inter-event time as fraction of total observation period T (0 <= y~ <= 1/(n or n+1)).
    ddof : int
        Passed to std.

    Returns
    -------
    BurstinessResult with method tags 'kimjo2016_periodic_min' or 'kimjo2016_open_min' and value in `.b`.
    """
    n, mean_dt, sd_dt, cv = _cv(np.asarray(inter_event_durs, dtype=float), ddof=ddof)
    if not np.isfinite(cv):
        tag = "kimjo2016_periodic_min" if method == "periodic" else "kimjo2016_open_min"
        return BurstinessResult(method=tag, n=n, mean_dt=mean_dt, sd_dt=sd_dt, cv=cv, b=float("nan"))

    if method == "periodic":
        # Eq. (28): A_{n, y~}(r)
        if n < 2:
            b = float("nan")
        else:
            r = cv
            sqrt_np1 = np.sqrt(n + 1.0)
            sqrt_nm1 = np.sqrt(n - 1.0)
            term = (1.0 - n * y_tilde)
            num = (n - 2.0) * (sqrt_np1 * r - sqrt_nm1 * term)
            den = ((n * sqrt_np1) - 2.0 * (n - 1.0)) * r + sqrt_nm1 * (n - 2.0 * sqrt_np1) * term
            b = num / den if den != 0 else float("nan")
        tag = "kimjo2016_periodic_min"
    elif method == "open":
        # Open-boundary analog via general B2 with:
        # r_minus = 1 - (n+1) y~ / sqrt(n)
        # r_0     = sqrt(n)/(n+2) * (1 - (n+1) y~)
        # r_plus  = sqrt(n) * (1 - (n+1) y~)
        # A(r) = (r_plus - r_minus)(r - r0) / ((r_plus + r_minus - 2 r0) r + (r_plus + r_minus) r0 - 2 r_plus r_minus)
        if n < 1:
            b = float("nan")
        else:
            r = cv
            sqn = np.sqrt(n * 1.0)
            term = (1.0 - (n + 1.0) * y_tilde)
            r_minus = (1.0 - (n + 1.0) * y_tilde) / sqn
            r_zero = (sqn / (n + 2.0)) * term
            r_plus = sqn * term
            num = (r_plus - r_minus) * (r - r_zero)
            den = (r_plus + r_minus - 2.0 * r_zero) * r + (r_plus + r_minus) * r_zero - 2.0 * r_plus * r_minus
            b = num / den if den != 0 else float("nan")
        tag = "kimjo2016_open_min"
    else:
        raise ValueError("method must be 'periodic' or 'open'")

    return BurstinessResult(method=tag, n=n, mean_dt=mean_dt, sd_dt=sd_dt, cv=cv, b=float(b))

# ---------- Memory coefficient ----------

def memory_coefficient(
    inter_event_durs: Union[Sequence[float], np.ndarray],
    lag: int = 1,
) -> float:
    """
    Lag-k Pearson correlation of consecutive inter-event durations; defaults to lag-1 (M).
    """
    x = np.asarray(inter_event_durs, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < lag + 1:
        return float("nan")
    x1 = x[:-lag]
    x2 = x[lag:]
    if x1.size < 2:
        return float("nan")
    m1 = x1.mean()
    m2 = x2.mean()
    s1 = x1.std(ddof=0)
    s2 = x2.std(ddof=0)
    if s1 == 0 or s2 == 0:
        return 0.0
    return float(np.corrcoef(x1, x2)[0, 1])

# ---------- Rolling windows by event count ----------

def rolling_burstiness_by_count(
    timestamps: Union[Sequence[pd.Timestamp], pd.Series, Sequence[str]],
    window_size: int,
    step: Optional[int] = None,
    unit: TimeUnit = "s",
    method: Method = "goh",
    min_valid: int = 3,
    sort: bool = True,
    include_intervals: bool = False,
    compute_memory: bool = False,
    memory_lag: int = 1,
    event_ids: Optional[Sequence] = None
) -> pd.DataFrame:
    if window_size < 2:
        raise ValueError("window_size must be >= 2")
    if step is None:
        step = max(1, window_size // 2)

    ts = pd.to_datetime(pd.Series(timestamps).dropna())
    if sort:
        ts = ts.sort_values(kind="mergesort").reset_index(drop=True)

    n = len(ts)
    results = []
    win_idx = 0
    for start in range(0, n - window_size + 1, step):
        end = start + window_size  # exclusive
        slice_ts = ts.iloc[start:end].reset_index(drop=True)
        diffs = slice_ts.diff().iloc[1:]
        durs = diffs.dt.total_seconds().to_numpy(dtype=float)
        durs_unit = _as_unit(durs, unit)
        n_intervals = len(durs_unit)
        if n_intervals >= min_valid:
            res = burstiness(durs_unit, method=method)
            b_val = res.b
            mean_dt = res.mean_dt
            sd_dt = res.sd_dt
            cv = res.cv
            M = memory_coefficient(durs_unit, lag=memory_lag) if compute_memory else np.nan
        else:
            b_val = np.nan
            mean_dt = np.nan
            sd_dt = np.nan
            cv = np.nan
            M = np.nan
        row = {
            "window_index": win_idx,
            "start_idx": start,
            "end_idx": end - 1,
            "start_time": slice_ts.iloc[0],
            "end_time": slice_ts.iloc[-1],
            "n_events": window_size,
            "n_intervals": n_intervals,
            "mean_dt": mean_dt,
            "sd_dt": sd_dt,
            "cv": cv,
            "r": cv,
            "b": b_val,
            "M": M,
        }
        # Add the event ID from the leading edge (end of window)
        if event_ids is not None:
            row["event_id"] = event_ids[end - 1]  # end-1 is the last event in window
        if include_intervals:
            row["intervals"] = durs_unit.tolist()
        results.append(row)
        win_idx += 1

    return pd.DataFrame(results)

def rolling_inter_event_durations_by_count(
    timestamps: Union[Sequence[pd.Timestamp], pd.Series, Sequence[str]],
    window_size: int,
    step: Optional[int] = None,
    unit: TimeUnit = "s",
    sort: bool = True,
) -> pd.DataFrame:
    if window_size < 2:
        raise ValueError("window_size must be >= 2")
    if step is None:
        step = max(1, window_size // 2)

    ts = pd.to_datetime(pd.Series(timestamps).dropna())
    if sort:
        ts = ts.sort_values(kind="mergesort").reset_index(drop=True)

    n = len(ts)
    rows = []
    win_idx = 0
    for start in range(0, n - window_size + 1, step):
        end = start + window_size
        slice_ts = ts.iloc[start:end].reset_index(drop=True)
        diffs = slice_ts.diff().iloc[1:]
        durs = diffs.dt.total_seconds().to_numpy(dtype=float)
        durs_unit = _as_unit(durs, unit)
        for i, dt in enumerate(durs_unit):
            rows.append({
                "window_index": win_idx,
                "interval_index": i,
                "start_idx": start,
                "end_idx": end - 1,
                "start_time": slice_ts.iloc[0],
                "end_time": slice_ts.iloc[-1],
                "dt": dt,
            })
        win_idx += 1

    return pd.DataFrame(rows)

# Convenience: dataframe-friendly wrapper
def compute_temporal_features(
    df: pd.DataFrame,
    time_col: str = "event_date",
    window_size: Optional[int] = None,
    step: Optional[int] = None,
    unit: TimeUnit = "s",
    method: Method = "goh",
    min_valid: int = 3,
    return_windows: bool = True,
    compute_memory: bool = False,
    memory_lag: int = 1,
    event_ids: str = 'eid'
) -> dict:
    if time_col not in df.columns:
        raise KeyError(f"{time_col!r} not found in df columns")
    # inter = inter_event_durations(df[time_col], sort=True, unit=unit)
    # out = {"inter_event_durations": inter}
    if window_size is not None and return_windows:
        rb = rolling_burstiness_by_count(
            df[time_col],
            window_size=window_size,
            step=step,
            unit=unit,
            method=method,
            min_valid=min_valid,
            sort=True,
            include_intervals=False,
            compute_memory=compute_memory,
            memory_lag=memory_lag,
            event_ids=df[event_ids]
        )
        # out["rolling_burstiness"] = rb
        out["rolling_burstiness"] = None
        return rb
    else:
        out["rolling_burstiness"] = None
    return out
