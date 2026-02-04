import os, sys, types, importlib.util, numpy as np, pandas as pd
HERE = os.path.dirname(__file__)
EWS_PATH = os.path.join(HERE, 'ews.py')

def load_ews(path):
    spec = importlib.util.spec_from_file_location('ews_mod', path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['ews_mod'] = mod
    spec.loader.exec_module(mod)
    return mod

def simulate_irregular(n=600, seed=7):
    rng = np.random.default_rng(seed)
    waits = rng.exponential(scale=60.0, size=n)
    t = np.cumsum(waits)
    phi = 0.7
    e = rng.normal(scale=1.0, size=n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i-1] + e[i]
    trend = 0.002 * t
    y = trend + x
    return pd.Series(y), pd.to_datetime(t, unit='s')

def pick_attr(obj, names):
    for name in names:
        if hasattr(obj, name):
            return name
    return None

def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def main():
    ews = load_ews(EWS_PATH)
    required = ['EWSConfig', 'rolling_or_expanding_ews', 'detrend_series']
    missing = [n for n in required if not hasattr(ews, n)]
    if missing:
        raise AssertionError(f'Missing expected symbols: {missing}')

    cfg = ews.EWSConfig()
    detrend_field = pick_attr(cfg, ['detrend', 'detrend_method', 'detrend_mode'])
    if detrend_field:
        setattr(cfg, detrend_field, 'moving_average')
    for name in ['ma_window','moving_average_window','window','ma_window_size']:
        if hasattr(cfg, name):
            setattr(cfg, name, 31)
            break

    y, ts = simulate_irregular(n=800, seed=42)
    out = ews.rolling_or_expanding_ews(values=y.to_numpy(), timestamps=ts, window_size=120, step=30, mode='rolling', cfg=cfg)

    synonyms = {
        'variance': ['variance','var'],
        'std': ['std','stdev','std_dev'],
        'skew': ['skew','skewness'],
        'kurtosis': ['kurtosis','excess_kurtosis','kurt'],
        'acf1': ['acf1','acf_1'],
        'ar1': ['ar1','phi1','phi_1'],
        'return_rate': ['return_rate','recovery_rate','rr'],
        'spectral_ratio': ['spectral_ratio','psd_ratio','low_high_ratio'],
        'spectral_slope': ['spectral_slope','psd_slope','beta'],
        'dfa_alpha': ['dfa_alpha','dfa','alpha_dfa']
    }

    for base in ['window_index','start_time','end_time','n_events']:
        if base not in out.columns:
            raise AssertionError(f'Missing required window meta column: {base}')

    present = {}
    for k, cand in synonyms.items():
        col = find_col(out, cand)
        if col:
            present[k] = col

    if len(present) < 7:
        raise AssertionError(f'Too few indicator columns; found {len(present)}: {present}')

    for k in ['ar1','return_rate','variance','spectral_ratio','dfa_alpha']:
        col = present.get(k)
        if not col:
            continue
        vals = out[col].to_numpy()
        finite = np.isfinite(vals)
        if finite.sum() < max(3, int(0.6*len(vals))):
            raise AssertionError(f'Too many non-finite values in {k}: {finite.sum()} of {len(vals)} finite.')

    print('OK: core structure & indicators present.')
    print('Rows:', len(out), '| Columns:', list(out.columns))
    print('Resolved indicators:', present)
    print(out.head(3).to_string(index=False))

if __name__ == '__main__':
    main()
