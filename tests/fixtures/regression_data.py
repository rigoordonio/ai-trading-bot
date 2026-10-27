"""Deterministic (formula-based, NOT random) price paths for strategy regression tests."""
import numpy as np
import pandas as pd


def regression_prices() -> dict[str, pd.DataFrame]:
    n = 520
    t = np.arange(n)
    paths = {
        # steady uptrend, a crash, then recovery
        "UPDN": np.concatenate([np.linspace(100, 180, 300), np.linspace(180, 130, 60), np.linspace(130, 170, 160)]),
        # trend + slow sine: produces dips (mean-reversion) and breakouts
        "WAVE": 100 + 0.08 * t + 8 * np.sin(t / 9.0),
        # benchmark-like gentle uptrend
        "SPY": 300 + 0.15 * t + 2 * np.sin(t / 25.0),
    }
    out = {}
    idx = pd.bdate_range("2019-01-01", periods=n, name="date")
    for s, c in paths.items():
        o = np.concatenate([[c[0]], c[:-1]])
        vol = np.full(n, 2_000_000.0)
        vol[::37] = 6_000_000.0          # periodic volume spikes
        out[s] = pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.002, "low": np.minimum(o, c) * 0.998,
                               "close": c, "adj_close": c, "volume": vol, "dividends": 0.0, "splits": 0.0},
                              index=idx)
    return out
