"""Shared fixtures. Tests run fully offline using the synthetic provider.

Isolation from YOUR environment: the project's .env (broker keys, BROKER=alpaca_paper,
TRADING_ENABLED, database URL, LLM keys) is never loaded during tests, and those
variables are removed from the test process. A test can therefore never reach your
Alpaca paper account, your database or an LLM API.
"""
from __future__ import annotations

import os

os.environ["TRADING_BOT_IGNORE_DOTENV"] = "1"
for _var in ("BROKER", "ALPACA_API_KEY", "ALPACA_API_SECRET", "ALPACA_BASE_URL", "TRADING_ENABLED",
             "TRADING_MODE", "DATABASE_URL", "LLM_PROVIDER", "LLM_MODEL", "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
             "OLLAMA_BASE_URL"):
    os.environ.pop(_var, None)

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.data.corporate_actions import apply_adjustment
from src.data.provider import SyntheticProvider

SYMBOLS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"]


@pytest.fixture
def cfg():
    return load_config(overrides={"settings": {
        "data": {"provider": "synthetic"},
        "universe": {"symbols": SYMBOLS, "benchmark": "SPY"},
    }})


@pytest.fixture
def provider():
    return SyntheticProvider(seed=7)


@pytest.fixture
def prices(provider):
    """Adjusted daily bars for a small universe, 2016-2020."""
    return {s: apply_adjustment(provider.get_history(s, "2016-01-01", "2020-12-31"), "adjusted",
                                provider.volume_split_adjusted) for s in SYMBOLS}


def make_bars(closes, volume=1_000_000.0, start="2020-01-01"):
    """Simple OHLCV frame from a list of closes (open = previous close)."""
    closes = np.asarray(closes, dtype=float)
    idx = pd.bdate_range(start, periods=len(closes), name="date")
    opens = np.concatenate([[closes[0]], closes[:-1]])
    df = pd.DataFrame({
        "open": opens,
        "high": np.maximum(opens, closes) * 1.01,
        "low": np.minimum(opens, closes) * 0.99,
        "close": closes,
        "adj_close": closes,
        "volume": volume,
        "dividends": 0.0,
        "splits": 0.0,
    }, index=idx)
    return df
