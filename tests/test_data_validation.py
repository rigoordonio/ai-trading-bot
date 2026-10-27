import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.data.corporate_actions import apply_adjustment, split_volume_factor
from src.data.downloader import DataPipelineError, MarketDataDownloader, with_retries
from src.data.provider import MarketDataProvider, SyntheticProvider, standardise
from src.data.universe import Universe
from src.data.validator import DataValidator
from src.errors import ConfigurationError
from tests.conftest import make_bars


def checks(report):
    return {i.check: i for i in report.issues}


def test_clean_data_passes():
    df = make_bars(100 + np.cumsum(np.random.default_rng(0).normal(0, 1, 300)).clip(-50))
    rep = DataValidator().validate(df, "X")
    assert rep.ok, rep.summary()


def test_detects_duplicates_nulls_nonpositive_and_impossible_ohlc():
    df = make_bars(np.linspace(100, 120, 50))
    df = pd.concat([df, df.iloc[[10]]]).sort_index()            # duplicate
    df.iloc[20, df.columns.get_loc("close")] = np.nan            # null
    df.iloc[25, df.columns.get_loc("low")] = -1.0                # non-positive
    df.iloc[30, df.columns.get_loc("high")] = df.iloc[30]["low"] * 0.5  # high < low
    rep = DataValidator().validate(df, "X")
    c = checks(rep)
    assert not rep.ok
    for name in ("duplicate_timestamps", "null_values", "non_positive_price", "impossible_ohlc"):
        assert name in c and c[name].severity == "ERROR"
    cleaned = DataValidator.clean(df, rep)
    assert cleaned.index.is_unique
    assert DataValidator().validate(cleaned, "X").ok


def test_detects_abnormal_price_volume_stale_and_gaps():
    closes = list(np.linspace(100, 110, 120))
    closes[60] = closes[59] * 2.0                    # +100% move
    closes[90:97] = [closes[89]] * 7                 # stale run
    df = make_bars(closes)
    df.iloc[100, df.columns.get_loc("volume")] = 1e12  # volume spike
    df = df.drop(df.index[40:48])                    # 8 business-day gap
    c = checks(DataValidator().validate(df, "X"))
    assert c["abnormal_price_move"].severity == "WARNING"
    assert "abnormal_volume" in c
    assert "stale_prices" in c
    assert "missing_timestamps" in c


def test_split_is_reported_as_corporate_action_not_abnormal_move():
    p = SyntheticProvider(seed=1, splits={"AAA": ("2020-06-01", 4.0)})
    raw = p.get_history("AAA", "2020-01-01", "2020-12-31")
    c = checks(DataValidator().validate(raw, "AAA"))
    assert "corporate_action_detected" in c
    assert "abnormal_price_move" not in c


def test_adjustment_removes_split_discontinuity_and_keeps_raw():
    p = SyntheticProvider(seed=1, splits={"AAA": ("2020-06-01", 4.0)})
    raw = p.get_history("AAA", "2020-01-01", "2020-12-31")
    adj = apply_adjustment(raw, "adjusted", volume_split_adjusted=False)
    assert raw["close"].pct_change().min() < -0.70          # fake crash in raw data
    assert adj["close"].pct_change().abs().max() < 0.2      # gone after adjustment
    assert "raw_close" in adj and np.allclose(adj["raw_close"], raw["close"])
    # Volume pre-split is scaled up by the split ratio, back to post-split units.
    pre = adj.index < "2020-06-01"
    assert np.allclose(adj.loc[pre, "volume"], raw.loc[pre, "volume"] * 4.0)
    # Latest bar is unadjusted (factor 1) so current prices match quotes.
    assert adj["adj_factor"].iloc[-1] == pytest.approx(1.0)


def test_split_volume_factor_only_counts_future_splits():
    df = make_bars([10] * 5)
    df["splits"] = [0, 0, 2.0, 0, 0]
    assert list(split_volume_factor(df)) == [2.0, 2.0, 1.0, 1.0, 1.0]


def test_synthetic_provider_is_deterministic_across_ranges():
    p = SyntheticProvider(seed=3)
    a = p.get_history("MSFT", "2018-01-01", "2019-12-31")
    b = SyntheticProvider(seed=3).get_history("MSFT", "2019-01-01", "2020-12-31")
    overlap = a.index.intersection(b.index)
    assert len(overlap) > 200
    raw_cols = ["open", "high", "low", "close", "volume"]
    pd.testing.assert_frame_equal(a.loc[overlap, raw_cols], b.loc[overlap, raw_cols])
    # adj_close is re-based to the last delivered bar, but its returns are identical
    assert np.allclose(a.loc[overlap, "adj_close"].pct_change().dropna(),
                       b.loc[overlap, "adj_close"].pct_change().dropna())


def test_standardise_vendor_columns():
    vendor = pd.DataFrame({"Open": [1.0], "High": [2.0], "Low": [0.5], "Close": [1.5], "Adj Close": [1.4],
                           "Volume": [100], "Stock Splits": [0.0]},
                          index=pd.DatetimeIndex(["2024-01-02 00:00:00-05:00"]))
    out = standardise(vendor)
    assert list(out.columns) == ["open", "high", "low", "close", "adj_close", "volume", "dividends", "splits"]
    assert out.index.tz is None and out.index.name == "date"


class FlakyProvider(MarketDataProvider):
    name = "flaky"

    def __init__(self, fail_times):
        self.calls = 0
        self.fail_times = fail_times

    def get_history(self, symbol, start, end=None, interval="1d"):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise ConnectionError("temporary outage")
        return SyntheticProvider().get_history(symbol, start, end)


def test_retry_policy_recovers_and_gives_up(tmp_path):
    ok = FlakyProvider(fail_times=2)
    dl = MarketDataDownloader(ok, DataValidator(), tmp_path / "raw", tmp_path / "proc", retries=3, backoff=0.0)
    res = dl.run(["SPY"], "2020-01-01", "2020-03-31")
    assert "SPY" in res.data and ok.calls == 3
    assert (tmp_path / "raw" / "flaky" / "SPY.csv").exists()     # raw stored
    assert (tmp_path / "proc" / "SPY.csv").exists()              # processed stored

    bad = FlakyProvider(fail_times=10)
    dl2 = MarketDataDownloader(bad, DataValidator(), tmp_path / "raw", tmp_path / "proc", retries=2, backoff=0.0)
    res2 = dl2.run(["SPY"], "2020-01-01", "2020-03-31")
    assert "SPY" in res2.failed


def test_with_retries_raises_pipeline_error():
    with pytest.raises(DataPipelineError):
        with_retries(lambda: 1 / 0, retries=2, backoff=0.0)


def test_universe_point_in_time_membership():
    m = pd.DataFrame({"symbol": ["AAA", "BBB"],
                      "start_date": pd.to_datetime(["2015-01-01", "2018-01-01"]),
                      "end_date": pd.to_datetime(["2017-12-31", None])})
    u = Universe(["AAA", "BBB"], membership=m)
    assert u.is_active("AAA", pd.Timestamp("2016-06-01"))
    assert not u.is_active("AAA", pd.Timestamp("2019-01-01"))      # delisted / removed
    assert not u.is_active("BBB", pd.Timestamp("2016-06-01"))      # not yet a member
    mask = u.active_mask(pd.bdate_range("2017-12-28", "2018-01-03"), ["AAA", "BBB"])
    assert mask.loc["2017-12-29", "AAA"] and not mask.loc["2017-12-29", "BBB"]
    assert mask.loc["2018-01-02", "BBB"] and not mask.loc["2018-01-02", "AAA"]


def test_config_rejects_live_mode_and_bad_splits():
    with pytest.raises(ConfigurationError, match="Live trading"):
        load_config(overrides={"settings": {"mode": "live"}})
    with pytest.raises(ConfigurationError):
        load_config(overrides={"settings": {"ml": {"split": {"train": ["2015-01-01", "2023-01-01"]}}}})
    with pytest.raises(ConfigurationError):
        load_config(overrides={"settings": {"backtest": {"execution_delay_bars": 0}}})
