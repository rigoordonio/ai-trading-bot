"""Point-of-decision validation: the bar and features an order would be based on."""
import numpy as np
import pandas as pd
import pytest

from src.errors import DataValidationError, FeatureSchemaError, StaleMarketDataError
from src.features.pipeline import FEATURE_SET_VERSION, FeatureConfig, FeaturePipeline
from src.safety.validation import validate_bar_values, validate_current_bar, validate_feature_row
from tests.conftest import make_bars

T = pd.Timestamp("2024-03-01")


def good():
    return pd.Series({"open": 100.0, "high": 102.0, "low": 99.0, "close": 101.0, "volume": 1e6})


def test_good_bar_passes():
    validate_bar_values("X", good(), T, T, prev_close=100.0)


def test_missing_prices():
    b = good().drop("close")
    with pytest.raises(DataValidationError, match="close"):
        validate_bar_values("X", b, T, T)
    with pytest.raises(DataValidationError):
        validate_bar_values("X", {}, T, T)
    with pytest.raises(DataValidationError, match="no market data"):
        validate_current_bar("X", pd.DataFrame(), T)


@pytest.mark.parametrize("field", ["open", "high", "low", "close", "volume"])
def test_nan_prices(field):
    b = good()
    b[field] = np.nan
    with pytest.raises(DataValidationError):
        validate_bar_values("X", b, T, T)


@pytest.mark.parametrize("field", ["open", "high", "low", "close"])
def test_infinite_values(field):
    b = good()
    b[field] = np.inf
    with pytest.raises(DataValidationError):
        validate_bar_values("X", b, T, T)


def test_negative_prices_and_volume():
    for f in ("open", "low", "close"):
        b = good()
        b[f] = -1.0
        with pytest.raises(DataValidationError):
            validate_bar_values("X", b, T, T)
    b = good()
    b["volume"] = -5
    with pytest.raises(DataValidationError, match="volume"):
        validate_bar_values("X", b, T, T)


@pytest.mark.parametrize("o,h,low,c", [(100, 99, 98, 98.5), (100, 101, 102, 101), (103, 102, 99, 101),
                                       (100, 102, 99, 103), (98, 102, 99, 101), (100, 102, 101.5, 101)])
def test_ohlc_consistency(o, h, low, c):
    with pytest.raises(DataValidationError, match="OHLC"):
        validate_bar_values("X", pd.Series({"open": o, "high": h, "low": low, "close": c, "volume": 1}), T, T)


def test_stale_data():
    with pytest.raises(StaleMarketDataError):
        validate_bar_values("X", good(), T - pd.Timedelta(days=7), T, max_staleness_days=4)
    validate_bar_values("X", good(), T - pd.Timedelta(days=3), T, max_staleness_days=4)   # a long weekend is fine


def test_future_bar_and_timezone_rejected():
    with pytest.raises(DataValidationError, match="after decision"):
        validate_bar_values("X", good(), T + pd.Timedelta(days=1), T)
    with pytest.raises(DataValidationError, match="timezone"):
        validate_bar_values("X", good(), pd.Timestamp("2024-03-01", tz="UTC"), T)
    df = make_bars([100, 101, 102])
    df.index = df.index.tz_localize("UTC")
    with pytest.raises(DataValidationError, match="timezone"):
        validate_current_bar("X", df, df.index[-1])


def test_duplicate_timestamps():
    df = make_bars([100, 101, 102])
    df = pd.concat([df, df.iloc[[-1]]])
    with pytest.raises(DataValidationError, match="duplicate"):
        validate_current_bar("X", df, df.index[-1])


def test_extreme_discontinuity():
    with pytest.raises(DataValidationError, match="discontinuity"):
        validate_bar_values("X", good(), T, T, prev_close=50.0)


# ------------------------------------------------------------------ features
def test_feature_schema_and_nan_handling():
    row = pd.Series({"a": 1.0, "b": 2.0})
    validate_feature_row("X", row, ["a", "b"])
    with pytest.raises(FeatureSchemaError, match="missing"):
        validate_feature_row("X", row, ["a", "c"])
    with pytest.raises(FeatureSchemaError, match="NaN"):
        validate_feature_row("X", pd.Series({"a": np.nan, "b": 1.0}), ["a", "b"])
    with pytest.raises(FeatureSchemaError, match="NaN"):
        validate_feature_row("X", pd.Series({"a": np.inf, "b": 1.0}), ["a", "b"])
    with pytest.raises(FeatureSchemaError, match="look-ahead"):
        validate_feature_row("X", row, ["a"], as_of=T, row_date=T + pd.Timedelta(days=1))


def test_feature_version():
    a, b = FeatureConfig(), FeatureConfig(sma_windows=[10, 20, 50, 200])
    assert a.version().startswith(FEATURE_SET_VERSION + "+")
    assert a.version() == FeatureConfig().version()            # deterministic
    assert a.version() != b.version()                           # parameters change the schema version
    with pytest.raises(Exception, match="unknown feature"):
        FeatureConfig.from_dict({"sma_window": [20]})            # typo fails loudly


def test_all_pipeline_features_numeric(prices):
    panel = FeaturePipeline().compute(prices)
    bad = [c for c in panel.columns if panel[c].dtype.kind not in "fiu"]
    assert not bad, f"non-numeric features: {bad}"


def test_current_bar_validation_uses_last_bar_up_to_as_of():
    df = make_bars([100, 101, 102, 103])
    bar = validate_current_bar("X", df, df.index[2])
    assert bar["close"] == 102                                   # never a later bar
