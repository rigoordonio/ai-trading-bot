"""SWING_V1 features: short-horizon price action on DAILY bars.

All columns are produced by the shared FeaturePipeline from the parameters
in config/bots/swing.yaml; this module pins the set the swing design
depends on so a config edit cannot silently drop one.
"""
FEATURES_VERSION = "swing-f1"

REQUIRED = (
    "ret_1d", "ret_5d", "ret_10d", "ret_20d",          # 1/5/10-day returns, 20-day momentum
    "sma_10", "sma_20", "sma_50", "ema_12", "ema_26",
    "rsi_14", "macd", "macd_signal", "atr_14",
    "bb_upper", "bb_lower", "bb_pct",                  # Bollinger Bands
    "realized_vol_20", "volume_ratio", "volume_zscore",
    "dist_high_20", "dist_sma_50",                     # distance from 20d high / 50d MA
)
