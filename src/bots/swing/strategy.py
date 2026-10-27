"""SWING_V1 strategy logic: short-term momentum, breakout, controlled mean
reversion and volume confirmation. Weights and thresholds are in
config/bots/swing.yaml and must be validated with walk-forward testing."""
ALLOWED_COMPONENTS = frozenset({"momentum", "breakout", "mean_reversion", "volume_confirmation"})
SCORE_NAME = "SHORT_TERM_SIGNAL_SCORE"
