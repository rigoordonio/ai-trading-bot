"""INVESTOR_V1 strategy logic: long-term momentum, weekly trend, low
volatility and a drawdown filter, all on weekly (w_) features."""
ALLOWED_COMPONENTS = frozenset({"long_term_momentum", "trend_following", "low_volatility", "drawdown_filter"})
SCORE_NAME = "LONG_TERM_INVESTMENT_SCORE"
