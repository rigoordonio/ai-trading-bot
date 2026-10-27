"""investor_model_v1: probability that the forward ~12-26 week return beats
a configured threshold. Uses weekly (w_) features ONLY. The horizon is in
trading days of the daily calendar the decisions run on (12 weeks ~ 60)."""
MODEL_ID = "investor_model_v1"
TARGET_HORIZON_RANGE = (55, 130)
ML_FEATURE_PREFIX = "w_"
