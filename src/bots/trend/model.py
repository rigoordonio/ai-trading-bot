"""trend_model_v1: probability that the forward N-day return (N in 20-60)
beats a configured threshold. ONE target per model version: 20d and 60d
targets are never mixed in one training set (no leakage between them)."""
MODEL_ID = "trend_model_v1"
TARGET_HORIZON_RANGE = (20, 60)
