"""swing_model_v1: probability that the forward 5-day return beats a
threshold (configured; not assumed to be zero). Estimates a probability
only; it never sizes or submits orders."""
MODEL_ID = "swing_model_v1"
TARGET_HORIZON_RANGE = (3, 10)        # trading days; ~5 initially
