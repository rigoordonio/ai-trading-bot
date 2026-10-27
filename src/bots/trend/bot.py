import pandas as pd

from src.bots.base import BaseTradingBot
from src.bots.trend import features, model, strategy


class TrendBot(BaseTradingBot):
    BOT_TYPE = "trend"
    TIMEFRAME = "daily"
    ALLOWED_COMPONENTS = strategy.ALLOWED_COMPONENTS
    TARGET_HORIZON_RANGE = model.TARGET_HORIZON_RANGE
    REQUIRED_FEATURES = features.REQUIRED
    BOT_FEATURES_VERSION = features.FEATURES_VERSION
    MODEL_ID = model.MODEL_ID
    SCORE_NAME = strategy.SCORE_NAME

    def add_bot_features(self, panel: pd.DataFrame) -> pd.DataFrame:
        return features.add_features(panel)
