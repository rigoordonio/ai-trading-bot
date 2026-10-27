from src.bots.base import BaseTradingBot
from src.bots.swing import features, model, strategy


class SwingBot(BaseTradingBot):
    BOT_TYPE = "swing"
    TIMEFRAME = "daily"
    ALLOWED_COMPONENTS = strategy.ALLOWED_COMPONENTS
    TARGET_HORIZON_RANGE = model.TARGET_HORIZON_RANGE
    REQUIRED_FEATURES = features.REQUIRED
    BOT_FEATURES_VERSION = features.FEATURES_VERSION
    MODEL_ID = model.MODEL_ID
    SCORE_NAME = strategy.SCORE_NAME
