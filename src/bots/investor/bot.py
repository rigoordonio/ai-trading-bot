import pandas as pd

from src.bots.base import BaseTradingBot
from src.bots.investor import features, model, strategy


class InvestorBot(BaseTradingBot):
    BOT_TYPE = "investor"
    TIMEFRAME = "weekly"
    ALLOWED_COMPONENTS = strategy.ALLOWED_COMPONENTS
    TARGET_HORIZON_RANGE = model.TARGET_HORIZON_RANGE
    ML_FEATURE_PREFIX = model.ML_FEATURE_PREFIX
    REQUIRED_FEATURES = features.REQUIRED
    BOT_FEATURES_VERSION = features.FEATURES_VERSION
    MODEL_ID = model.MODEL_ID
    SCORE_NAME = strategy.SCORE_NAME

    def add_weekly_features(self, weekly_panel: pd.DataFrame) -> pd.DataFrame:
        return features.add_weekly_features(weekly_panel)

    def validate_contract(self) -> None:
        super().validate_contract()
        from src.errors import ConfigurationError
        if self.weekly_pipeline is None:
            raise ConfigurationError(f"{self.bot_id}: the investor bot needs weekly_features")
        prefixed = [n for n, c in self.cfg.components.items() if c.enabled and c.params.get("prefix") != "w_"]
        if prefixed:
            raise ConfigurationError(f"{self.bot_id}: components {prefixed} must use weekly (w_) features")
