"""High-level workflows shared by scripts, the API and the dashboard."""
from __future__ import annotations

import pandas as pd

from src.backtesting.engine import BacktestEngine, BacktestResult, BotInput, StrategyBot
from src.backtesting.walk_forward import WalkForwardResult, make_windows, walk_forward_predictions
from src.bots.bot import TradingBot, build_bots
from src.config import AppConfig, resolve_path
from src.data.downloader import ETLResult, build_downloader
from src.data.provider import get_provider
from src.data.universe import Universe
from src.execution.simulator import CostModel
from src.features.pipeline import pipeline_from_config
from src.logging_setup import get_logger
from src.models.prediction import save_model
from src.models.training import TrainResult, train_select_evaluate
from src.strategies.ensemble import EnsembleStrategy, build_rule_strategies, build_strategy

log = get_logger(__name__)


def load_market_data(cfg: AppConfig, provider_name: str | None = None, save: bool = True, end=None) -> ETLResult:
    provider = get_provider(provider_name or cfg.settings.data.provider)
    universe = Universe.from_config(cfg.settings.universe)
    dl = build_downloader(cfg, provider, save=save)
    res = dl.run(universe.all_symbols, cfg.settings.data.start, end or cfg.settings.data.end)
    if not res.data:
        raise RuntimeError(f"no market data could be loaded: {res.failed}")
    return res


def research_backtest(cfg: AppConfig, prices: dict[str, pd.DataFrame], strategy: str = "ensemble", start=None,
                      end=None, cost_multiplier: float = 1.0, params: dict | None = None) -> BacktestResult:
    """Backtest ONE rule strategy (or the rule-only classic ensemble) with 100% of capital."""
    panel = pipeline_from_config(cfg).compute(prices)
    universe = Universe.from_config(cfg.settings.universe)
    if strategy == "ensemble":
        strats = build_rule_strategies(cfg)
        ens_cfg = cfg.strategies.ensemble.model_copy(update={"weights": {
            k: v for k, v in cfg.strategies.ensemble.weights.items() if k in strats}})
        sf = EnsembleStrategy(strats, ens_cfg).signal_frame(panel)
        feats = sorted({c for s in strats.values() for c in s.required_features})
    else:
        scfg = cfg.strategies.strategies[strategy]
        if params:
            scfg = scfg.model_copy(update={"params": {**scfg.params, **params}})
        strat = build_strategy(strategy, scfg)
        sf = strat.signal_frame(panel)
        feats = list(strat.required_features)
    bot = StrategyBot(strategy, universe.symbols, cfg.risk, feats, cfg.settings.backtest.rebalance_frequency,
                      pipeline_from_config(cfg).version)
    costs = CostModel.from_config(cfg.settings.backtest.costs).scaled(cost_multiplier)
    return BacktestEngine(cfg, prices, [BotInput(bot, panel, sf)], start, end, cost_model=costs,
                          universe=universe).run()


def bot_walk_forward(cfg: AppConfig, bot: TradingBot, panel: pd.DataFrame) -> WalkForwardResult | None:
    if not bot.ml_cfg:
        return None
    wf = cfg.settings.walk_forward
    dates = panel.index.get_level_values("date")
    windows = make_windows(dates.min(), dates.max(), wf.train_years, wf.validation_years, wf.test_years,
                           wf.step_years)
    return walk_forward_predictions(panel, bot.ml_features, bot.ml_cfg.target_horizon_days,
                                    bot.ml_cfg.target_threshold, windows, bot.ml_cfg.models,
                                    cfg.settings.ml.selection_metric, cfg.settings.ml.purge_days,
                                    cfg.settings.ml.random_state)


def bots_backtest(cfg: AppConfig, prices: dict[str, pd.DataFrame], bot_ids: list[str] | None = None, start=None,
                  end=None, cost_multiplier: float = 1.0) -> tuple[BacktestResult, dict]:
    """Backtest bots on one account. ML components use walk-forward
    OUT-OF-SAMPLE predictions only; before the first OOS date ML bots HOLD."""
    bots = build_bots(cfg, bot_ids)
    if not bots:
        raise ValueError("no enabled bots selected")
    inputs, wf_results = [], {}
    for bot_id, bot in bots.items():
        panel = bot.compute_features(prices)
        wf = bot_walk_forward(cfg, bot, panel)
        wf_results[bot_id] = wf
        sf = bot.signal_frame(panel, wf.predictions if wf is not None else None)
        inputs.append(BotInput(bot, panel, sf))
    costs = CostModel.from_config(cfg.settings.backtest.costs).scaled(cost_multiplier)
    res = BacktestEngine(cfg, prices, inputs, start, end, cost_model=costs,
                         universe=Universe.from_config(cfg.settings.universe)).run()
    return res, wf_results


def train_bot_model(cfg: AppConfig, bot: TradingBot, prices: dict[str, pd.DataFrame]) -> TrainResult:
    """Train / select / test one bot's model with the configured chronological
    split, then save it with its feature-set version."""
    m = bot.ml_cfg
    if m is None:
        raise ValueError(f"{bot.bot_id} has no ML component")
    panel = bot.compute_features(prices)
    s = cfg.settings.ml.split
    split = {"train": list(s.train), "validation": list(s.validation), "test": list(s.test)}
    tr = train_select_evaluate(panel, bot.ml_features, m.target_horizon_days, m.target_threshold, split, m.models,
                               cfg.settings.ml.selection_metric, cfg.settings.ml.purge_days,
                               cfg.settings.ml.random_state)
    path = save_model(tr.model, resolve_path(cfg.settings.ml.artifacts_dir), bot_id=bot.bot_id,
                      model_version=m.model_version, feature_set_version=bot.feature_version,
                      features=bot.ml_features,
                      target={"horizon_days": m.target_horizon_days, "threshold": m.target_threshold},
                      periods=tr.periods, metrics={"validation": tr.validation_scores, "test": tr.test_metrics,
                                                   **tr.extra}, model_name=tr.best_name,
                      model_id=getattr(bot, "MODEL_ID", None))
    log.info("saved %s model %s (%s) to %s", bot.bot_id, m.model_version, tr.best_name, path)
    return tr
