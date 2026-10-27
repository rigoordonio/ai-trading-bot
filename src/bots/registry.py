"""bot_type -> bot class. One platform, several independent bots."""
from __future__ import annotations

from src.bots.base import BaseTradingBot
from src.bots.investor import InvestorBot
from src.bots.swing import SwingBot
from src.bots.trend import TrendBot
from src.config import AppConfig
from src.errors import ConfigurationError

BOT_TYPES: dict[str, type[BaseTradingBot]] = {"swing": SwingBot, "trend": TrendBot, "investor": InvestorBot}


def build_bots(cfg: AppConfig, only: list[str] | None = None) -> dict[str, BaseTradingBot]:
    if cfg.bots is None:
        return {}
    if only:
        unknown = sorted(set(only) - set(cfg.bots.bots))
        if unknown:
            raise ConfigurationError(f"unknown bot(s) {unknown}; configured: {sorted(cfg.bots.bots)}")
    out: dict[str, BaseTradingBot] = {}
    order = list(BOT_TYPES)
    # Deterministic order (swing, trend, investor): when several bots compete
    # for the same account-level capacity in one cycle, it is granted in this order.
    items = sorted(cfg.bots.bots.items(), key=lambda kv: (order.index(kv[1].bot_type), kv[0]))
    for bot_id, bcfg in items:
        if not bcfg.enabled or (only and bot_id not in only):
            continue
        cls = BOT_TYPES[bcfg.bot_type]
        out[bot_id] = cls(bot_id, bcfg, cfg.bot_risk_config(bot_id), cfg.settings.universe.symbols,
                          cfg.settings.universe.benchmark)
    return out
