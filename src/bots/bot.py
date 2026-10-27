"""Backward-compatible import path. The bot contract lives in
src/bots/base.py; concrete bots in src/bots/{swing,trend,investor}/."""
from src.bots.base import BaseTradingBot, BotVersions
from src.bots.registry import BOT_TYPES, build_bots

TradingBot = BaseTradingBot

__all__ = ["BaseTradingBot", "BotVersions", "TradingBot", "build_bots", "BOT_TYPES"]
