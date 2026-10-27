"""Weekly candles and weekly features mapped onto the daily calendar.

Weekly bars are built from validated, adjusted DAILY bars (one data source,
one validation path). A week's bar is labelled with the last trading date
that falls in that week (Mon-Fri).

AVAILABILITY RULE (no look-ahead): weekly features for week W become usable
on the first trading day AFTER W's last session. On day t a bot sees the
latest week whose label is strictly earlier than t. That rule needs no
knowledge of the future calendar and gives identical answers whether the
data ends mid-week or not (tested in tests/test_bots.py).

Columns are prefixed ``w_`` (e.g. ``w_sma_40`` = 40-week SMA,
``w_ret_26d`` = 26-WEEK return despite the inherited "d" suffix).
"""
from __future__ import annotations

from typing import Callable

import pandas as pd

from src.data.resampling import WEEKLY_AGG, to_weekly
from src.features.pipeline import FeatureConfig, FeaturePipeline

__all__ = ["WeeklyFeaturePipeline", "to_weekly", "AGG"]
AGG = WEEKLY_AGG


class WeeklyFeaturePipeline:
    prefix = "w_"

    def __init__(self, config: FeatureConfig | None = None, benchmark: str = "SPY",
                 post: Callable[[pd.DataFrame], pd.DataFrame] | None = None):
        """``post`` adds bot-specific features computed in WEEKLY space
        (rows = completed weeks), before the availability rule is applied."""
        self.inner = FeaturePipeline(config, benchmark)
        self.post = post

    @property
    def version(self) -> str:
        return "weekly:" + self.inner.version

    def weekly_panel(self, prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
        weekly = {s: to_weekly(df) for s, df in prices.items() if not df.empty}
        panel = self.inner.compute(weekly)
        return self.post(panel) if self.post is not None else panel

    def compute(self, prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
        wpanel = self.weekly_panel(prices).add_prefix(self.prefix)
        frames = []
        for sym, df in prices.items():
            if df.empty or sym not in wpanel.index.get_level_values("symbol"):
                continue
            w = wpanel.xs(sym, level="symbol").reset_index().rename(columns={"date": "week_end"})
            daily = pd.DataFrame({"date": df.index})
            m = pd.merge_asof(daily, w.sort_values("week_end"), left_on="date", right_on="week_end",
                              direction="backward", allow_exact_matches=False)   # strictly earlier weeks
            m["symbol"] = sym
            frames.append(m)
        out = pd.concat(frames).set_index(["date", "symbol"]).sort_index()
        return out
