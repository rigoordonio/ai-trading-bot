"""Strategy-level (per-bot) position attribution.

The broker sees ONE aggregate position per symbol. Internally every share
belongs to exactly one bot's sleeve:

    AAPL   SWING_V1 50 + TREND_V1 75 + INVESTOR_V1 100  = expected 225
           broker position                               = must be 225

A bot can only sell shares its own sleeve holds, so a SWING exit can never
liquidate the INVESTOR allocation. Reconciliation compares the expected
aggregate with the broker and reports the per-bot breakdown on mismatch.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Mapping

import pandas as pd

from src.portfolio.portfolio import Portfolio


def virtual_positions(sleeves: Mapping[str, Portfolio]) -> dict[str, dict[str, int]]:
    """symbol -> {bot_id: quantity} (only non-zero holdings)."""
    out: dict[str, dict[str, int]] = defaultdict(dict)
    for bot_id, p in sleeves.items():
        for sym, pos in p.positions.items():
            if pos.quantity:
                out[sym][bot_id] = pos.quantity
    return dict(out)


def aggregate_positions(sleeves: Mapping[str, Portfolio]) -> dict[str, int]:
    return {s: sum(by_bot.values()) for s, by_bot in virtual_positions(sleeves).items()}


def attribution_table(sleeves: Mapping[str, Portfolio], broker_positions: Mapping[str, int] | None = None
                      ) -> pd.DataFrame:
    """Rows = symbols; one column per bot, `expected` (sum) and, if given,
    `broker` and `difference`."""
    vp = virtual_positions(sleeves)
    syms = sorted(set(vp) | set(broker_positions or {}))
    bots = list(sleeves)
    rows = []
    for s in syms:
        row: dict = {"symbol": s, **{b: vp.get(s, {}).get(b, 0) for b in bots}}
        row["expected"] = sum(vp.get(s, {}).values())
        if broker_positions is not None:
            row["broker"] = int(broker_positions.get(s, 0))
            row["difference"] = row["broker"] - row["expected"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("symbol") if rows else pd.DataFrame()
