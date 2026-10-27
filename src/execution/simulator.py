"""Execution simulator: turns an approved order + the NEXT bar into a fill.

Cost model (all adverse to the trader):
    reference      = next bar OPEN (never the signal bar's close)
    half spread    = half_spread_bps
    slippage       = slippage_bps
    market impact  = impact_coefficient * sqrt(participation %)  [bps]
    commission     = max(commission_min, per_share * qty) + commission_pct * notional

Rejections / partial fills:
    - no bar or zero volume on the execution day  -> REJECTED
    - quantity above max_participation * volume   -> capped (PARTIALLY_FILLED)
    - buy costs more than available cash          -> shrunk to what cash allows,
                                                     REJECTED if that is 0
    - sell more than held (no shorting)           -> capped / REJECTED
    - limit order not marketable                  -> REJECTED (day order expired)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from src.execution.orders import Fill, Order, OrderStatus, OrderType, Side


@dataclass
class CostModel:
    commission_per_share: float = 0.005
    commission_min: float = 1.0
    commission_pct: float = 0.0
    half_spread_bps: float = 2.0
    slippage_bps: float = 5.0
    impact_coefficient: float = 10.0
    max_participation: float = 0.05

    @classmethod
    def from_config(cls, c) -> "CostModel":
        return cls(**c.model_dump()) if hasattr(c, "model_dump") else cls(**c)

    def commission(self, qty: int, price: float) -> float:
        if qty <= 0:
            return 0.0
        return max(self.commission_min, self.commission_per_share * qty) + self.commission_pct * qty * price

    def impact_bps(self, qty: int, bar_volume: float) -> float:
        if bar_volume <= 0:
            return 0.0
        return self.impact_coefficient * math.sqrt(100.0 * qty / bar_volume)

    def execution_price(self, side: Side, ref: float, qty: int, bar_volume: float) -> tuple[float, dict]:
        bps = {"spread": self.half_spread_bps, "slippage": self.slippage_bps,
               "impact": self.impact_bps(qty, bar_volume)}
        sign = 1 if side == Side.BUY else -1
        price = ref * (1 + sign * sum(bps.values()) / 10_000)
        costs = {k: ref * v / 10_000 * qty for k, v in bps.items()}
        return price, costs

    def scaled(self, factor: float) -> "CostModel":
        """Copy with all variable costs multiplied (for sensitivity analysis)."""
        return CostModel(self.commission_per_share * factor, self.commission_min * factor,
                         self.commission_pct * factor, self.half_spread_bps * factor, self.slippage_bps * factor,
                         self.impact_coefficient * factor, self.max_participation)


class ExecutionSimulator:
    def __init__(self, costs: CostModel | None = None):
        self.costs = costs or CostModel()

    def execute(self, order: Order, bar: pd.Series | None, cash: float, held_qty: int,
                timestamp: pd.Timestamp) -> Fill | None:
        """Fill an order that is SUBMITTED/ACKNOWLEDGED. Mutates its status and
        returns a Fill (or None if rejected)."""
        if order.status not in (OrderStatus.SUBMITTED, OrderStatus.ACKNOWLEDGED):
            raise ValueError(f"cannot execute order in state {order.status.value}")
        if bar is None or pd.isna(bar.get("open")) or bar.get("volume", 0) <= 0:
            return self._reject(order, "no tradable bar (missing data, halt or zero volume)")
        ref = float(bar["open"])
        qty = order.quantity

        if order.order_type == OrderType.LIMIT:
            if order.limit_price is None:
                return self._reject(order, "limit order without limit price")
            lim = float(order.limit_price)
            if order.side == Side.BUY:
                if ref <= lim:
                    pass
                elif bar["low"] <= lim:
                    ref = lim
                else:
                    return self._reject(order, f"limit {lim:.2f} not reached")
            else:
                if ref >= lim:
                    pass
                elif bar["high"] >= lim:
                    ref = lim
                else:
                    return self._reject(order, f"limit {lim:.2f} not reached")

        cap = int(self.costs.max_participation * float(bar["volume"]))
        reasons = []
        if qty > cap:
            qty = cap
            reasons.append(f"capped at {self.costs.max_participation:.0%} of volume")
        if order.side == Side.SELL:
            if held_qty <= 0:
                return self._reject(order, "no position to sell (shorting disabled)")
            if qty > held_qty:
                qty = held_qty
                reasons.append("capped at held quantity")
        if qty <= 0:
            return self._reject(order, "quantity rounds to zero after liquidity cap")

        price, costs = self.costs.execution_price(order.side, ref, qty, float(bar["volume"]))
        if order.side == Side.BUY:
            # The open can gap above the decision-time close: shrink to what cash allows.
            requested = qty
            for _ in range(5):
                if qty <= 0 or qty * price + self.costs.commission(qty, price) <= cash + 1e-9:
                    break
                qty = max(0, int((cash - self.costs.commission(qty, price)) // price))
                price, costs = self.costs.execution_price(order.side, ref, qty, float(bar["volume"]))
            if qty <= 0 or qty * price + self.costs.commission(qty, price) > cash + 1e-9:
                return self._reject(order, "insufficient cash")
            if qty < requested:
                reasons.append("reduced to available cash")

        order.filled_quantity = qty
        order.avg_fill_price = price
        order.transition(OrderStatus.FILLED if qty == order.quantity else OrderStatus.PARTIALLY_FILLED,
                         "simulated fill")
        if reasons:
            order.reject_reason = "; ".join(reasons)
        return Fill(order_id=order.order_id, symbol=order.symbol, side=order.side, quantity=qty, price=price,
                    reference_price=ref, commission=self.costs.commission(qty, price), timestamp=timestamp,
                    spread_cost=costs["spread"], slippage_cost=costs["slippage"], impact_cost=costs["impact"])

    @staticmethod
    def _reject(order: Order, reason: str) -> None:
        order.transition(OrderStatus.REJECTED, reason)
        order.reject_reason = reason
        return None
