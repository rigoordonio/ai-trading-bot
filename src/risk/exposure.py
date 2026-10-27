"""Exposure arithmetic (pure functions on weights)."""
from __future__ import annotations

from collections import defaultdict

NON_SECTOR = {None, "ETF", "Index", "Cash"}


def gross_exposure(weights: dict[str, float]) -> float:
    return sum(abs(w) for w in weights.values())


def net_exposure(weights: dict[str, float]) -> float:
    return sum(weights.values())


def sector_weights(weights: dict[str, float], sectors: dict[str, str]) -> dict[str, float]:
    out: dict[str, float] = defaultdict(float)
    for s, w in weights.items():
        sec = sectors.get(s)
        if sec is not None and sec not in NON_SECTOR:
            out[sec] += w
    return dict(out)


def largest_position(weights: dict[str, float]) -> tuple[str | None, float]:
    if not weights:
        return None, 0.0
    s = max(weights, key=lambda k: abs(weights[k]))
    return s, weights[s]
