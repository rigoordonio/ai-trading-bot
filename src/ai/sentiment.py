"""News/headline sentiment as an OPTIONAL, research-only feature.

Deterministic lexicon scorer (offline). An LLM may be plugged in later to
categorise headlines, but its output is only ever a number in [-1, 1] that
would have to be backtested as a feature like any other. It is NOT used by
any bot by default and cannot create orders.

Look-ahead warning: news must be timestamped and only used AFTER its
publication time (a headline published at 16:30 belongs to the next bar).
"""
from __future__ import annotations

import re

POSITIVE = {"beat", "beats", "record", "upgrade", "upgraded", "growth", "surge", "strong", "raises", "raised",
            "outperform", "profit", "approval", "buyback", "expands"}
NEGATIVE = {"miss", "misses", "downgrade", "downgraded", "lawsuit", "probe", "recall", "weak", "cuts", "cut",
            "loss", "plunge", "fraud", "investigation", "warning", "layoffs", "delay"}


def score_headline(text: str) -> float:
    words = re.findall(r"[a-z]+", text.lower())
    pos = sum(w in POSITIVE for w in words)
    neg = sum(w in NEGATIVE for w in words)
    if pos + neg == 0:
        return 0.0
    return (pos - neg) / (pos + neg)


def categorise(text: str) -> str:
    s = score_headline(text)
    return "positive" if s > 0.2 else "negative" if s < -0.2 else "neutral"
