"""Data-quality checks for daily OHLCV bars.

Severity levels:
    ERROR   -> the row is unusable and is removed by ``clean`` (nulls,
               non-positive prices, impossible OHLC, duplicates)
    WARNING -> suspicious but possibly real (huge move, volume spike, stale
               prices, gaps). Kept, but reported so a human can look.
    INFO    -> informational (e.g. a corporate action detected)

Nothing is ever forward-filled or invented: missing data stays missing.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.data.provider import PRICE_COLUMNS
from src.logging_setup import get_logger

log = get_logger(__name__)

ERROR, WARNING, INFO = "ERROR", "WARNING", "INFO"


@dataclass
class Issue:
    check: str
    severity: str
    count: int
    examples: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        ex = f" e.g. {', '.join(self.examples[:3])}" if self.examples else ""
        return f"[{self.severity}] {self.check}: {self.count}{ex}"


@dataclass
class ValidationReport:
    symbol: str
    rows: int
    issues: list[Issue] = field(default_factory=list)
    bad_dates: set = field(default_factory=set)  # rows ``clean`` will drop

    @property
    def ok(self) -> bool:
        return not any(i.severity == ERROR for i in self.issues)

    def add(self, check: str, severity: str, mask_or_dates, sample: int = 5) -> None:
        if isinstance(mask_or_dates, pd.Series):
            dates = list(mask_or_dates[mask_or_dates].index)
        else:
            dates = list(mask_or_dates)
        if not dates:
            return
        self.issues.append(Issue(check, severity, len(dates), [str(pd.Timestamp(d).date()) for d in dates[:sample]]))
        if severity == ERROR:
            self.bad_dates.update(dates)

    def summary(self) -> str:
        if not self.issues:
            return f"{self.symbol}: {self.rows} rows, no issues"
        return f"{self.symbol}: {self.rows} rows\n  " + "\n  ".join(str(i) for i in self.issues)

    def to_dict(self) -> dict:
        return {"symbol": self.symbol, "rows": self.rows, "ok": self.ok,
                "issues": [i.__dict__ for i in self.issues]}


class DataValidator:
    def __init__(self, max_daily_abs_return: float = 0.5, volume_zscore_threshold: float = 8.0,
                 stale_run_length: int = 5, max_missing_business_days: int = 3):
        self.max_ret = max_daily_abs_return
        self.vol_z = volume_zscore_threshold
        self.stale_n = stale_run_length
        self.max_gap = max_missing_business_days

    @classmethod
    def from_config(cls, cfg) -> "DataValidator":
        return cls(cfg.max_daily_abs_return, cfg.volume_zscore_threshold,
                   cfg.stale_run_length, cfg.max_missing_business_days)

    # ------------------------------------------------------------------ checks
    def validate(self, df: pd.DataFrame, symbol: str = "?") -> ValidationReport:
        rep = ValidationReport(symbol=symbol, rows=len(df))
        if df.empty:
            rep.issues.append(Issue("empty_dataset", ERROR, 1))
            return rep

        # Schema
        missing = [c for c in PRICE_COLUMNS if c not in df.columns]
        if missing:
            rep.issues.append(Issue("schema_missing_columns", ERROR, len(missing), missing))
            return rep
        if not isinstance(df.index, pd.DatetimeIndex):
            rep.issues.append(Issue("schema_index_not_datetime", ERROR, 1))
            return rep
        if not df.index.is_monotonic_increasing:
            rep.issues.append(Issue("unsorted_index", WARNING, 1))

        # Duplicates (keep the last occurrence; earlier copies are bad rows)
        dup = df.index.duplicated(keep="last")
        if dup.any():
            rep.issues.append(Issue("duplicate_timestamps", ERROR, int(dup.sum()),
                                    [str(d.date()) for d in df.index[dup][:5]]))

        d = df[~dup]
        prices = d[["open", "high", "low", "close", "adj_close"]]

        rep.add("null_values", ERROR, d[PRICE_COLUMNS].isna().any(axis=1))
        rep.add("non_positive_price", ERROR, (prices <= 0).any(axis=1))
        rep.add("negative_volume", ERROR, d["volume"] < 0)
        impossible = (
            (d["high"] < d[["open", "close", "low"]].max(axis=1) - 1e-9)
            | (d["low"] > d[["open", "close", "high"]].min(axis=1) + 1e-9)
        )
        rep.add("impossible_ohlc", ERROR, impossible)

        # Abnormal moves are judged on adj_close so splits are not false alarms.
        adj_ret = d["adj_close"].pct_change()
        raw_ret = d["close"].pct_change()
        rep.add("abnormal_price_move", WARNING, adj_ret.abs() > self.max_ret)
        # Raw price jumped but adjusted did not -> corporate action (split).
        corp = (raw_ret.abs() > 0.25) & (adj_ret.abs() < 0.15)
        rep.add("corporate_action_detected", INFO, corp)
        if "splits" in d.columns:
            rep.add("split_events", INFO, d["splits"].fillna(0) > 0)

        # Abnormal volume: z-score of log volume vs trailing 60 days (past only).
        logv = np.log1p(d["volume"].clip(lower=0))
        mu = logv.rolling(60, min_periods=20).mean().shift(1)
        sd = logv.rolling(60, min_periods=20).std().shift(1)
        z = (logv - mu) / sd
        rep.add("abnormal_volume", WARNING, z.abs() > self.vol_z)
        rep.add("zero_volume", WARNING, d["volume"] == 0)

        # Stale: close unchanged for N consecutive sessions.
        same = d["close"].diff().eq(0)
        run = same.groupby((~same).cumsum()).cumsum()
        rep.add("stale_prices", WARNING, run >= self.stale_n - 1)

        # Missing timestamps: business-day gaps longer than the threshold
        # (single-day gaps are usually exchange holidays).
        idx = d.index.sort_values()
        if len(idx) > 1:
            gaps = np.busday_count(idx[:-1].values.astype("datetime64[D]"), idx[1:].values.astype("datetime64[D]")) - 1
            gap_end = idx[1:][gaps > self.max_gap]
            rep.add("missing_timestamps", WARNING, list(gap_end))
        if len(d) and (d.index.dayofweek >= 5).any():
            rep.add("weekend_timestamps", WARNING, pd.Series(d.index.dayofweek >= 5, index=d.index))
        return rep

    # ------------------------------------------------------------------- clean
    @staticmethod
    def clean(df: pd.DataFrame, report: ValidationReport) -> pd.DataFrame:
        out = df[~df.index.duplicated(keep="last")].sort_index()
        if report.bad_dates:
            out = out.drop(index=[d for d in report.bad_dates if d in out.index], errors="ignore")
        return out
