"""Tradable universe with optional point-in-time membership.

Survivorship bias: if you backtest today's S&P 500 constituents over 2010-2024
you only include companies that *survived and grew*. Results look better than
any investor could have achieved. The fix is point-in-time membership: a
symbol is tradable only between the dates it was actually in the universe,
and delisted names must be included with their (often bad) final prices.

The membership file is a CSV:  symbol,start_date,end_date[,sector]
(empty end_date = still a member). A provider with delisted history (e.g. a
paid vendor) is required for a truly survivorship-free study; yfinance does
not carry delisted tickers.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


class Universe:
    def __init__(self, symbols: list[str], benchmark: str = "SPY", sectors: dict[str, str] | None = None,
                 membership: pd.DataFrame | None = None):
        self.symbols = [s.upper() for s in symbols]
        self.benchmark = benchmark.upper()
        self.sectors = {k.upper(): v for k, v in (sectors or {}).items()}
        self.membership = membership

    @classmethod
    def from_config(cls, ucfg, root: Path | None = None) -> "Universe":
        membership = None
        symbols = list(ucfg.symbols)
        sectors = dict(ucfg.sectors)
        if ucfg.membership_file:
            path = Path(ucfg.membership_file)
            if root is not None and not path.is_absolute():
                path = root / path
            membership = pd.read_csv(path, parse_dates=["start_date", "end_date"])
            membership["symbol"] = membership["symbol"].str.upper()
            symbols = sorted(set(symbols) | set(membership["symbol"]))
            if "sector" in membership.columns:
                sectors.update(membership.dropna(subset=["sector"]).set_index("symbol")["sector"].to_dict())
        return cls(symbols, ucfg.benchmark, sectors, membership)

    @property
    def all_symbols(self) -> list[str]:
        """Symbols to download (universe + benchmark)."""
        return sorted(set(self.symbols) | {self.benchmark})

    def sector(self, symbol: str) -> str | None:
        return self.sectors.get(symbol.upper())

    def is_active(self, symbol: str, date: pd.Timestamp) -> bool:
        if self.membership is None:
            return symbol.upper() in self.symbols
        rows = self.membership[self.membership["symbol"] == symbol.upper()]
        if rows.empty:
            return False
        end = rows["end_date"].fillna(pd.Timestamp.max)
        return bool(((rows["start_date"] <= date) & (end >= date)).any())

    def active_mask(self, dates: pd.DatetimeIndex, symbols: list[str]) -> pd.DataFrame:
        """Boolean frame (dates x symbols): tradable on that date?"""
        if self.membership is None:
            return pd.DataFrame(True, index=dates, columns=symbols)
        mask = pd.DataFrame(False, index=dates, columns=symbols)
        for _, r in self.membership.iterrows():
            if r["symbol"] in mask.columns:
                end = r["end_date"] if pd.notna(r["end_date"]) else dates.max()
                mask.loc[(dates >= r["start_date"]) & (dates <= end), r["symbol"]] = True
        return mask
