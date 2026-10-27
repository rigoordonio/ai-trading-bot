"""EXTRACT -> TRANSFORM -> LOAD for market data.

    extract   : provider.get_history with retries + exponential backoff
    store raw : data/raw/<provider>/<SYMBOL>.csv   (exactly what the vendor sent)
    transform : validate -> clean -> corporate-action adjustment
    store     : data/processed/<SYMBOL>.csv        (what features are built on)
    load      : optional database upsert (see src/data/database.py)

Both raw and processed data are kept so any number can be traced back.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.data.corporate_actions import apply_adjustment
from src.data.provider import STANDARD_COLUMNS, MarketDataProvider
from src.data.validator import DataValidator, ValidationReport
from src.logging_setup import get_logger

log = get_logger(__name__)


class DataPipelineError(RuntimeError):
    pass


def with_retries(fn, retries: int = 3, backoff: float = 2.0, what: str = "call"):
    """Run ``fn`` with exponential backoff. Re-raises the last error."""
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - vendor libraries raise anything
            last = exc
            if attempt < retries:
                wait = backoff * (2 ** (attempt - 1))
                log.warning("%s failed (attempt %d/%d): %s -> retrying in %.1fs", what, attempt, retries, exc, wait)
                time.sleep(wait)
    raise DataPipelineError(f"{what} failed after {retries} attempts: {last}") from last


@dataclass
class ETLResult:
    data: dict[str, pd.DataFrame] = field(default_factory=dict)
    reports: dict[str, ValidationReport] = field(default_factory=dict)
    failed: dict[str, str] = field(default_factory=dict)


class MarketDataDownloader:
    def __init__(self, provider: MarketDataProvider, validator: DataValidator, raw_dir: str | Path,
                 processed_dir: str | Path, retries: int = 3, backoff: float = 2.0,
                 price_adjustment: str = "adjusted", fail_on_error: bool = False, save: bool = True):
        self.provider = provider
        self.validator = validator
        self.raw_dir = Path(raw_dir) / provider.name
        self.processed_dir = Path(processed_dir)
        self.retries = retries
        self.backoff = backoff
        self.price_adjustment = price_adjustment
        self.fail_on_error = fail_on_error
        self.save = save

    # ---------------------------------------------------------------- extract
    def extract(self, symbol: str, start, end=None) -> pd.DataFrame:
        def _call():
            df = self.provider.get_history(symbol, start, end)
            if df is None or df.empty:
                raise DataPipelineError(f"no data returned for {symbol}")
            return df

        df = with_retries(_call, self.retries, self.backoff, what=f"download {symbol}")
        missing = [c for c in STANDARD_COLUMNS if c not in df.columns]
        if missing:  # schema validation at the boundary
            raise DataPipelineError(f"{symbol}: provider schema missing {missing}")
        if self.save:
            self.raw_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(self.raw_dir / f"{symbol}.csv")
        return df

    # -------------------------------------------------------------- transform
    def transform(self, symbol: str, raw: pd.DataFrame) -> tuple[pd.DataFrame, ValidationReport]:
        report = self.validator.validate(raw, symbol)
        for issue in report.issues:
            level = {"ERROR": log.error, "WARNING": log.warning}.get(issue.severity, log.info)
            level("%s %s", symbol, issue)
        if not report.ok and self.fail_on_error:
            raise DataPipelineError(report.summary())
        clean = self.validator.clean(raw, report)
        adjusted = apply_adjustment(clean, self.price_adjustment, self.provider.volume_split_adjusted)
        if self.save:
            self.processed_dir.mkdir(parents=True, exist_ok=True)
            adjusted.to_csv(self.processed_dir / f"{symbol}.csv")
        return adjusted, report

    # -------------------------------------------------------------------- run
    def run(self, symbols: list[str], start, end=None) -> ETLResult:
        result = ETLResult()
        for sym in symbols:
            try:
                raw = self.extract(sym, start, end)
                processed, report = self.transform(sym, raw)
                result.data[sym] = processed
                result.reports[sym] = report
                log.info("ETL %s: %d bars %s..%s (%d issues)", sym, len(processed),
                         processed.index.min().date(), processed.index.max().date(), len(report.issues))
            except Exception as exc:  # noqa: BLE001 - one bad symbol must not kill the batch
                log.error("ETL failed for %s: %s", sym, exc)
                result.failed[sym] = str(exc)
        return result


def load_processed(processed_dir: str | Path, symbols: list[str]) -> dict[str, pd.DataFrame]:
    out = {}
    for sym in symbols:
        path = Path(processed_dir) / f"{sym}.csv"
        if path.exists():
            out[sym] = pd.read_csv(path, parse_dates=["date"], index_col="date")
    return out


def build_downloader(cfg, provider: MarketDataProvider, save: bool = True) -> MarketDataDownloader:
    from src.config import resolve_path

    d = cfg.settings.data
    return MarketDataDownloader(
        provider=provider,
        validator=DataValidator.from_config(d.validation),
        raw_dir=resolve_path(d.raw_dir),
        processed_dir=resolve_path(d.processed_dir),
        retries=d.retries,
        backoff=d.retry_backoff_seconds,
        price_adjustment=d.price_adjustment,
        fail_on_error=d.validation.fail_on_error,
        save=save,
    )
