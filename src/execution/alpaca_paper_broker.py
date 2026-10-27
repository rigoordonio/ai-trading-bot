"""Alpaca **paper** trading adapter (REST, no SDK dependency).

PAPER ONLY. The constructor refuses any base URL other than Alpaca's paper
endpoint, so a live key/URL can never be used by this class.

Behaviour
* our idempotency key is sent as Alpaca's ``client_order_id``: after a
  timeout the OrderManager looks the order up by that id instead of
  resubmitting (``get_order_by_client_id``)
* daily orders are plain MARKET ``day`` orders submitted while the market is
  CLOSED, so Alpaca queues them and fills them at the next open: this matches
  the backtest's "decide at close, fill at next open". (Opening-auction "opg"
  orders are still available via ALPACA_TIME_IN_FORCE=opg, but Alpaca's paper
  simulator often lets them expire unfilled.) During regular hours the adapter
  refuses locally: an order sent then would fill at an intraday price instead
* GET requests are retried (idempotent); order POSTs are NEVER retried
* HTTP 401 -> credentials problem (CRITICAL, halts trading)
  HTTP 403/422 on an order -> business rejection (e.g. buying power)
  duplicate client_order_id -> DuplicateOrderError
  timeouts / 5xx on POST -> raised; OrderManager marks UNKNOWN and reconciles
* API secrets are never logged and never appear in ``repr``
"""
from __future__ import annotations

import os
import time as time_module
from datetime import datetime, time
from typing import Any, Callable
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from src.errors import (
    BrokerTimeoutError,
    ConfigurationError,
    DuplicateOrderError,
    LiveTradingNotAllowedError,
    OrderRejectedError,
    TradingDisabledError,
)
from src.execution.broker import AccountInfo, BrokerAdapter, BrokerOrder, BrokerPosition
from src.execution.orders import Order, OrderType
from src.logging_setup import get_logger
from src.risk.engine import RiskEngine
from src.safety.api_errors import APIError, MalformedResponseError, parse_json_response, retry_call

log = get_logger(__name__)

PAPER_URL = "https://paper-api.alpaca.markets"
ET = ZoneInfo("America/New_York")

# Alpaca order status -> our broker-level status vocabulary
STATUS_MAP = {
    "new": "ACKNOWLEDGED", "accepted": "ACKNOWLEDGED", "pending_new": "ACKNOWLEDGED",
    "accepted_for_bidding": "ACKNOWLEDGED", "held": "ACKNOWLEDGED", "pending_cancel": "ACKNOWLEDGED",
    "pending_replace": "ACKNOWLEDGED", "calculated": "ACKNOWLEDGED",
    "partially_filled": "PARTIALLY_FILLED", "filled": "FILLED",
    "canceled": "CANCELLED", "expired": "CANCELLED", "done_for_day": "CANCELLED", "replaced": "CANCELLED",
    "stopped": "CANCELLED", "suspended": "CANCELLED",
    "rejected": "REJECTED",
}


def opg_window_open(now_utc: datetime) -> bool:
    """True when Alpaca accepts OPG orders: after 19:00 ET or before 09:28 ET."""
    t = now_utc.astimezone(ET).time()
    return t >= time(19, 0) or t < time(9, 28)


def day_window_open(now_utc: datetime) -> bool:
    """True when a market DAY order will queue for the NEXT open: outside
    09:28-16:00 ET on weekdays, any time at weekends. (Exchange holidays just
    queue the order to the following session.)"""
    et = now_utc.astimezone(ET)
    if et.weekday() >= 5:
        return True
    t = et.time()
    return t >= time(16, 0) or t < time(9, 28)


class AlpacaPaperBroker(BrokerAdapter):
    name = "alpaca_paper"
    is_paper = True

    def __init__(self, api_key: str | None = None, api_secret: str | None = None, base_url: str | None = None,
                 time_in_force: str | None = None, trading_enabled: Callable[[], bool] = lambda: True,
                 known_symbols: set[str] | None = None, session: Any = None, timeout: float = 15.0,
                 clock: Callable[[], datetime] | None = None, sleep: Callable[[float], None] | None = None):
        url = (base_url or os.getenv("ALPACA_BASE_URL") or PAPER_URL).rstrip("/")
        if url.endswith("/v2"):
            url = url[:-3]
        if url != PAPER_URL:
            raise LiveTradingNotAllowedError(f"AlpacaPaperBroker only talks to {PAPER_URL}, got {url!r}")
        key = api_key or os.getenv("ALPACA_API_KEY")
        secret = api_secret or os.getenv("ALPACA_API_SECRET")
        if not key or not secret:
            raise ConfigurationError("ALPACA_API_KEY / ALPACA_API_SECRET missing (put them in .env)")
        time_in_force = (time_in_force or os.getenv("ALPACA_TIME_IN_FORCE") or "day").strip().lower()
        if time_in_force not in ("opg", "day"):
            raise ConfigurationError("time_in_force must be 'opg' or 'day'")
        self.base_url = url
        self.time_in_force = time_in_force
        self._headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
        self._trading_enabled = trading_enabled
        self.known_symbols = known_symbols
        if session is None and os.getenv("TRADING_BOT_IGNORE_DOTENV") == "1":
            # Test suite: a test must never reach a real (even paper) account.
            raise ConfigurationError("AlpacaPaperBroker without an injected session is not allowed in tests")
        self.session = session or requests.Session()
        self.timeout = timeout
        self.clock = clock or (lambda: datetime.now(tz=ZoneInfo("UTC")))
        self._sleep = sleep

    def submission_window_open(self) -> tuple[bool, str]:
        now = self.clock()
        now_et = now.astimezone(ET).strftime("%a %H:%M ET")
        if self.time_in_force == "opg" and not opg_window_open(now):
            return False, (f"outside Alpaca's opening-auction order window (19:00-09:28 ET; now {now_et}). "
                           "Nothing was decided or sent; run again after 19:00 ET (08:15 Manila is safe).")
        if self.time_in_force == "day" and not day_window_open(now):
            return False, (f"the US market is open or about to open (now {now_et}); an order now would not fill at "
                           "the next open. Nothing was decided or sent; run after 16:00 ET (08:15 Manila is safe).")
        return True, ""

    def __repr__(self) -> str:   # never expose credentials
        return f"AlpacaPaperBroker(base_url={self.base_url!r}, tif={self.time_in_force!r})"

    # ----------------------------------------------------------------- http
    def _request(self, method: str, path: str, *, params: dict | None = None, json_body: dict | None = None,
                 allow_404: bool = False) -> Any:
        try:
            r = self.session.request(method, f"{self.base_url}{path}", headers=self._headers, params=params,
                                     json=json_body, timeout=self.timeout)
        except requests.Timeout as exc:
            raise BrokerTimeoutError(f"Alpaca {method} {path} timed out") from exc
        except requests.ConnectionError as exc:
            raise ConnectionError(f"Alpaca {method} {path}: connection failed") from exc
        if allow_404 and r.status_code == 404:
            return None
        if r.status_code == 204:
            return {}
        return parse_json_response(r.status_code, r.text)

    def _get(self, path: str, **kw) -> Any:
        return retry_call(lambda: self._request("GET", path, **kw), idempotent=True, what=f"GET {path}",
                          sleep=self._sleep or time_module.sleep)

    # --------------------------------------------------------------- reads
    def get_account(self) -> AccountInfo:
        a = self._get("/v2/account")
        try:
            return AccountInfo(str(a["account_number"]), float(a["cash"]), float(a["equity"]),
                               float(a["buying_power"]), a.get("currency", "USD"))
        except (KeyError, TypeError, ValueError) as exc:
            raise MalformedResponseError(f"unexpected account payload: {exc}") from exc

    def get_positions(self) -> dict[str, BrokerPosition]:
        rows = self._get("/v2/positions")
        if not isinstance(rows, list):
            raise MalformedResponseError("positions payload is not a list")
        out = {}
        for p in rows:
            qty = float(p["qty"])
            if not qty.is_integer():
                raise MalformedResponseError(f"fractional position in {p['symbol']} (not supported)")
            out[p["symbol"]] = BrokerPosition(p["symbol"], int(qty), float(p["avg_entry_price"]))
        return out

    def live_snapshot(self) -> dict:
        """READ-ONLY live view for the KPI page: account value now and every
        position at Alpaca's current price. Never used for trading decisions."""
        def f(x):
            try:
                return float(x)
            except (TypeError, ValueError):
                return None
        a = self._get("/v2/account")
        rows = self._get("/v2/positions")
        if not isinstance(a, dict) or not isinstance(rows, list):
            raise MalformedResponseError("unexpected live snapshot payload")
        eq, last = f(a.get("equity")), f(a.get("last_equity"))
        return {"account": {"equity": eq, "last_equity": last, "cash": f(a.get("cash")),
                            "buying_power": f(a.get("buying_power")),
                            "change_today": (eq - last) if eq is not None and last is not None else None},
                "positions": [{"symbol": p.get("symbol"), "qty": f(p.get("qty")),
                               "avg_entry_price": f(p.get("avg_entry_price")),
                               "current_price": f(p.get("current_price")), "market_value": f(p.get("market_value")),
                               "unrealized_pl": f(p.get("unrealized_pl")),
                               "unrealized_plpc": f(p.get("unrealized_plpc")),
                               "change_today": f(p.get("change_today")),
                               "unrealized_intraday_pl": f(p.get("unrealized_intraday_pl"))} for p in rows]}

    def get_market_data(self, symbol: str, start=None, end=None) -> pd.DataFrame:
        raise NotImplementedError("market data comes from the configured MarketDataProvider")

    @staticmethod
    def _to_broker_order(o: dict) -> BrokerOrder:
        filled = float(o.get("filled_qty") or 0)
        avg = o.get("filled_avg_price")
        return BrokerOrder(
            broker_order_id=o["id"], client_order_id=o.get("client_order_id", ""), symbol=o["symbol"],
            side=str(o["side"]).upper(), quantity=int(float(o.get("qty") or 0)),
            status=STATUS_MAP.get(o.get("status", ""), "ACKNOWLEDGED"), filled_quantity=int(filled),
            avg_fill_price=float(avg) if avg not in (None, "") else None,
            submitted_at=o.get("submitted_at") or "",
            reject_reason="rejected by Alpaca" if o.get("status") == "rejected" else None,
            filled_at=o.get("filled_at"))

    def get_orders(self, status: str | None = None) -> list[BrokerOrder]:
        rows = self._get("/v2/orders", params={"status": "all", "limit": 500, "direction": "desc"})
        if not isinstance(rows, list):
            raise MalformedResponseError("orders payload is not a list")
        out = [self._to_broker_order(o) for o in rows]
        return [o for o in out if status is None or o.status == status]

    def get_order_by_client_id(self, client_order_id: str) -> BrokerOrder | None:
        o = self._get("/v2/orders:by_client_order_id", params={"client_order_id": client_order_id}, allow_404=True)
        return self._to_broker_order(o) if o else None

    # -------------------------------------------------------------- writes
    def submit_order(self, order: Order) -> BrokerOrder:
        if not self._trading_enabled():
            raise TradingDisabledError("trading is disabled; order not sent to Alpaca")
        if not order.risk_decision_id or not RiskEngine.verify(order):
            raise OrderRejectedError(f"{order.symbol}: order lacks a valid risk approval")
        if self.known_symbols is not None and order.symbol not in self.known_symbols:
            raise OrderRejectedError(f"unknown symbol {order.symbol}")
        tif = self.time_in_force
        if tif == "opg" and not opg_window_open(self.clock()):
            raise OrderRejectedError("outside Alpaca's OPG submission window (19:00-09:28 ET); run the bot "
                                     "after 19:00 ET (08:15 Manila is safe all year)")
        if tif == "day" and not day_window_open(self.clock()):
            raise OrderRejectedError("market open or about to open (09:28-16:00 ET): order not sent; run the bot "
                                     "after the close (08:15 Manila is safe all year)")
        body: dict[str, Any] = {"symbol": order.symbol, "qty": str(order.quantity), "side": order.side.value.lower(),
                                "type": "market" if order.order_type == OrderType.MARKET else "limit",
                                "time_in_force": tif, "client_order_id": order.idempotency_key}
        if order.order_type == OrderType.LIMIT:
            body["limit_price"] = f"{order.limit_price:.2f}"
        try:
            o = self._request("POST", "/v2/orders", json_body=body)   # never retried
        except APIError as exc:
            msg = str(exc)
            if exc.status_code == 422 and "client_order_id" in msg.lower():
                raise DuplicateOrderError(f"Alpaca: duplicate client_order_id {order.idempotency_key}") from exc
            if exc.status_code in (403, 422):
                raise OrderRejectedError(f"Alpaca rejected {order.symbol}: {msg}") from exc
            raise
        if not isinstance(o, dict) or "id" not in o:
            raise MalformedResponseError("order response without id")
        return self._to_broker_order(o)

    def cancel_order(self, broker_order_id: str) -> BrokerOrder:
        self._request("DELETE", f"/v2/orders/{broker_order_id}", allow_404=True)
        o = self._get(f"/v2/orders/{broker_order_id}")
        return self._to_broker_order(o)
