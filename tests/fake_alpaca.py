"""In-memory fake of the Alpaca paper REST API (only what the adapter uses)."""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass

import requests


@dataclass
class Resp:
    status_code: int
    text: str


class FakeAlpaca:
    def __init__(self, cash: float = 100_000.0):
        self.cash = cash
        self.positions: dict[str, dict] = {}
        self.orders: dict[str, dict] = {}
        self.calls: list[tuple[str, str]] = []
        # failure injection: list of (method, path_prefix, action) consumed in order
        self.faults: list[tuple[str, str, str]] = []

    # ------------------------------------------------------------ transport
    def request(self, method, url, headers=None, params=None, json=None, timeout=None):  # noqa: A002
        assert headers and headers.get("APCA-API-KEY-ID") and headers.get("APCA-API-SECRET-KEY")
        assert url.startswith("https://paper-api.alpaca.markets/")
        path = url.split("alpaca.markets", 1)[1]
        self.calls.append((method, path))
        for i, (m, prefix, action) in enumerate(self.faults):
            if m == method and path.startswith(prefix):
                self.faults.pop(i)
                if action == "timeout_before":
                    raise requests.Timeout("read timed out")
                if action == "timeout_after":
                    self._dispatch(method, path, params, json)
                    raise requests.Timeout("read timed out")
                if action.startswith("http_"):
                    return Resp(int(action[5:]), '{"message": "injected"}')
                if action == "malformed":
                    return Resp(200, "{not json")
        return self._dispatch(method, path, params, json)

    def _dispatch(self, method, path, params, body):
        if method == "GET" and path == "/v2/account":
            eq = self.cash + sum(p["qty"] * p["price"] for p in self.positions.values())
            return self._ok({"account_number": "PA3ZSFPZJ", "cash": str(self.cash), "equity": str(eq),
                             "buying_power": str(self.cash * 4), "currency": "USD"})
        if method == "GET" and path == "/v2/positions":
            return self._ok([{"symbol": s, "qty": str(p["qty"]), "avg_entry_price": str(p["price"])}
                             for s, p in self.positions.items() if p["qty"]])
        if method == "GET" and path == "/v2/orders":
            return self._ok(list(self.orders.values()))
        if method == "GET" and path == "/v2/orders:by_client_order_id":
            for o in self.orders.values():
                if o["client_order_id"] == params["client_order_id"]:
                    return self._ok(o)
            return Resp(404, '{"message": "order not found"}')
        if method == "GET" and path.startswith("/v2/orders/"):
            return self._ok(self.orders[path.rsplit("/", 1)[1]])
        if method == "POST" and path == "/v2/orders":
            if any(o["client_order_id"] == body["client_order_id"] for o in self.orders.values()):
                return Resp(422, '{"message": "client_order_id must be unique"}')
            oid = str(uuid.uuid4())
            self.orders[oid] = {"id": oid, "client_order_id": body["client_order_id"], "symbol": body["symbol"],
                                "side": body["side"], "qty": body["qty"], "type": body["type"],
                                "time_in_force": body["time_in_force"], "status": "accepted", "filled_qty": "0",
                                "filled_avg_price": None, "submitted_at": "2026-06-01T23:00:00Z", "filled_at": None}
            return self._ok(self.orders[oid])
        if method == "DELETE" and path.startswith("/v2/orders/"):
            oid = path.rsplit("/", 1)[1]
            if oid not in self.orders:
                return Resp(404, "{}")
            self.orders[oid]["status"] = "canceled"
            return Resp(204, "")
        return Resp(404, '{"message": "no route"}')

    @staticmethod
    def _ok(obj):
        return Resp(200, json.dumps(obj))

    # ------------------------------------------------------------ simulation
    def fill(self, oid: str, qty: int, price: float, when: str = "2026-06-02T13:30:00Z"):
        o = self.orders[oid]
        prev = int(float(o["filled_qty"]))
        prev_avg = float(o["filled_avg_price"] or 0)
        new_total = prev + qty
        o["filled_avg_price"] = str((prev_avg * prev + price * qty) / new_total)
        o["filled_qty"] = str(new_total)
        o["status"] = "filled" if new_total == int(o["qty"]) else "partially_filled"
        o["filled_at"] = when
        sign = 1 if o["side"] == "buy" else -1
        self.cash -= sign * qty * price
        p = self.positions.setdefault(o["symbol"], {"qty": 0, "price": price})
        if sign > 0:
            p["price"] = (p["price"] * p["qty"] + price * qty) / (p["qty"] + qty)
        p["qty"] += sign * qty

    def fill_all(self, price: float = 100.0, fraction: float = 1.0):
        for oid, o in list(self.orders.items()):
            if o["status"] in ("accepted", "new", "partially_filled"):
                remaining = int(o["qty"]) - int(float(o["filled_qty"]))
                q = max(1, int(remaining * fraction)) if fraction < 1 else remaining
                self.fill(oid, min(q, remaining), price)
