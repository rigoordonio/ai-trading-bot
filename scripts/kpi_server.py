"""Local KPI page for the trading bots.

    python scripts/kpi_server.py            # then open http://localhost:8765
    python scripts/kpi_server.py --port 9000

Read-only: it reads data/paper/* (written by the daily run and the Reviewer)
and, with BROKER=alpaca_paper, asks Alpaca for the live account value and
positions (GET requests only, at most once every 30 s). It cannot place,
change or cancel orders. Listens on 127.0.0.1 only: nothing outside this Mac
can reach it.
"""
import argparse
import json
import os
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import _bootstrap  # noqa: F401

from src.config import load_config, resolve_path
from src.dashboard.kpi_data import build_kpis, live_view
from src.dashboard.kpi_page import render_html


class LiveFeed:
    """Cached, read-only live snapshot from the Alpaca PAPER account."""

    def __init__(self, ttl: float = 30.0, broker=None):
        self.ttl, self._broker, self._lock = ttl, broker, threading.Lock()
        self._at, self._snap, self._err = 0.0, None, None

    def get(self) -> tuple[dict | None, str | None, str | None]:
        with self._lock:
            if time.time() - self._at >= self.ttl:
                self._at = time.time()
                try:
                    if self._broker is None:
                        if (os.getenv("BROKER") or "").strip().lower() != "alpaca_paper":
                            raise RuntimeError("BROKER is not alpaca_paper in .env")
                        from src.execution.broker import create_broker
                        self._broker = create_broker("PAPER", "alpaca_paper")
                    self._snap, self._err = self._broker.live_snapshot(), None
                except Exception as exc:  # noqa: BLE001 - shown on the page; never affects trading
                    self._snap, self._err = None, f"{type(exc).__name__}: {exc}"[:200]
            at = datetime.fromtimestamp(self._at).strftime("%H:%M:%S") if self._at else None
            return self._snap, self._err, at


def make_handler(state_dir, allocations, live: LiveFeed | None = None):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802 - http.server API
            try:
                k = build_kpis(state_dir, allocations)
                lv = None
                if live is not None:
                    snap, err, at = live.get()
                    lv = live_view(snap, k, err, at)
                if self.path.startswith("/api/kpis"):
                    self._send(200, json.dumps({**k, "live": lv}, default=str).encode(), "application/json")
                elif self.path in ("/", "/index.html"):
                    self._send(200, render_html(k, live=lv).encode(), "text/html; charset=utf-8")
                else:
                    self._send(404, b"not found", "text/plain")
            except Exception as exc:  # noqa: BLE001 - show the problem instead of a blank page
                self._send(500, f"KPI page error: {type(exc).__name__}: {exc}".encode(), "text/plain")

        def log_message(self, *args):  # keep the terminal quiet
            return

    return Handler


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--state-dir", default="data/paper")
    args = ap.parse_args()
    cfg = load_config()
    alloc = {b: x.capital_allocation for b, x in (cfg.bots.bots.items() if cfg.bots else []) if x.enabled}
    ap_live = LiveFeed()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(resolve_path(args.state_dir), alloc, ap_live))
    print(f"KPI page: http://localhost:{args.port}   (Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
