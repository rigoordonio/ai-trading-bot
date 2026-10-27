"""Render the KPI page (one self-contained HTML document, no external assets)."""
from __future__ import annotations

import html
import json
import math

BOT_COLORS = {"SWING_V1": "--series-1", "TREND_V1": "--series-2", "INVESTOR_V1": "--series-3"}
BOT_LABEL = {"SWING_V1": "Swing", "TREND_V1": "Trend", "INVESTOR_V1": "Investor"}
BOT_ROLE = {"SWING_V1": "daily · 5-20 days", "TREND_V1": "daily · 1-3 months", "INVESTOR_V1": "weekly · 3-12 months"}


def _e(x) -> str:
    return html.escape("" if x is None else str(x))


def money(x, sign=False) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    s = f"{abs(x):,.0f}"
    return (("+" if x > 0 else "−" if x < 0 else "") if sign else ("−" if x < 0 else "")) + "$" + s


def pct(x, sign=True, nd=2) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    return (f"{x:+.{nd}%}" if sign else f"{x:.{nd}%}").replace("-", "−")


def price(x) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    return f"${x:,.2f}"


def num(x, nd=2) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    return f"{x:.{nd}f}".replace("-", "−")


def _delta_cls(x) -> str:
    if x is None or not isinstance(x, (int, float)) or x == 0:
        return ""
    return "up" if x > 0 else "down"


STATUS_UI = {"ACTIVE": ("good", "●", "Active"), "PROBATION": ("warning", "▲", "Probation"),
             "PAUSED": ("critical", "■", "Paused"), "UNKNOWN": ("critical", "■", "Unknown")}


def _status_pill(status: str, mult) -> str:
    cls, icon, label = STATUS_UI.get(status, ("warning", "▲", status.title()))
    extra = "" if status == "ACTIVE" else f" · size ×{mult:g}" if isinstance(mult, (int, float)) else ""
    return f'<span class="pill {cls}"><span aria-hidden="true">{icon}</span> {label}{extra}</span>'


def _tile(label, value, sub="", cls="") -> str:
    return (f'<div class="tile"><div class="tile-label">{_e(label)}</div>'
            f'<div class="tile-value {cls}">{value}</div><div class="tile-sub">{sub}</div></div>')


def _live_section(live: dict | None) -> str:
    if not live:
        return ""
    if not live.get("available"):
        why = live.get("error") or "live prices need BROKER=alpaca_paper and Alpaca keys in .env"
        return (f'<h2>Live from Alpaca</h2><p class="empty">Live data unavailable: {_e(why)}. '
                "Everything below still shows the bots&#39; last recorded run.</p>")
    acc = live.get("account") or {}
    eq, last = acc.get("equity"), acc.get("last_equity")
    upl = sum((p.get("unrealized_pl") or 0) for p in live.get("positions", []))
    tiles = "".join([
        _tile("Account value now", money(eq), f"Alpaca paper · {_e(live.get('at'))}"),
        _tile("Change today", money(acc.get("change_today"), True),
              pct((eq / last - 1) if eq and last else None) + " vs previous close", _delta_cls(acc.get("change_today"))),
        _tile("Unrealised P&L", money(upl, True), "open positions at current price", _delta_cls(upl)),
        _tile("Cash", money(acc.get("cash")), f"buying power {money(acc.get('buying_power'))}"),
    ])
    rows = ""
    for p in live.get("positions", []):
        rows += (f'<tr><td>{_e(p.get("symbol"))}</td><td>{_e(p.get("owner"))}</td>'
                 f'<td class="r">{num(p.get("qty"), 0)}</td><td class="r">{price(p.get("avg_entry_price"))}</td>'
                 f'<td class="r">{price(p.get("current_price"))}</td>'
                 f'<td class="r {_delta_cls(p.get("change_today"))}">{pct(p.get("change_today"))}</td>'
                 f'<td class="r {_delta_cls(p.get("unrealized_intraday_pl"))}">'
                 f'{money(p.get("unrealized_intraday_pl"), True)}</td>'
                 f'<td class="r {_delta_cls(p.get("unrealized_pl"))}">{money(p.get("unrealized_pl"), True)} '
                 f'({pct(p.get("unrealized_plpc"))})</td></tr>')
    table = (f'<div class="tablewrap"><table><thead><tr><th>Symbol</th><th>Owner (bot qty)</th><th class="r">Qty</th>'
             f'<th class="r">Avg entry</th><th class="r">Price now</th><th class="r">Today</th>'
             f'<th class="r">P&amp;L today</th><th class="r">Unrealised</th></tr></thead><tbody>{rows}</tbody></table>'
             f'</div>') if rows else '<p class="empty">No positions at Alpaca.</p>'
    return (f'<h2>Live from Alpaca <span class="live">● live</span></h2><div class="tiles live-tiles">{tiles}</div>'
            f'<div style="margin-top:12px">{table}</div>')


def render_html(k: dict, refresh_seconds: int = 60, live: dict | None = None) -> str:
    a = k.get("account") or {}
    bots = k.get("bots") or []
    trading = k.get("trading_enabled")
    rec_ok = k.get("reconciliation_ok")
    alerts = k.get("alerts") or []
    n_alert = len([x for x in alerts if x.get("level") == "ALERT"])
    n_warn = len([x for x in alerts if x.get("level") == "WARN"])

    pills = [
        ('good', '●', 'Trading on') if trading else ('critical', '■', 'Trading halted') if trading is False
        else ('warning', '▲', 'No run yet'),
        ('good', '●', 'Books match Alpaca') if rec_ok else ('critical', '■', 'Books mismatch') if rec_ok is False
        else ('warning', '▲', 'Not reconciled'),
        ('critical', '■', f'{n_alert} alert(s)') if n_alert else ('warning', '▲', f'{n_warn} warning(s)')
        if n_warn else ('good', '●', 'No alerts'),
    ]
    pill_html = "".join(f'<span class="pill {c}"><span aria-hidden="true">{i}</span> {_e(t)}</span>'
                        for c, i, t in pills)

    tiles = "".join([
        _tile("Account equity", money(a.get("equity")), f"started {money(a.get('start'))}"),
        _tile("Total P&L", money(a.get("total_pnl"), True), pct(a.get("total_return")),
              _delta_cls(a.get("total_pnl"))),
        _tile("Today", money(a.get("daily_pnl"), True), "vs previous run", _delta_cls(a.get("daily_pnl"))),
        _tile("Drawdown", pct(a.get("drawdown")), f"worst {pct(a.get('max_drawdown'))}",
              "down" if (a.get("drawdown") or 0) < 0 else ""),
        _tile("vs SPY", "—" if a.get("benchmark_return") is None or a.get("total_return") is None else
              f'{(a["total_return"] - a["benchmark_return"]) * 100:+.2f} pts'.replace("-", "−"),
              f"SPY {pct(a.get('benchmark_return'))} over the same period",
              _delta_cls(None if a.get("benchmark_return") is None or a.get("total_return") is None
                         else a["total_return"] - a["benchmark_return"])),
        _tile("Invested", pct(a.get("exposure"), sign=False, nd=0),
              f"{money(a.get('positions_value'))} · {a.get('open_orders', 0)} open order(s)"),
    ])

    cards = []
    for b in bots:
        bid = b["bot_id"]
        var = BOT_COLORS.get(bid, "--text-secondary")
        rows = [("Equity", money(b.get("equity")), ""),
                ("Total P&L", f'{money(b.get("total_pnl"), True)} ({pct(b.get("total_return"))})',
                 _delta_cls(b.get("total_pnl"))),
                ("Today", money(b.get("daily_pnl"), True), _delta_cls(b.get("daily_pnl"))),
                ("Drawdown", pct(b.get("drawdown")), "down" if (b.get("drawdown") or 0) < 0 else ""),
                ("Sharpe", num(b.get("sharpe")), ""),
                ("Win rate", pct(b.get("win_rate"), sign=False, nd=0), ""),
                ("Trades closed", _e(b.get("trades")), ""),
                ("Avg hold", "—" if b.get("avg_holding_days") is None else f'{b["avg_holding_days"]:.0f} days', ""),
                ("Invested", pct(b.get("exposure"), sign=False, nd=0), ""),
                ("Positions / open orders", f'{b.get("positions", 0)} / {b.get("open_orders", 0)}', ""),
                ("Capital share", pct(b.get("allocation"), sign=False, nd=0), "")]
        kv = "".join(f'<div class="kv"><span>{_e(n)}</span><span class="{c}">{v}</span></div>' for n, v, c in rows)
        cards.append(
            f'<section class="card bot"><header><span class="swatch" style="background:var({var})"></span>'
            f'<div><h3>{_e(BOT_LABEL.get(bid, bid))} <small>{_e(bid)}</small></h3>'
            f'<div class="muted">{_e(BOT_ROLE.get(bid, ""))}</div></div></header>'
            f'<div class="badges">{_status_pill(b.get("status", "ACTIVE"), b.get("size_multiplier"))}'
            f'<span class="grade">Grade <b>{_e(b.get("grade"))}</b></span></div>'
            f'{kv}<p class="review">{_e(b.get("review"))}</p></section>')

    def table(rows: list[dict], cols: list[tuple[str, str, str]], empty: str) -> str:
        if not rows:
            return f'<p class="empty">{_e(empty)}</p>'
        head = "".join(f'<th class="{al}">{_e(t)}</th>' for _, t, al in cols)
        body = ""
        for r in rows:
            cells = ""
            for key, _, al in cols:
                v = r.get(key)
                if key in ("market_value", "avg_cost", "last_price", "avg_fill_price"):
                    v = "—" if v is None else f"${v:,.2f}"
                    cells += f'<td class="{al}">{v}</td>'
                elif key == "unrealized_pnl":
                    cells += f'<td class="{al} {_delta_cls(v)}">{money(v, True)}</td>'
                elif key in ("score", "ml_probability"):
                    cells += f'<td class="{al}">{num(v)}</td>'
                elif key in ("created_at", "entry_date"):
                    cells += f'<td class="{al}">{_e(str(v)[:10] if v else "—")}</td>'
                elif key == "bot_id":
                    var = BOT_COLORS.get(v, "--text-secondary")
                    cells += (f'<td><span class="dot" style="background:var({var})"></span>'
                              f'{_e(BOT_LABEL.get(v, v))}</td>')
                else:
                    cells += f'<td class="{al}">{_e(v if v is not None else "—")}</td>'
            body += f"<tr>{cells}</tr>"
        return f'<div class="tablewrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'

    positions = table(k.get("positions") or [], [("bot_id", "Bot", ""), ("symbol", "Symbol", ""),
                                                 ("quantity", "Qty", "r"), ("avg_cost", "Avg cost", "r"),
                                                 ("last_price", "Last", "r"), ("market_value", "Value", "r"),
                                                 ("unrealized_pnl", "Unrealised", "r"), ("entry_date", "Since", "")],
                      "No positions yet. Fills are recorded at the next morning run.")
    decisions = table(k.get("decisions") or [], [("bot_id", "Bot", ""), ("symbol", "Symbol", ""),
                                                 ("action", "Signal", ""), ("score", "Score", "r"),
                                                 ("ml_probability", "Model", "r"), ("risk_status", "Risk", ""),
                                                 ("final_action", "Final", ""), ("quantity", "Qty", "r"),
                                                 ("reason", "Why", "")],
                      "No trade decisions in the latest run log yet.")
    orders = table(k.get("orders") or [], [("bot_id", "Bot", ""), ("symbol", "Symbol", ""), ("side", "Side", ""),
                                           ("quantity", "Qty", "r"), ("status", "Status", ""),
                                           ("filled_quantity", "Filled", "r"), ("avg_fill_price", "Fill price", "r"),
                                           ("created_at", "Decided on", "")], "No orders yet.")
    alert_html = "".join(
        f'<li class="{ {"ALERT": "critical", "WARN": "warning"}.get(x.get("level"), "info") }">'
        f'<b>{_e(x.get("level"))}</b> {_e(x.get("detail"))}</li>' for x in alerts) or '<li class="good">None</li>'

    chart_data = json.dumps(k.get("history") or {"dates": [], "series": {}})
    blocking = "; ".join(k.get("blocking_reasons") or [])
    return TEMPLATE.format(
        refresh=refresh_seconds, as_of=_e(k.get("as_of") or "no runs yet"),
        reviewed=_e((k.get("reviewed_at") or "")[:16].replace("T", " ")), pills=pill_html,
        blocking=f'<p class="blocking">{_e(blocking)}</p>' if blocking else "", tiles=tiles,
        cards="".join(cards), positions=positions, decisions=decisions, orders=orders, alerts=alert_html,
        chart_data=chart_data.replace("</", "<\\/"), live=_live_section(live))


TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="{refresh}">
<title>Trading Bots KPI</title>
<style>
:root {{
  color-scheme: light;
  --page: #f9f9f7; --surface-1: #fcfcfb; --text-primary: #0b0b0b; --text-secondary: #52514e;
  --muted: #898781; --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10);
  --series-1: #2a78d6; --series-2: #eb6834; --series-3: #1baf7a;
  --up: #006300; --down: #d03b3b;
  --good: #0ca30c; --warning: #fab219; --critical: #d03b3b;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    color-scheme: dark;
    --page: #0d0d0d; --surface-1: #1a1a19; --text-primary: #f0efec; --text-secondary: #c3c2b7;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
    --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70; --up: #0ca30c; --down: #e66767;
  }}
}}
:root[data-theme="dark"] {{
  color-scheme: dark;
  --page: #0d0d0d; --surface-1: #1a1a19; --text-primary: #f0efec; --text-secondary: #c3c2b7;
  --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
  --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70; --up: #0ca30c; --down: #e66767;
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--page); color: var(--text-primary);
  font: 14px/1.45 -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", Roboto, sans-serif; }}
main {{ max-width: 1240px; margin: 0 auto; padding: 24px 16px 48px; }}
h1 {{ font-size: 22px; margin: 0; letter-spacing: -0.01em; }}
h2 {{ font-size: 15px; margin: 32px 0 12px; color: var(--text-secondary); font-weight: 600; }}
h3 {{ margin: 0; font-size: 16px; }} h3 small {{ color: var(--muted); font-weight: 500; font-size: 12px; }}
.muted {{ color: var(--muted); font-size: 12px; }}
.top {{ display: flex; flex-wrap: wrap; gap: 12px; align-items: baseline; justify-content: space-between; }}
.pills {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }}
.pill {{ display: inline-flex; gap: 6px; align-items: center; border: 1px solid var(--border); background: var(--surface-1);
  border-radius: 999px; padding: 3px 10px; font-size: 12px; color: var(--text-primary); white-space: nowrap; }}
.pill.good span {{ color: var(--good); }} .pill.warning span {{ color: var(--warning); }}
.pill.critical span {{ color: var(--critical); }}
.blocking {{ color: var(--text-secondary); font-size: 12px; margin: 8px 0 0; }}
.tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin-top: 20px; }}
.tile, .card {{ background: var(--surface-1); border: 1px solid var(--border); border-radius: 12px; padding: 14px 16px; }}
.tile-label {{ color: var(--text-secondary); font-size: 12px; }}
.tile-value {{ font-size: 24px; font-weight: 650; margin-top: 4px; font-variant-numeric: tabular-nums; }}
.tile-sub {{ color: var(--muted); font-size: 12px; margin-top: 2px; font-variant-numeric: tabular-nums; }}
.up {{ color: var(--up); }} .down {{ color: var(--down); }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 12px; }}
.card header {{ display: flex; gap: 10px; align-items: center; }}
.swatch {{ width: 10px; height: 34px; border-radius: 4px; flex: none; }}
.badges {{ display: flex; gap: 10px; align-items: center; margin: 12px 0 8px; flex-wrap: wrap; }}
.grade {{ font-size: 12px; color: var(--text-secondary); }}
.kv {{ display: flex; justify-content: space-between; gap: 12px; padding: 5px 0; border-top: 1px solid var(--grid);
  font-variant-numeric: tabular-nums; }}
.kv span:first-child {{ color: var(--text-secondary); }}
.review {{ color: var(--muted); font-size: 12px; margin: 8px 0 0; }}
.chartcard {{ padding: 16px; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 16px; font-size: 12px; color: var(--text-secondary); margin-bottom: 8px; }}
.legend i {{ display: inline-block; width: 16px; height: 2px; vertical-align: middle; margin-right: 6px; }}
.legend i.dash {{ background: repeating-linear-gradient(90deg, var(--muted) 0 4px, transparent 4px 7px) !important; }}
#chart {{ position: relative; }} #chart svg {{ display: block; width: 100%; height: 280px; }}
.tip {{ position: absolute; pointer-events: none; background: var(--surface-1); border: 1px solid var(--border);
  border-radius: 8px; padding: 8px 10px; font-size: 12px; box-shadow: 0 4px 16px rgba(0,0,0,.12); display: none;
  font-variant-numeric: tabular-nums; min-width: 150px; }}
.tip .row {{ display: flex; justify-content: space-between; gap: 12px; }}
.empty {{ color: var(--muted); margin: 8px 0; }}
.tablewrap {{ overflow-x: auto; background: var(--surface-1); border: 1px solid var(--border); border-radius: 12px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; font-variant-numeric: tabular-nums; }}
th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid var(--grid); white-space: nowrap; }}
td:last-child {{ white-space: normal; }}
th {{ color: var(--text-secondary); font-weight: 600; font-size: 12px; }}
.r {{ text-align: right; }} tr:last-child td {{ border-bottom: none; }}
.dot {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; }}
ul.alerts {{ list-style: none; padding: 0; margin: 0; }}
ul.alerts li {{ background: var(--surface-1); border: 1px solid var(--border); border-left-width: 4px;
  border-radius: 8px; padding: 8px 12px; margin-bottom: 6px; }}
ul.alerts li.critical {{ border-left-color: var(--critical); }} ul.alerts li.warning {{ border-left-color: var(--warning); }}
ul.alerts li.good {{ border-left-color: var(--good); }} ul.alerts li.info {{ border-left-color: var(--axis); }}
.live {{ color: var(--good); font-size: 12px; font-weight: 500; margin-left: 6px; }}
.live-tiles {{ margin-top: 0; }}
footer {{ color: var(--muted); font-size: 12px; margin-top: 32px; }}
</style></head>
<body><main>
<div class="top"><div><h1>Trading bots KPI</h1>
<div class="muted">Paper account · decisions on US close of {as_of} · reviewed {reviewed} · page refreshes every minute</div></div></div>
<div class="pills">{pills}</div>{blocking}
{live}
<h2>Bots' books (last daily run)</h2>
<div class="tiles">{tiles}</div>
<h2>Bots</h2><div class="cards">{cards}</div>
<h2>Growth of 100 (each bot on its own capital, vs SPY)</h2>
<div class="card chartcard"><div class="legend" id="legend"></div><div id="chart"></div></div>
<h2>Alerts from the Reviewer</h2><ul class="alerts">{alerts}</ul>
<h2>Open positions (who owns what)</h2>{positions}
<h2>Latest decisions (signal, risk check, outcome)</h2>{decisions}
<h2>Recent orders</h2>{orders}
<footer>Read-only page: nothing here can place or change an order. Paper trading only.</footer>
</main>
<script id="data" type="application/json">{chart_data}</script>
<script>
(function () {{
  const data = JSON.parse(document.getElementById('data').textContent);
  const S = [["SWING_V1","Swing","var(--series-1)",""],["TREND_V1","Trend","var(--series-2)",""],
             ["INVESTOR_V1","Investor","var(--series-3)",""],["ACCOUNT","Whole account","var(--text-primary)",""],
             ["SPY","SPY (buy & hold)","var(--muted)","4 3"]].filter(s => data.series[s[0]]);
  const el = document.getElementById('chart'), lg = document.getElementById('legend');
  lg.innerHTML = S.map(s => '<span><i class="' + (s[3] ? 'dash' : '') + '" style="background:' + s[2] + '"></i>'
    + s[1] + '</span>').join('');
  const n = data.dates.length;
  if (n < 2) {{ el.innerHTML = '<p class="empty">The chart appears after the second daily run.</p>'; return; }}
  const W = 1000, H = 280, L = 44, R = 12, T = 12, B = 28;
  let lo = Infinity, hi = -Infinity;
  S.forEach(s => data.series[s[0]].forEach(v => {{ if (v != null) {{ lo = Math.min(lo, v); hi = Math.max(hi, v); }} }}));
  const pad = Math.max((hi - lo) * 0.1, 0.5); lo -= pad; hi += pad;
  const x = i => L + (W - L - R) * (n === 1 ? 0.5 : i / (n - 1)), y = v => T + (H - T - B) * (1 - (v - lo) / (hi - lo));
  let g = '';
  for (let k = 0; k <= 4; k++) {{ const v = lo + (hi - lo) * k / 4, yy = y(v).toFixed(1);
    g += '<line x1="' + L + '" x2="' + (W - R) + '" y1="' + yy + '" y2="' + yy + '" stroke="var(--grid)" stroke-width="1"/>'
      + '<text x="' + (L - 6) + '" y="' + yy + '" dy="4" text-anchor="end" font-size="11" fill="var(--muted)">' + v.toFixed(1) + '</text>'; }}
  const ticks = Math.min(6, n);
  for (let k = 0; k < ticks; k++) {{ const i = Math.round(k * (n - 1) / Math.max(1, ticks - 1));
    g += '<text x="' + x(i).toFixed(1) + '" y="' + (H - 8) + '" text-anchor="middle" font-size="11" fill="var(--muted)">' + data.dates[i].slice(5) + '</text>'; }}
  const paths = S.map(s => {{ let d = '', pen = false;
    data.series[s[0]].forEach((v, i) => {{ if (v == null) {{ pen = false; return; }}
      d += (pen ? 'L' : 'M') + x(i).toFixed(1) + ' ' + y(v).toFixed(1) + ' '; pen = true; }});
    return '<path d="' + d + '" fill="none" stroke="' + s[2] + '" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"'
      + (s[3] ? ' stroke-dasharray="' + s[3] + '"' : '') + '/>'; }}).join('');
  el.innerHTML = '<svg viewBox="0 0 ' + W + ' ' + H + '" preserveAspectRatio="none" role="img" aria-label="Growth of 100 by bot">'
    + g + paths + '<line id="xh" y1="' + T + '" y2="' + (H - B) + '" stroke="var(--axis)" stroke-width="1" visibility="hidden"/>'
    + '<g id="dots"></g><rect x="' + L + '" y="0" width="' + (W - L - R) + '" height="' + H + '" fill="transparent" id="hit"/></svg>'
    + '<div class="tip" id="tip"></div>';
  const svg = el.querySelector('svg'), tip = el.querySelector('#tip'), xh = el.querySelector('#xh'), dots = el.querySelector('#dots');
  function show(evt) {{
    const r = svg.getBoundingClientRect(), px = (evt.clientX - r.left) / r.width * W;
    const i = Math.max(0, Math.min(n - 1, Math.round((px - L) / (W - L - R) * (n - 1))));
    xh.setAttribute('x1', x(i)); xh.setAttribute('x2', x(i)); xh.setAttribute('visibility', 'visible');
    dots.innerHTML = S.map(s => {{ const v = data.series[s[0]][i]; return v == null ? '' :
      '<circle cx="' + x(i) + '" cy="' + y(v) + '" r="4" fill="' + s[2] + '" stroke="var(--surface-1)" stroke-width="2"/>'; }}).join('');
    tip.innerHTML = '<b>' + data.dates[i] + '</b>' + S.map(s => {{ const v = data.series[s[0]][i];
      return '<div class="row"><span>' + s[1] + '</span><span>' + (v == null ? '—' : v.toFixed(2)) + '</span></div>'; }}).join('');
    tip.style.display = 'block';
    const tx = (x(i) / W) * r.width; tip.style.left = Math.min(tx + 12, r.width - tip.offsetWidth - 4) + 'px'; tip.style.top = '8px';
  }}
  svg.addEventListener('mousemove', show);
  svg.addEventListener('mouseleave', () => {{ tip.style.display = 'none'; xh.setAttribute('visibility', 'hidden'); dots.innerHTML = ''; }});
}})();
</script></body></html>
"""
