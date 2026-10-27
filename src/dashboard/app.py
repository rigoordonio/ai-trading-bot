"""Streamlit dashboard (dark theme in .streamlit/config.toml).

    streamlit run src/dashboard/app.py

Read-only: there is no control that can place an order. The Backtesting page
runs research backtests in-process.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from src.ai.explanation import DecisionFacts, explain  # noqa: E402
from src.analytics.attribution import cost_breakdown, exit_reason_breakdown, pnl_by  # noqa: E402
from src.analytics.store import ResultsStore  # noqa: E402
from src.bots.bot import build_bots  # noqa: E402
from src.config import load_config  # noqa: E402
from src.dashboard import data as D  # noqa: E402
from src.execution.broker import trading_mode  # noqa: E402

st.set_page_config(page_title="AI Trading Platform", page_icon="📈", layout="wide")
ACCENT, NEG, MUTED = "#3FB8AF", "#F85149", "#8B949E"
PAGES = ["Overview", "Bots", "Markets", "Signals", "Portfolio", "Trades", "Backtesting", "ML Models", "Risk",
         "System Health"]


@st.cache_resource
def cfg():
    return load_config()


@st.cache_resource
def store():
    return ResultsStore()


@st.cache_data(ttl=3600, show_spinner="Loading market data...")
def prices(provider: str):
    from src.workflows import load_market_data
    return load_market_data(cfg(), provider, save=False).data


def fmt_money(x):
    return "—" if x is None or pd.isna(x) else f"${x:,.0f}"


def fmt_pct(x):
    return "—" if x is None or pd.isna(x) else f"{x:+.2%}"


def line_chart(series: dict[str, pd.Series], title: str, height=360, yfmt=None):
    fig = go.Figure()
    for i, (name, s) in enumerate(series.items()):
        fig.add_trace(go.Scatter(x=s.index, y=s.values, name=name, mode="lines",
                                 line=dict(width=2 if i == 0 else 1.2, dash="dash" if "Bench" in name else None)))
    fig.update_layout(template="plotly_dark", title=title, height=height, margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", y=1.08), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    if yfmt:
        fig.update_yaxes(tickformat=yfmt)
    st.plotly_chart(fig, use_container_width=True)


def status_banner():
    s = store().system_status()
    enabled = s.get("trading_enabled")
    health = s.get("health", {}).get("overall", "UNKNOWN")
    color = ACCENT if enabled else NEG
    st.markdown(
        f"<div style='padding:8px 14px;border-radius:8px;background:#161B22;border-left:6px solid {color}'>"
        f"<b>MODE:</b> {trading_mode()} &nbsp;|&nbsp; <b>REAL MONEY:</b> DISABLED &nbsp;|&nbsp; "
        f"<b>TRADING:</b> {'ENABLED' if enabled else 'DISABLED' if enabled is False else 'NO PAPER RUN YET'} "
        f"&nbsp;|&nbsp; <b>SYSTEM HEALTH:</b> {health}</div>", unsafe_allow_html=True)


def pick_run(key="run"):
    runs = store().runs()
    if not runs:
        st.info("No backtest results yet. Run `python scripts/backtest.py --bots` or use the Backtesting page.")
        return None
    labels = {f"{r['run_id']}  ({', '.join(r.get('bots', []))})": r["run_id"] for r in runs}
    return labels[st.sidebar.selectbox("Backtest run", list(labels), key=key)]


# ----------------------------------------------------------------- pages
def page_overview():
    run = pick_run()
    if not run:
        return
    eq = store().run_frame(run, "equity")
    k = D.overview_kpis(eq)
    cols = st.columns(7)
    for c, (name, v) in zip(cols, k.items(), strict=False):
        c.metric(name, fmt_pct(v) if name in ("Market exposure", "Max drawdown") else
                 f"{v:.2f}" if name == "Sharpe" else fmt_money(v))
    bench = store().run_frame(run, "benchmark")
    series = {"Account": eq["equity"]}
    if not bench.empty:
        series["Benchmark (buy & hold)"] = bench.iloc[:, 0]
    line_chart(series, "Equity curve")
    line_chart({"Drawdown": D.drawdown(eq["equity"])}, "Drawdown", 220, ".0%")


def page_bots():
    st.subheader("Multi-bot overview")
    c = cfg()
    alloc = {b: x.capital_allocation for b, x in (c.bots.bots.items() if c.bots else []) if x.enabled}
    snaps, sigs = D.paper_history(store().paper)
    source = st.radio("Source", ["Paper trading", "Backtest run"], horizontal=True,
                      index=0 if not snaps.empty else 1)
    if source == "Paper trading":
        if snaps.empty:
            st.info("No paper history yet. It appears after the first `python scripts/paper_trade.py` run.")
        else:
            ov = D.multi_bot_overview(snaps, alloc, D.active_signal_counts(sigs))
            st.dataframe(_fmt_overview(ov), use_container_width=True)
    run = pick_run("run_bots")
    if not run:
        return
    m = store().run_metrics(run)
    if source == "Backtest run":
        bot_eq = {b: store().run_frame(run, f"equity_{b}") for b in m.get("bot_metrics", {})}
        trades = store().run_frame(run, "trades")
        audit = store().run_frame(run, "audit")
        snaps_bt = D.snapshots_from_backtest(bot_eq, store().run_frame(run, "equity"), trades)
        ov = D.multi_bot_overview(snaps_bt, alloc, D.active_signal_counts(audit) if not audit.empty else {})
        st.dataframe(_fmt_overview(ov), use_container_width=True)
        st.caption("Model accuracy is not shown here on purpose: it does not measure trading quality.")
        div = {k: m.get(k) for k in ("avg_pairwise_bot_correlation", "diversification_ratio", "capital_utilisation",
                                     "avg_gross_exposure") if k in m}
        if div:
            st.caption("Measured: " + ", ".join(f"{k} = {v:.2f}" for k, v in div.items() if isinstance(v, float)))
    st.subheader("Backtest metrics by bot")
    rows = [{"Bot": b, "Return": v["total_return"], "CAGR": v["annualised_return"], "Sharpe": v["sharpe"],
             "Max DD": v["max_drawdown"], "Trades": v["num_trades"], "Win rate": v["win_rate"],
             "Turnover": v["turnover"]} for b, v in m.get("bot_metrics", {}).items()]
    st.dataframe(pd.DataFrame(rows).set_index("Bot").style.format(
        {"Return": "{:+.2%}", "CAGR": "{:+.2%}", "Sharpe": "{:.2f}", "Max DD": "{:.2%}", "Win rate": "{:.0%}",
         "Turnover": "{:.1f}"}), use_container_width=True)
    series = {b: store().run_frame(run, f"equity_{b}")["equity"] for b in m.get("bot_metrics", {})}
    if series:
        line_chart(series, "Equity by bot (each on its own capital sleeve)")
    c = cfg()
    if c.bots:
        st.subheader("Bot definitions")
        st.dataframe(pd.DataFrame([{"Bot": b, "Type": x.bot_type, "Timeframe": x.timeframe,
                                    "Holding period": f"{x.holding_period.min}-{x.holding_period.max} "
                                                      f"{x.holding_period.unit}",
                                    "Capital": f"{x.capital_allocation:.0%}", "Rebalance": x.rebalance_frequency,
                                    "Strategy version": x.strategy_version,
                                    "Model version": x.ml.model_version if x.ml else "—",
                                    "Components": ", ".join(x.components)} for b, x in c.bots.bots.items()])
                     .set_index("Bot"), use_container_width=True)


def page_markets():
    c = cfg()
    provider = st.sidebar.selectbox("Data provider", ["synthetic", "yfinance", "csv"], index=0)
    bots = build_bots(c)
    bot_id = st.sidebar.selectbox("Bot", list(bots))
    try:
        px = prices(provider)
    except Exception as exc:  # noqa: BLE001 - shown to the user, non-trading page
        st.error(f"Market data unavailable: {exc}")
        return
    bot = bots[bot_id]
    panel = bot.compute_features(px)
    preds = None
    if bot.ml_cfg:
        from src.config import resolve_path
        from src.models.prediction import Predictor, load_model
        try:
            lm = load_model(resolve_path(c.settings.ml.artifacts_dir), bot_id, bot.ml_cfg.model_version,
                            bot.feature_version)
            last = panel.xs(panel.index.get_level_values("date").max(), level="date", drop_level=False)
            preds = Predictor.from_loaded(lm).predict(last, bot.feature_version)
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Model unavailable ({exc}). ML-dependent signals show NO SIGNAL (fail closed).")
    t = D.market_table(bot, panel, preds)
    st.caption(f"As of {panel.index.get_level_values('date').max().date()} | provider: {provider}")
    st.dataframe(t.style.format({"Price": "{:,.2f}", "Daily %": "{:+.2%}", "Momentum 20d": "{:+.2%}",
                                 "Volatility 20d": "{:.1%}", "ML probability": "{:.2f}", "Score": "{:.2f}"},
                                na_rep="—"), use_container_width=True)


def page_signals():
    run = pick_run("run_sig")
    if not run:
        return
    a = store().run_frame(run, "audit")
    if a.empty:
        st.info("No signals recorded.")
        return
    bots = ["All"] + sorted(a["bot_id"].dropna().unique())
    b = st.sidebar.selectbox("Bot", bots)
    acts = st.sidebar.multiselect("Final action", sorted(a["final_action"].dropna().unique()),
                                  default=[x for x in ("BUY", "SELL") if x in set(a["final_action"])])
    df = a if b == "All" else a[a["bot_id"] == b]
    if acts:
        df = df[df["final_action"].isin(acts)]
    show = df[["timestamp", "bot_id", "symbol", "strategy", "action", "final_action", "score", "ml_probability",
               "risk_status", "rule_triggered", "reason"]].rename(columns={"score": "confidence"})
    st.dataframe(show.sort_values("timestamp", ascending=False).head(1000), use_container_width=True)
    st.subheader("Why? (explanation of a decision)")
    if len(df):
        idx = st.number_input("Row (most recent = 0)", 0, len(df) - 1, 0)
        rec = df.sort_values("timestamp", ascending=False).iloc[int(idx)].to_dict()
        st.write(explain(DecisionFacts.from_audit(rec)))
        st.caption("Explanations describe decisions already made by the rules/model/risk engine. "
                   "The LLM layer cannot create orders.")


def _fmt_overview(ov: pd.DataFrame) -> pd.DataFrame:
    money = {"Portfolio value", "Cash", "Daily P&L", "Total P&L"}
    pct = {"Capital allocation", "Return", "Volatility", "Max drawdown"}
    out = ov.astype(object).copy()
    for r in ov.index:
        for c in ov.columns:
            v = ov.at[r, c]
            if v is None or (isinstance(v, float) and not np.isfinite(v)):
                out.at[r, c] = "—"
            elif r in money:
                out.at[r, c] = fmt_money(v)
            elif r in pct:
                out.at[r, c] = fmt_pct(v)
            elif r == "Sharpe":
                out.at[r, c] = f"{v:.2f}"
            else:
                out.at[r, c] = f"{int(v)}"
    return out


def page_portfolio():
    pos = store().paper_positions()
    st.subheader("Paper portfolio (by bot sleeve)")
    if pos.empty:
        st.info("No paper positions. Run `python scripts/paper_trade.py`.")
    else:
        total = pos["market_value"].sum()
        pos["weight"] = pos["market_value"] / max(total, 1e-9)
        try:
            from src.risk.portfolio_risk import covariance, risk_contributions
            px = prices("synthetic")
            rets = pd.DataFrame({s: d["close"] for s, d in px.items()}).pct_change().tail(60)
            agg = pos.groupby("symbol")["weight"].sum().to_dict()
            rc = risk_contributions(agg, covariance(rets, list(agg)))
            pos["risk_contribution"] = pos["symbol"].map(rc)
        except Exception:  # noqa: BLE001 - optional analytic
            pos["risk_contribution"] = float("nan")
        st.dataframe(pos.style.format({"avg_cost": "{:,.2f}", "last_price": "{:,.2f}", "market_value": "{:,.0f}",
                                       "unrealized_pnl": "{:+,.0f}", "weight": "{:.1%}",
                                       "risk_contribution": "{:.1%}"}, na_rep="—"), use_container_width=True)
    bs = store().broker_state()
    if bs:
        st.caption(f"Broker (authoritative) cash: {fmt_money(bs.get('cash'))}")


def page_trades():
    run = pick_run("run_tr")
    if not run:
        return
    t = store().run_frame(run, "trades")
    f = store().run_frame(run, "fills")
    if t.empty:
        st.info("No closed trades.")
        return
    st.dataframe(t.sort_values("exit_date", ascending=False), use_container_width=True)
    c1, c2 = st.columns(2)
    c1.subheader("P&L by symbol")
    c1.dataframe(pnl_by(t, "symbol"), use_container_width=True)
    c2.subheader("P&L by exit reason")
    c2.dataframe(exit_reason_breakdown(t), use_container_width=True)
    st.subheader("Cost drag")
    st.json({k: round(v, 2) for k, v in cost_breakdown(f).items()})


def page_backtesting():
    c = cfg()
    st.subheader("Run a research backtest")
    with st.form("bt"):
        provider = st.selectbox("Data provider", ["synthetic", "yfinance", "csv"])
        mode = st.selectbox("What to test", ["All bots (one account)"] + [f"Bot: {b}" for b in (c.bots.bots if c.bots
                                                                                                else [])]
                            + [f"Strategy: {s}" for s in c.strategies.strategies if s != "ml"] + ["Strategy: ensemble"])
        universe = st.text_input("Universe (comma separated)", ",".join(c.settings.universe.symbols))
        col1, col2, col3 = st.columns(3)
        start = col1.date_input("Start", pd.Timestamp(c.settings.backtest.start))
        end = col2.date_input("End", pd.Timestamp.today())
        cost_mult = col3.slider("Transaction cost multiplier", 0.0, 5.0, 1.0, 0.5)
        go_btn = st.form_submit_button("Run backtest")
    if go_btn:
        from src.analytics.performance import save_result
        from src.config import load_config as lc
        from src.workflows import bots_backtest, research_backtest
        syms = [s.strip().upper() for s in universe.split(",") if s.strip()]
        c2 = lc(overrides={"settings": {"universe": {"symbols": syms}}})
        with st.spinner("Backtesting (walk-forward ML for bots can take a minute)..."):
            px = prices(provider)
            if mode.startswith("All bots"):
                res, _ = bots_backtest(c2, px, None, str(start), str(end), cost_mult)
            elif mode.startswith("Bot: "):
                res, _ = bots_backtest(c2, px, [mode[5:]], str(start), str(end), cost_mult)
            else:
                res = research_backtest(c2, px, mode[10:], str(start), str(end), cost_mult)
        save_result(res, ROOT / "reports")
        st.success(f"Saved as {res.run_id}")
        m = res.metrics
        cols = st.columns(6)
        for col, (k, fmt) in zip(cols, [("total_return", "{:+.2%}"), ("annualised_return", "{:+.2%}"),
                                         ("sharpe", "{:.2f}"), ("sortino", "{:.2f}"), ("max_drawdown", "{:.2%}"),
                                         ("num_trades", "{}")], strict=True):
            col.metric(k.replace("_", " ").title(), fmt.format(m.get(k, 0)))
        series = {"Strategy": res.equity["equity"]}
        if res.benchmark is not None:
            series["Benchmark (buy & hold)"] = res.benchmark
        line_chart(series, "Equity vs benchmark")
        line_chart({"Drawdown": D.drawdown(res.equity["equity"])}, "Drawdown", 220, ".0%")
        mt = D.monthly(res.equity["equity"])
        if not mt.empty:
            st.subheader("Monthly returns")
            st.dataframe(mt.style.format("{:+.1%}", na_rep=""), use_container_width=True)
        st.subheader("All metrics")
        st.json({k: v for k, v in m.items() if not isinstance(v, dict)})
        st.caption("A single backtest is one path. Use scripts/robustness.py for cost/parameter/start-date "
                   "sensitivity and Monte Carlo distributions.")


def page_ml():
    arts = D.artifacts(cfg())
    if not arts:
        st.info("No trained models. Run `python scripts/train_model.py`.")
        return
    labels = {f"{a['bot_id']} / {a['model_version']} ({a['model_name']})": a for a in arts}
    a = labels[st.sidebar.selectbox("Model", list(labels))]
    p = a["periods"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Model", a["model_name"])
    c2.metric("Training period", f"{p['train'][0]} → {p['final_fit'][1]}")
    c3.metric("Out-of-sample (test)", f"{p['test'][0]} → {p['test'][1] or 'latest'}")
    st.caption(f"Feature set {a['feature_set_version']} | target: forward {a['target']['horizon_days']}d return > "
               f"{a['target']['threshold']:.1%} | sha256 {a['sha256'][:12]}…")
    test = a["metrics"].get("test") or {}
    cols = st.columns(6)
    keys = ["accuracy", "precision", "recall", "roc_auc", "ic_spearman", "quintile_spread"]
    for col, k in zip(cols, keys, strict=True):
        col.metric(k.replace("_", " "), f"{test.get(k, float('nan')):.3f}")
    st.subheader("Trading-oriented evaluation (test period)")
    q = test.get("quintile_returns", {})
    if q:
        fig = go.Figure(go.Bar(x=[f"Q{int(k) + 1}" for k in q], y=list(q.values()), marker_color=ACCENT))
        fig.update_layout(template="plotly_dark", title="Mean forward return by predicted-probability quintile",
                          height=300, yaxis_tickformat=".2%", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Accuracy alone is not enough: a useful model ranks higher probabilities above higher returns.")
    st.subheader("Validation scores by candidate")
    st.dataframe(pd.DataFrame(a["metrics"].get("validation", {})).T[["roc_auc", "accuracy", "precision", "recall"]],
                 use_container_width=True)
    fi = a["metrics"].get("feature_importance") or {}
    if fi:
        s = pd.Series(fi).sort_values()
        fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h", marker_color=ACCENT))
        fig.update_layout(template="plotly_dark", title="Permutation importance (validation, ΔAUC)", height=420,
                          paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)


def page_risk():
    run = pick_run("run_risk")
    s = store().system_status()
    st.subheader("Live (paper) risk state")
    st.write({"trading_enabled": s.get("trading_enabled"), "blocking_reasons": s.get("blocking_reasons")})
    if run:
        eq = store().run_frame(run, "equity")
        ev = store().run_frame(run, "risk_events")
        c = st.columns(5)
        c[0].metric("Gross exposure", fmt_pct(eq["gross_exposure"].iloc[-1]))
        c[1].metric("Net exposure", fmt_pct(eq["gross_exposure"].iloc[-1]))
        c[2].metric("Max drawdown", fmt_pct(D.drawdown(eq["equity"]).min()))
        vol = eq["equity"].pct_change().tail(60).std() * (252 ** 0.5)
        c[3].metric("Portfolio vol (60d)", f"{vol:.1%}")
        c[4].metric("Positions", int(eq["n_positions"].iloc[-1]))
        line_chart({"Gross exposure": eq["gross_exposure"]}, "Exposure", 220, ".0%")
        if not ev.empty:
            st.subheader("Risk violations / events")
            st.dataframe(ev.groupby(["event", "rule"], dropna=False).size().rename("count").reset_index()
                         .sort_values("count", ascending=False), use_container_width=True)
            st.dataframe(ev.sort_values("date", ascending=False).head(500), use_container_width=True)


def page_health():
    s = store().system_status()
    h = s.get("health", {})
    seen = h.get("last_seen", {})
    c = st.columns(4)
    c[0].metric("SYSTEM HEALTH", h.get("overall", "UNKNOWN"))
    c[1].metric("TRADING STATUS", "ENABLED" if s.get("trading_enabled") else "DISABLED")
    c[2].metric("LAST MARKET DATA", seen.get("last_market_data", "—"))
    c[3].metric("LAST RECONCILIATION", (seen.get("last_reconciliation") or "—")[:25])
    c = st.columns(3)
    c[0].metric("LAST SIGNAL", seen.get("last_signal", "—"))
    c[1].metric("LAST ORDER", seen.get("last_order", "—"))
    cb = s.get("circuit_breakers", {})
    c[2].metric("CIRCUIT BREAKERS", ", ".join(f"{k}: {v}" for k, v in cb.items()) or "—")
    if s.get("blocking_reasons"):
        st.error("Trading blocked because:\n\n- " + "\n- ".join(s["blocking_reasons"]))
    st.subheader("Components")
    comps = h.get("components", {})
    if comps:
        st.dataframe(pd.DataFrame(comps).T, use_container_width=True)
    st.subheader("Active risk / system events")
    ev = s.get("recent_events", [])
    st.dataframe(pd.DataFrame(ev), use_container_width=True) if ev else st.write("None")
    if s.get("reconciliation"):
        st.subheader("Last reconciliation")
        st.json(s["reconciliation"])


ROUTES = {"Overview": page_overview, "Bots": page_bots, "Markets": page_markets, "Signals": page_signals,
          "Portfolio": page_portfolio, "Trades": page_trades, "Backtesting": page_backtesting, "ML Models": page_ml,
          "Risk": page_risk, "System Health": page_health}


def main():
    st.sidebar.title("📈 AI Trading Platform")
    page = st.sidebar.radio("Navigation", PAGES)
    status_banner()
    st.title(page)
    ROUTES[page]()


main()
