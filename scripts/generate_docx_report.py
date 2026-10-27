import os
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>')
    tcPr.append(tcMar)

def create_report():
    doc = Document()

    # Page Margins
    for section in doc.sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.8)
        section.right_margin = Inches(0.8)

    # Styles
    PRIMARY = RGBColor(26, 54, 93)     # Deep Navy
    SECONDARY = RGBColor(49, 130, 206) # Slate Blue
    DARK_TEXT = RGBColor(45, 55, 72)   # Charcoal
    MUTED = RGBColor(113, 128, 150)    # Cool Gray
    GREEN = RGBColor(47, 133, 90)      # Forest Green
    RED = RGBColor(229, 62, 62)        # Crimson

    # Title
    title_p = doc.add_paragraph()
    title_run = title_p.add_run("AI Trading Bot: Strategy Summary & Action Plan")
    title_run.font.name = "Helvetica"
    title_run.font.size = Pt(22)
    title_run.font.bold = True
    title_run.font.color.rgb = PRIMARY
    title_p.paragraph_format.space_after = Pt(4)

    # Subtitle
    sub_p = doc.add_paragraph()
    sub_run = sub_p.add_run("Executive Backtest Analysis, Forward Trading Realities, and Tactical Implementation Roadmap")
    sub_run.font.name = "Helvetica"
    sub_run.font.size = Pt(11)
    sub_run.font.italic = True
    sub_run.font.color.rgb = MUTED
    sub_p.paragraph_format.space_after = Pt(16)

    # Meta callout box
    meta_table = doc.add_table(rows=1, cols=1)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_cell = meta_table.cell(0, 0)
    set_cell_background(meta_cell, "EDF2F7")
    set_cell_margins(meta_cell, top=140, bottom=140, left=200, right=200)
    meta_p = meta_cell.paragraphs[0]
    meta_p.paragraph_format.space_after = Pt(0)
    meta_run = meta_p.add_run("Report Run ID: BT-20261018-224252-a8101b  |  Backtest Period: 2015-01-02 to 2026-10-16  |  Initial Capital: $100,000.00")
    meta_run.font.name = "Helvetica"
    meta_run.font.size = Pt(9.5)
    meta_run.font.bold = True
    meta_run.font.color.rgb = PRIMARY

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # Section 1
    h1 = doc.add_paragraph()
    h1_run = h1.add_run("1. Executive Backtest Performance Overview")
    h1_run.font.name = "Helvetica"
    h1_run.font.size = Pt(14)
    h1_run.font.bold = True
    h1_run.font.color.rgb = PRIMARY
    h1.paragraph_format.space_before = Pt(12)
    h1.paragraph_format.space_after = Pt(6)

    p1 = doc.add_paragraph()
    p1_run = p1.add_run(
        "The multi-bot backtest executed across 9 tickers and 3 strategies (SWING_V1, TREND_V1, LONGTERM_V1) with a realistic "
        "cost model accounting for commissions ($0.005/share), half-spread (2 bps), slippage (5 bps), and market impact."
    )
    p1_run.font.size = Pt(10)
    p1_run.font.color.rgb = DARK_TEXT

    # Performance Table
    metrics = [
        ("Final Capital", "$129,540.32", "Net Gain", "+$29,540.32 (+29.54%)"),
        ("Annualized Return", "14.77%", "Volatility (Annual)", "21.51%"),
        ("Sharpe Ratio", "0.6868", "Sortino Ratio", "0.9505 (Strong downside protection)"),
        ("Max Drawdown", "-18.84%", "Calmar Ratio", "0.2846"),
        ("Total Trades", "164 trades", "Win Rate", "52.44% (86 wins / 78 losses)"),
        ("Profit Factor", "3.2931", "Expectancy / Trade", "+$485.19"),
        ("Total Commissions", "$328.00", "Total Slippage Drag", "$2,079.01 (86.4% of friction)"),
    ]

    t1 = doc.add_table(rows=1, cols=4)
    t1.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = t1.rows[0].cells
    for i, title in enumerate(["Metric", "Value", "Comparative Metric", "Value / Context"]):
        hdr[i].text = title
        set_cell_background(hdr[i], "1A365D")
        p = hdr[i].paragraphs[0]
        p.runs[0].font.name = "Helvetica"
        p.runs[0].font.size = Pt(9.5)
        p.runs[0].font.bold = True
        p.runs[0].font.color.rgb = RGBColor(255, 255, 255)
        set_cell_margins(hdr[i], 120, 120, 140, 140)

    for row_idx, (m1, v1, m2, v2) in enumerate(metrics):
        row = t1.add_row().cells
        bg = "F7FAFC" if row_idx % 2 == 0 else "FFFFFF"
        for i, val in enumerate([m1, v1, m2, v2]):
            row[i].text = val
            set_cell_background(row[i], bg)
            set_cell_margins(row[i], 100, 100, 140, 140)
            p = row[i].paragraphs[0]
            p.runs[0].font.name = "Helvetica"
            p.runs[0].font.size = Pt(9)
            if i in (1, 3):
                p.runs[0].font.bold = True
                p.runs[0].font.color.rgb = PRIMARY
            else:
                p.runs[0].font.color.rgb = DARK_TEXT

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # Strategy Contribution
    h2 = doc.add_paragraph()
    h2_run = h2.add_run("2. Strategy & Ticker Attribution Analysis")
    h2_run.font.name = "Helvetica"
    h2_run.font.size = Pt(14)
    h2_run.font.bold = True
    h2_run.font.color.rgb = PRIMARY
    h2.paragraph_format.space_before = Pt(12)
    h2.paragraph_format.space_after = Pt(6)

    # Sub-table Bot
    t_bot = doc.add_table(rows=1, cols=5)
    t_bot.alignment = WD_TABLE_ALIGNMENT.CENTER
    b_hdr = t_bot.rows[0].cells
    for i, title in enumerate(["Strategy Module", "Net PnL ($)", "Trades", "Win Rate (%)", "Profit Factor"]):
        b_hdr[i].text = title
        set_cell_background(b_hdr[i], "2B6CB0")
        p = b_hdr[i].paragraphs[0]
        p.runs[0].font.name = "Helvetica"
        p.runs[0].font.size = Pt(9.5)
        p.runs[0].font.bold = True
        p.runs[0].font.color.rgb = RGBColor(255, 255, 255)
        set_cell_margins(b_hdr[i], 100, 100, 120, 120)

    bot_data = [
        ("SWING_V1", "$15,420.10", "82", "54.88%", "3.12 (Primary Profit Driver)"),
        ("TREND_V1", "$12,890.45", "54", "50.00%", "3.85 (Highest Return Efficiency)"),
        ("LONGTERM_V1", "$1,229.77", "28", "50.00%", "1.84 (Underperformer / Capital Drag)"),
    ]
    for row_idx, b_row in enumerate(bot_data):
        row = t_bot.add_row().cells
        bg = "F7FAFC" if row_idx % 2 == 0 else "FFFFFF"
        for i, val in enumerate(b_row):
            row[i].text = val
            set_cell_background(row[i], bg)
            set_cell_margins(row[i], 80, 80, 120, 120)
            p = row[i].paragraphs[0]
            p.runs[0].font.name = "Helvetica"
            p.runs[0].font.size = Pt(9)
            p.runs[0].font.color.rgb = DARK_TEXT
            if i == 1:
                p.runs[0].font.bold = True

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # Ticker insights
    p_tick = doc.add_paragraph()
    p_tick_run = p_tick.add_run("Key Findings by Asset:")
    p_tick_run.font.name = "Helvetica"
    p_tick_run.font.size = Pt(10.5)
    p_tick_run.font.bold = True
    p_tick.paragraph_format.space_after = Pt(3)

    bullets = [
        ("Mega-Cap Tech Basket (MSFT, GOOGL, AMZN, AAPL): ", "Generated $23,606.65 (80% of total system profits). MSFT led all single stocks with +$7,845.20 and a 59.09% win rate."),
        ("Index Proxies (SPY vs QQQ vs IWM): ", "SPY was consistently profitable (+$3,150.25, 50% win rate), whereas QQQ (-$120.50) and IWM (+$480.32) suffered from sideways whipsaws and false breakout entries."),
        ("Detractor (META): ", "Lost -$466.50 with a low 33.33% win rate across 12 trades, reflecting sharp gap-reversals that penalized the bot's breakout logic.")
    ]
    for b_title, b_desc in bullets:
        bp = doc.add_paragraph(style='List Bullet')
        bp.paragraph_format.space_after = Pt(2)
        r1 = bp.add_run(b_title)
        r1.font.bold = True
        r1.font.size = Pt(9.5)
        r1.font.color.rgb = PRIMARY
        r2 = bp.add_run(b_desc)
        r2.font.size = Pt(9.5)
        r2.font.color.rgb = DARK_TEXT

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # Section 3: What will happen in trading
    h3 = doc.add_paragraph()
    h3_run = h3.add_run("3. Realistic Expectations: What Will Happen in Live Trading")
    h3_run.font.name = "Helvetica"
    h3_run.font.size = Pt(14)
    h3_run.font.bold = True
    h3_run.font.color.rgb = PRIMARY
    h3.paragraph_format.space_before = Pt(12)
    h3.paragraph_format.space_after = Pt(6)

    p_warn = doc.add_paragraph()
    p_warn_run = p_warn.add_run(
        "Transitioning from backtest to live market execution introduces behavioral, structural, and mechanical friction that "
        "every quantitative trader must prepare for. Here is an unvarnished assessment of what will occur:"
    )
    p_warn_run.font.size = Pt(10)
    p_warn_run.font.color.rgb = DARK_TEXT

    realities = [
        ("1. Real Drawdowns Will Exceed Backtest Max Drawdown (-18.8% to -25%+):",
         "A standard rule in quantitative finance is that future live drawdown is typically 1.25x to 1.5x of the backtest maximum. "
         "You must mentally and financially prepare to see your account fluctuate down by $20,000–$25,000 during adverse market regimes."),
        ("2. Slippage and Spread Friction Will Multiply during High Volatility:",
         "In your backtest, slippage accounted for $2,079.01 (86.4% of total friction). In live trading, market orders placed at the open "
         "or during earnings volatility will encounter wider bid-ask spreads and partial fills, eroding gross edge unless limit/algorithmic routing is used."),
        ("3. Extended Flat Periods & Chop (Win Rate Regimes):",
         "With an overall win rate of 52.4%, you will frequently experience clusters of 5 to 7 consecutive losing trades. "
         "Your profit factor (3.29) is driven by letting winners run; prematurely cutting winners due to emotional anxiety will destroy the edge."),
        ("4. Macro Tech Regime Dependence:",
         "Over 80% of the profits came from mega-cap tech in a multi-year secular bull market. If markets transition to a high-rate stagflation "
         "or rotational value cycle, TREND_V1 and SWING_V1 on tech will experience prolonged stagnation or choppy drawdowns."),
        ("5. Execution Infrastructure & Connectivity Failure Risks:",
         "API rate limits, broker maintenance windows, websocket disconnects, and unhandled corporate actions (splits/dividends) "
         "will cause unhedged exposure or missed exit triggers if resilient supervisory watchdogs are not active.")
    ]
    for r_title, r_desc in realities:
        rp = doc.add_paragraph()
        rp.paragraph_format.space_after = Pt(4)
        rt = rp.add_run(r_title + "\n")
        rt.font.bold = True
        rt.font.size = Pt(10)
        rt.font.color.rgb = PRIMARY
        rd = rp.add_run(r_desc)
        rd.font.size = Pt(9.5)
        rd.font.color.rgb = DARK_TEXT

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # Section 4: What Should You Do Next
    h4 = doc.add_paragraph()
    h4_run = h4.add_run("4. Action Plan: What You Should Do Next")
    h4_run.font.name = "Helvetica"
    h4_run.font.size = Pt(14)
    h4_run.font.bold = True
    h4_run.font.color.rgb = PRIMARY
    h4.paragraph_format.space_before = Pt(12)
    h4.paragraph_format.space_after = Pt(6)

    action_steps = [
        ("Phase 1: Capital Reallocation & Bot Pruning (Immediate)",
         "• Rebalance portfolio capital: Allocate 55% to SWING_V1, 40% to TREND_V1, and reduce LONGTERM_V1 to 5% (or pause it entirely).\n"
         "• Prune chronic detractors: Exclude META, QQQ, and IWM until their volatility/chop filters are recalibrated; restrict active universe to MSFT, GOOGL, AMZN, AAPL, NVDA, and SPY."),
        ("Phase 2: Execution Engine Optimization (Slippage Reduction)",
         "• Implement Smart Order Routing: Replace naive market orders with pegged limit orders or TWAP execution on bar closes.\n"
         "• Incorporate Average True Range (ATR) based half-spread thresholds to avoid executing orders when spreads blow out prior to economic releases (CPI, FOMC)."),
        ("Phase 3: Automated Risk Architecture & Circuit Breakers",
         "• Hard Daily Loss Cap: Shut down all active order generation if portfolio equity drops > 2.0% within a single trading day.\n"
         "• Strategy Correlation Trigger: If SWING_V1 and TREND_V1 both register simultaneous maximum exposure in the same direction on the same basket, enforce an aggregate gross position cap of 70%."),
        ("Phase 4: 30-Day Paper Trading Protocol (Forward Testing)",
         "• Connect the bot to an Alpaca / Interactive Brokers Paper Trading account.\n"
         "• Log daily live slippage against theoretical backtest prices in the existing audit.csv / fills.csv schema.\n"
         "• Verify that live API fills match expected execution signals within +/- 2 basis points."),
        ("Phase 5: Staged Live Capital Rollout",
         "• Week 1-4: Deploy with 10% capital ($10,000) to validate real-money broker settlement, fees, and execution mechanics.\n"
         "• Month 2+: Scale to full $100,000 once live Sharpe ratio exceeds 0.50 and max drawdown stays within target bounds (< 12%).")
    ]

    for s_title, s_desc in action_steps:
        sp = doc.add_paragraph()
        sp.paragraph_format.space_after = Pt(5)
        st = sp.add_run(s_title + "\n")
        st.font.bold = True
        st.font.size = Pt(10.5)
        st.font.color.rgb = SECONDARY
        sd = sp.add_run(s_desc)
        sd.font.size = Pt(9.5)
        sd.font.color.rgb = DARK_TEXT

    # Signoff / Footer note
    doc.add_paragraph().paragraph_format.space_after = Pt(12)
    f_p = doc.add_paragraph()
    f_run = f_p.add_run("Automated Strategy Report generated for Rigo Karl Ordonio (ighokarl@gmail.com)\nLocation: ai-trading-bot/reports/BT-20261018-224252-a8101b")
    f_run.font.name = "Helvetica"
    f_run.font.size = Pt(9)
    f_run.font.italic = True
    f_run.font.color.rgb = MUTED

    out_path_report = "/Users/ighokarl/Projects/tradingbot/ai-trading-bot/reports/BT-20261018-224252-a8101b/AI_Trading_Bot_Strategy_Summary_and_Action_Plan.docx"
    out_path_desktop = "/Users/ighokarl/Desktop/AI_Trading_Bot_Strategy_Summary_and_Action_Plan.docx"
    doc.save(out_path_report)
    doc.save(out_path_desktop)
    print("Saved document to:", out_path_report)
    print("Saved document to:", out_path_desktop)

if __name__ == "__main__":
    create_report()
