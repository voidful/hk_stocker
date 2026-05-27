#!/usr/bin/env python3
"""Combined HK master strategy report.

Master v1 = Benchmark Beater v1.1 core swing book + Day Trade v1.1 intraday
overlay. The overlay is represented as additional same-day PnL on top of the
core equity curve, because the day-trade sleeve is flat overnight.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmark_beater_report import compare_with_benchmark, latest_signals
from day_trade_report import latest_daytrade_signals, next_trading_day, run_daytrade_backtest
from strategy.ai_strategy import build_liquid_universe, engineer_features, fetch_panel_data
from strategy.benchmark import fetch_benchmark
from strategy.event_backtest import EventDrivenBacktester
from strategy.market import (
    CURRENCY,
    DEFAULT_BENCHMARK,
    DEFAULT_BENCHMARK_LABEL,
    DEFAULT_BUY_COST,
    DEFAULT_SELL_COST,
    SECONDARY_BENCHMARK,
    SECONDARY_BENCHMARK_LABEL,
)
from strategy.risk_metrics import compute_risk_metrics, format_metrics_summary
from strategy.universe import DEFAULT_TICKERS, EXTENDED_TICKERS


STRATEGY_VERSION = "master-v1"


def pct(value):
    return float(value) * 100


def format_pct(value, digits=1):
    return f"{pct(value):+.{digits}f}%"


def run_swing_book(close_df, open_df, high_df, low_df, vol_df, args):
    universe_mask = build_liquid_universe(close_df, vol_df, top_n=args.swing_universe)
    total_score, ma_df, _, _ = engineer_features(
        close_df,
        vol_df,
        universe_mask,
        ma_period=args.swing_ma_period,
        rsi_weight=args.swing_rsi_weight,
        breakout_weight=args.swing_breakout_weight,
    )
    backtester = EventDrivenBacktester(
        initial_capital=args.capital,
        position_size=args.swing_position_size,
        tp_sl_mode="atr",
        tp_atr_mult=args.swing_tp_atr,
        sl_atr_mult=args.swing_sl_atr,
        max_hold_days=args.swing_hold_days,
        slippage=args.slippage,
        buy_cost=args.buy_cost,
        sell_cost=args.sell_cost,
        gap_filter_atr=args.swing_gap_filter,
        dd_pause_pct=args.swing_dd_pause_pct,
        dd_pause_days=args.swing_dd_pause_days,
        consec_loss_limit=args.swing_consec_loss_limit,
        consec_loss_pause=args.swing_consec_loss_pause,
        sector_max_pct=args.swing_sector_max_pct,
        gap_aware_sizing=True,
        breadth_regime=True,
    )
    trades_df, equity_df = backtester.run(
        total_score,
        close_df,
        open_df,
        high_df,
        low_df,
        ma_df,
        top_k=args.swing_top_k,
        threshold=args.swing_threshold,
        vol_df=vol_df,
        universe_mask=universe_mask,
    )
    config = {
        "universe_size": args.swing_universe,
        "top_k": args.swing_top_k,
        "threshold": args.swing_threshold,
        "position_size": args.swing_position_size,
        "tp_atr": args.swing_tp_atr,
        "sl_atr": args.swing_sl_atr,
        "hold_days": args.swing_hold_days,
        "ma_period": args.swing_ma_period,
        "rsi_weight": args.swing_rsi_weight,
        "breakout_weight": args.swing_breakout_weight,
        "gap_filter": args.swing_gap_filter,
        "dd_pause_pct": args.swing_dd_pause_pct,
        "dd_pause_days": args.swing_dd_pause_days,
        "consec_loss_limit": args.swing_consec_loss_limit,
        "consec_loss_pause": args.swing_consec_loss_pause,
        "sector_max_pct": args.swing_sector_max_pct,
    }
    latest_date, signals = latest_signals(total_score, close_df, ma_df, universe_mask, config)
    return {
        "trades_df": trades_df,
        "equity_df": equity_df,
        "metrics": compute_risk_metrics(equity_df, trades_df, args.capital),
        "total_score": total_score,
        "ma_df": ma_df,
        "universe_mask": universe_mask,
        "signals": signals,
        "latest_date": latest_date,
        "config": config,
    }


def run_day_overlay(close_df, open_df, high_df, low_df, vol_df, args):
    universe_mask = build_liquid_universe(close_df, vol_df, top_n=args.day_universe)
    total_score, ma_df, _, _ = engineer_features(
        close_df,
        vol_df,
        universe_mask,
        ma_period=args.day_ma_period,
        rsi_weight=args.day_rsi_weight,
        breakout_weight=args.day_breakout_weight,
    )
    config = {
        "initial_capital": args.capital,
        "universe_size": args.day_universe,
        "scan_k": args.day_scan_k,
        "top_k": args.day_top_k,
        "threshold": args.day_threshold,
        "position_size": args.day_position_size,
        "tp_atr": args.day_tp_atr,
        "sl_atr": args.day_sl_atr,
        "gap_min": args.day_gap_min,
        "gap_max": args.day_gap_max,
        "gap_bonus": args.day_gap_bonus,
        "atr_period": args.day_atr_period,
        "rsi_weight": args.day_rsi_weight,
        "breakout_weight": args.day_breakout_weight,
        "ma_period": args.day_ma_period,
        "slippage": args.slippage,
        "buy_cost": args.buy_cost,
        "sell_cost": args.sell_cost,
        "direction": args.day_direction,
        "ambiguous": args.day_ambiguous,
    }
    trades_df, equity_df, atr_df = run_daytrade_backtest(
        total_score,
        close_df,
        open_df,
        high_df,
        low_df,
        ma_df,
        universe_mask,
        config,
    )
    latest_date, signals = latest_daytrade_signals(
        total_score,
        close_df,
        high_df,
        low_df,
        ma_df,
        atr_df,
        universe_mask,
        config,
    )
    return {
        "trades_df": trades_df,
        "equity_df": equity_df,
        "metrics": compute_risk_metrics(equity_df, trades_df, args.capital),
        "signals": signals,
        "latest_date": latest_date,
        "config": config,
    }


def combine_equity(swing, day, args):
    swing_eq = swing["equity_df"]["Equity"]
    day_eq = day["equity_df"]["Equity"]
    common = swing_eq.index.intersection(day_eq.index)
    master_equity = swing_eq.loc[common] + args.day_overlay_scale * (day_eq.loc[common] - args.capital)
    return pd.DataFrame({
        "Equity": master_equity,
        "Swing_Equity": swing_eq.loc[common],
        "Day_Equity": day_eq.loc[common],
        "Day_Overlay_PnL": args.day_overlay_scale * (day_eq.loc[common] - args.capital),
    }, index=common)


def combine_trades(swing, day, args):
    swing_trades = swing["trades_df"].copy()
    if not swing_trades.empty:
        swing_trades["Strategy"] = "Swing"
        swing_trades["Overlay_Scale"] = 1.0
    day_trades = day["trades_df"].copy()
    if not day_trades.empty:
        day_trades["Strategy"] = "DayOverlay"
        day_trades["Overlay_Scale"] = args.day_overlay_scale
        if "PnL" in day_trades.columns:
            day_trades["Scaled_PnL"] = day_trades["PnL"] * args.day_overlay_scale
    return pd.concat([swing_trades, day_trades], ignore_index=True, sort=False)


def gate_from_comparisons(comparisons):
    checks = []
    for comp in comparisons.values():
        checks.extend([
            bool(comp and comp["beats_total"]),
            bool(comp and comp["beats_ann"]),
            bool(comp and comp["beats_sharpe"]),
        ])
    return bool(checks and all(checks))


def run_strategy(args):
    tickers = args.tickers if args.tickers else (DEFAULT_TICKERS if args.static_pool else EXTENDED_TICKERS)

    print("=" * 64)
    print("🚀 HK Master Strategy v1")
    print("=" * 64)
    print(f"   Core: Benchmark Beater Top-{args.swing_universe}, 每筆 {args.swing_position_size*100:.1f}%")
    print(f"   Overlay: Day Trade Top-{args.day_universe}, scale {args.day_overlay_scale:.2f}x")
    print("=" * 64)

    close_df, open_df, high_df, low_df, vol_df = fetch_panel_data(
        tickers,
        days=args.days,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    swing = run_swing_book(close_df, open_df, high_df, low_df, vol_df, args)
    day = run_day_overlay(close_df, open_df, high_df, low_df, vol_df, args)
    equity_df = combine_equity(swing, day, args)
    trades_df = combine_trades(swing, day, args)
    metrics = compute_risk_metrics(equity_df[["Equity"]], trades_df, args.capital)
    print(format_metrics_summary(metrics))

    print("📊 載入 Benchmark...")
    benchmark_equity = fetch_benchmark(DEFAULT_BENCHMARK, days=args.days, start_date=args.start_date, end_date=args.end_date)
    benchmark2_equity = fetch_benchmark(SECONDARY_BENCHMARK, days=args.days, start_date=args.start_date, end_date=args.end_date)
    comparisons = {
        DEFAULT_BENCHMARK_LABEL: compare_with_benchmark(equity_df[["Equity"]], benchmark_equity, DEFAULT_BENCHMARK_LABEL, args.capital),
        SECONDARY_BENCHMARK_LABEL: compare_with_benchmark(equity_df[["Equity"]], benchmark2_equity, SECONDARY_BENCHMARK_LABEL, args.capital),
    }
    gate = {
        "beats_2800_total": bool(comparisons[DEFAULT_BENCHMARK_LABEL] and comparisons[DEFAULT_BENCHMARK_LABEL]["beats_total"]),
        "beats_2800_ann": bool(comparisons[DEFAULT_BENCHMARK_LABEL] and comparisons[DEFAULT_BENCHMARK_LABEL]["beats_ann"]),
        "beats_2800_sharpe": bool(comparisons[DEFAULT_BENCHMARK_LABEL] and comparisons[DEFAULT_BENCHMARK_LABEL]["beats_sharpe"]),
        "beats_2828_total": bool(comparisons[SECONDARY_BENCHMARK_LABEL] and comparisons[SECONDARY_BENCHMARK_LABEL]["beats_total"]),
        "beats_2828_ann": bool(comparisons[SECONDARY_BENCHMARK_LABEL] and comparisons[SECONDARY_BENCHMARK_LABEL]["beats_ann"]),
        "beats_2828_sharpe": bool(comparisons[SECONDARY_BENCHMARK_LABEL] and comparisons[SECONDARY_BENCHMARK_LABEL]["beats_sharpe"]),
    }
    gate["pass"] = bool(all(gate.values()))
    return {
        "swing": swing,
        "day": day,
        "equity_df": equity_df,
        "trades_df": trades_df,
        "metrics": metrics,
        "benchmark_equity": benchmark_equity,
        "benchmark2_equity": benchmark2_equity,
        "comparisons": comparisons,
        "gate": gate,
        "latest_date": max(swing["latest_date"], day["latest_date"]),
        "config": {
            "initial_capital": args.capital,
            "day_overlay_scale": args.day_overlay_scale,
            "slippage": args.slippage,
            "buy_cost": args.buy_cost,
            "sell_cost": args.sell_cost,
            "swing": swing["config"],
            "day": day["config"],
        },
    }


def normalize_equity(series, initial_capital):
    return series / series.iloc[0] * initial_capital


def plot_equity(result, initial_capital):
    equity_df = result["equity_df"]
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), gridspec_kw={"height_ratios": [3, 1]})
    ax1, ax2 = axes
    for ax in axes:
        ax.set_facecolor("#ffffff")

    ax1.plot(equity_df.index, equity_df["Equity"], color="#007aff", lw=2.1, label="Master Strategy")
    ax1.plot(equity_df.index, equity_df["Swing_Equity"], color="#8e8e93", lw=1.2, alpha=0.8, label="Swing Core")
    day_component = initial_capital + equity_df["Day_Overlay_PnL"]
    ax1.plot(equity_df.index, day_component, color="#af52de", lw=1.1, alpha=0.75, label="Day Overlay Component")
    for bench, label, color, style in [
        (result["benchmark_equity"], DEFAULT_BENCHMARK_LABEL, "#ff9500", "--"),
        (result["benchmark2_equity"], SECONDARY_BENCHMARK_LABEL, "#34c759", "-."),
    ]:
        if bench is None or len(bench) == 0:
            continue
        common = equity_df.index.intersection(bench.index)
        if len(common) > 0:
            bench_eq = normalize_equity(bench.loc[common], equity_df.loc[common, "Equity"].iloc[0])
            ax1.plot(common, bench_eq, color=color, linestyle=style, lw=1.4, label=f"{label} Buy & Hold")
    ax1.set_title("HK Master Strategy Equity Curve", fontsize=14, fontweight="bold", color="#1d1d1f")
    ax1.set_ylabel(f"Portfolio Value ({CURRENCY})")
    ax1.grid(alpha=0.08)
    ax1.legend(fontsize=9, loc="upper left")

    dd = equity_df["Equity"] / equity_df["Equity"].cummax() - 1
    ax2.fill_between(dd.index, 0, dd * 100, color="#ff3b30", alpha=0.35)
    ax2.plot(dd.index, dd * 100, color="#ff3b30", lw=1)
    ax2.set_ylabel("Drawdown (%)")
    ax2.grid(alpha=0.08)
    fig.tight_layout()
    fig.savefig("master_strategy_chart.png", dpi=150, bbox_inches="tight", facecolor="#f5f5f7")
    plt.close(fig)


def comparison_cards(comparisons):
    blocks = []
    for label, comp in comparisons.items():
        if not comp:
            continue
        color = "#00ff00" if comp["beats_total"] and comp["beats_ann"] else "#ff4444"
        blocks.append(f"""
        <div class="stats">
            <div class="stat-card benchmark"><div class="label">{label} 區間</div><div class="value small">{comp['start']}<br>{comp['end']}</div></div>
            <div class="stat-card benchmark"><div class="label">{label} 總報酬</div><div class="value">{format_pct(comp['benchmark_total'])}</div></div>
            <div class="stat-card benchmark"><div class="label">{label} 年化</div><div class="value">{format_pct(comp['benchmark_ann'])}</div></div>
            <div class="stat-card benchmark"><div class="label">{label} Sharpe</div><div class="value">{comp['benchmark_sharpe']:.2f}</div></div>
            <div class="stat-card" style="border-left-color:{color}"><div class="label">主策略總超額 vs {label}</div><div class="value" style="color:{color}">{format_pct(comp['total_excess'])}</div></div>
            <div class="stat-card" style="border-left-color:{color}"><div class="label">主策略年化 alpha vs {label}</div><div class="value" style="color:{color}">{format_pct(comp['ann_alpha'])}</div></div>
        </div>""")
    return "\n".join(blocks)


def signal_table(rows, kind):
    body = ""
    if kind == "swing":
        for idx, item in enumerate(rows[:12], 1):
            qualified = item["status"] == "候選"
            color = "#00ff00" if qualified else "#a8b3bd"
            body += (
                f'<tr><td>{idx}</td><td>{item["ticker"]}</td><td>{item["score"]:.2f}</td>'
                f'<td>{item["price"]:.2f}</td><td>{item["ma"]:.2f}</td>'
                f'<td style="color:{color};font-weight:bold;">{item["status"]}</td></tr>'
            )
        header = "<thead><tr><th>#</th><th>股票</th><th>Score</th><th>昨收</th><th>MA</th><th>狀態</th></tr></thead>"
    else:
        for item in rows[:12]:
            active = item["status"].startswith("建議")
            color = "#ffab00" if active else "#a8b3bd"
            plan = "-"
            if active:
                plan = f'{item["ref_entry"]:.2f} / TP {item["tp"]:.2f} / SL {item["sl"]:.2f}'
            body += (
                f'<tr><td>{item["ticker"]}</td><td>{item["score"]:.2f}</td>'
                f'<td>{item["price"]:.2f}</td><td>{item["trigger"]}</td>'
                f'<td style="color:{color};font-weight:bold;">{item["status"]}</td><td>{plan}</td></tr>'
            )
        header = "<thead><tr><th>股票</th><th>Score</th><th>昨收</th><th>觸發</th><th>狀態</th><th>計畫</th></tr></thead>"
    if not body:
        body = '<tr><td colspan="6" style="color:#a8b3bd;">沒有符合條件的標的。</td></tr>'
    return f"<table>{header}<tbody>{body}</tbody></table>"


def generate_html(result):
    metrics = result["metrics"]
    config = result["config"]
    report_date = result["latest_date"].strftime("%Y-%m-%d")
    next_date = next_trading_day(result["latest_date"])
    gate_color = "#00ff00" if result["gate"]["pass"] else "#ff4444"
    total_ret_color = "#00ff00" if metrics["total_return"] > 0 else "#ff4444"
    sharpe_color = "#00ff00" if metrics["sharpe"] > 1 else "#ffab00"
    dd_color = "#ff4444" if metrics["max_drawdown_pct"] < -0.20 else "#ffab00"

    monthly = result["equity_df"]["Equity"].resample("ME").last().pct_change().dropna()
    monthly_rows = ""
    for idx, val in monthly.tail(24).items():
        color = "#00ff00" if val > 0 else "#ff4444"
        monthly_rows += f'<tr><td>{idx.strftime("%Y-%m")}</td><td style="color:{color};font-weight:bold;">{val*100:+.2f}%</td></tr>\n'

    recent_rows = ""
    if not result["trades_df"].empty:
        date_col = "Exit_Date" if "Exit_Date" in result["trades_df"].columns else "Entry_Date"
        recent_source = result["trades_df"].copy()
        recent_source["_Sort_Date"] = pd.to_datetime(recent_source[date_col], errors="coerce")
        recent = recent_source.sort_values("_Sort_Date", ascending=False).head(50)
        for _, tr in recent.iterrows():
            ret = tr.get("Return_Pct", 0)
            color = "#00ff00" if ret > 0 else "#ff4444"
            entry = pd.Timestamp(tr["Entry_Date"]).strftime("%Y-%m-%d") if "Entry_Date" in tr else "-"
            exit_d = pd.Timestamp(tr["Exit_Date"]).strftime("%Y-%m-%d") if "Exit_Date" in tr else entry
            recent_rows += (
                f'<tr><td>{tr.get("Strategy", "-")}</td><td>{entry}</td><td>{exit_d}</td>'
                f'<td>{tr.get("Ticker", "-")}</td><td>{tr.get("Reason", "-")}</td>'
                f'<td style="color:{color};font-weight:bold;">{ret*100:+.2f}%</td></tr>\n'
            )

    html = f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HK Master Strategy 報告 — {report_date}</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:#0b0f14; color:#f5f5f7; line-height:1.55; }}
        .container {{ max-width:1320px; margin:0 auto; padding:28px 18px 56px; }}
        h1 {{ font-size:2rem; margin:0 0 8px; letter-spacing:0; }}
        h2 {{ margin-top:28px; color:#ffffff; border-bottom:1px solid #25313d; padding-bottom:8px; }}
        .subtitle,.section-note {{ color:#a8b3bd; }}
        .badge {{ display:inline-block; margin:4px 6px 4px 0; padding:5px 9px; border-radius:6px; background:#17212b; color:#dbe7f0; border:1px solid #25313d; font-size:.9rem; }}
        .stats {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; margin:14px 0; }}
        .stat-card {{ background:#111820; border:1px solid #25313d; border-left:4px solid #007aff; border-radius:8px; padding:14px; min-height:86px; }}
        .stat-card.benchmark {{ border-left-color:#ff9500; }}
        .label {{ color:#98a6b3; font-size:.82rem; margin-bottom:6px; }}
        .value {{ font-size:1.55rem; font-weight:750; color:#f5f5f7; }}
        .value.small {{ font-size:1rem; line-height:1.4; }}
        table {{ width:100%; border-collapse:collapse; background:#111820; border:1px solid #25313d; border-radius:8px; overflow:hidden; margin-top:12px; }}
        th,td {{ padding:10px 12px; border-bottom:1px solid #25313d; text-align:left; vertical-align:top; }}
        th {{ color:#9fb0bf; background:#17212b; font-size:.85rem; }}
        td {{ color:#edf5fb; }}
        img {{ width:100%; max-width:100%; border-radius:8px; border:1px solid #25313d; background:#fff; }}
        .grid-2 {{ display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr); gap:16px; }}
        .disclaimer {{ margin-top:28px; color:#a8b3bd; background:#111820; border:1px solid #25313d; border-radius:8px; padding:14px; font-size:.92rem; }}
        @media (max-width:800px) {{ .container {{ padding:20px 10px 44px; }} .grid-2 {{ grid-template-columns:1fr; }} table {{ font-size:.86rem; }} th,td {{ padding:8px; }} .value {{ font-size:1.25rem; }} }}
    </style>
</head>
<body>
<div class="container">
    <h1>🚀 HK Master Strategy 報告</h1>
    <p class="subtitle">
        報表日期：{report_date} ｜ 下一交易日：{next_date}
        <br>
        <span class="badge">Gate: <b style="color:{gate_color};">{'PASS' if result['gate']['pass'] else 'FAIL'}</b></span>
        <span class="badge">Core: Benchmark Beater v1.1</span>
        <span class="badge">Overlay: Day Trade v1.1 × {config['day_overlay_scale']:.2f}</span>
        <span class="badge">Swing Universe-{config['swing']['universe_size']}</span>
        <span class="badge">Day Universe-{config['day']['universe_size']}</span>
    </p>

    <h2>績效總覽</h2>
    <div class="stats">
        <div class="stat-card"><div class="label">主策略總報酬率</div><div class="value" style="color:{total_ret_color};">{format_pct(metrics['total_return'])}</div></div>
        <div class="stat-card"><div class="label">年化報酬率</div><div class="value" style="color:{total_ret_color};">{format_pct(metrics['ann_return'])}</div></div>
        <div class="stat-card"><div class="label">Sharpe Ratio</div><div class="value" style="color:{sharpe_color};">{metrics['sharpe']:.2f}</div></div>
        <div class="stat-card"><div class="label">最大回撤</div><div class="value" style="color:{dd_color};">{pct(metrics['max_drawdown_pct']):.1f}%</div></div>
        <div class="stat-card"><div class="label">Calmar</div><div class="value">{metrics['calmar']:.2f}</div></div>
        <div class="stat-card"><div class="label">總交易數</div><div class="value">{metrics['total_trades']}</div></div>
        <div class="stat-card"><div class="label">勝率</div><div class="value">{pct(metrics['win_rate']):.1f}%</div></div>
        <div class="stat-card"><div class="label">Profit Factor</div><div class="value">{metrics['profit_factor']:.2f}</div></div>
    </div>

    <h2>子策略貢獻</h2>
    <div class="stats">
        <div class="stat-card"><div class="label">Swing Core 年化</div><div class="value">{format_pct(result['swing']['metrics']['ann_return'])}</div></div>
        <div class="stat-card"><div class="label">Swing Core Sharpe</div><div class="value">{result['swing']['metrics']['sharpe']:.2f}</div></div>
        <div class="stat-card"><div class="label">Day Trade 年化</div><div class="value">{format_pct(result['day']['metrics']['ann_return'])}</div></div>
        <div class="stat-card"><div class="label">Day Trade Sharpe</div><div class="value">{result['day']['metrics']['sharpe']:.2f}</div></div>
    </div>

    <h2>Benchmark Gate</h2>
    <p class="section-note">Gate 要求主策略在共存期間同時勝過 benchmark 的總報酬、年化報酬與 Sharpe。</p>
    {comparison_cards(result['comparisons'])}

    <h2>資金曲線</h2>
    <img src="master_strategy_chart.png" alt="HK master strategy equity curve">

    <div class="grid-2">
        <div>
            <h2>中線 Watchlist</h2>
            {signal_table(result['swing']['signals'], 'swing')}
        </div>
        <div>
            <h2>日內 Overlay Watchlist</h2>
            {signal_table(result['day']['signals'], 'day')}
        </div>
    </div>

    <div class="grid-2">
        <div>
            <h2>近 24 個月</h2>
            <table><thead><tr><th>月份</th><th>報酬</th></tr></thead><tbody>{monthly_rows}</tbody></table>
        </div>
        <div>
            <h2>近期交易</h2>
            <table><thead><tr><th>策略</th><th>進場</th><th>出場</th><th>股票</th><th>原因</th><th>報酬</th></tr></thead><tbody>{recent_rows}</tbody></table>
        </div>
    </div>

    <div class="disclaimer">
        <b>方法論：</b>Master v1 將 Benchmark Beater v1.1 作為中線核心，並把 Day Trade v1.1 的當日 PnL 以 overlay scale 疊加。
        日內 overlay 不持隔夜倉，但放大倍數代表更高日內成交額與 short/融資需求；此回測未納入借券費、融資利息、實際可 short 名單與券商額外限制。
        本報表僅供研究與技術交流，不構成投資建議。
    </div>
</div>
</body>
</html>
"""
    with open("master_strategy_report.html", "w", encoding="utf-8") as f:
        f.write(html)


def write_artifacts(date_str, result):
    os.makedirs("artifacts", exist_ok=True)
    equity_path = f"artifacts/master_equity_{date_str}.csv"
    trades_path = f"artifacts/master_trades_{date_str}.csv"
    swing_signals_path = f"artifacts/master_swing_signals_{date_str}.csv"
    day_signals_path = f"artifacts/master_day_signals_{date_str}.csv"
    metadata_path = f"artifacts/master_metadata_{date_str}.json"
    gate_path = f"artifacts/master_gate_{date_str}.json"

    result["equity_df"].to_csv(equity_path)
    result["trades_df"].to_csv(trades_path, index=False)
    pd.DataFrame(result["swing"]["signals"]).to_csv(swing_signals_path, index=False)
    pd.DataFrame(result["day"]["signals"]).to_csv(day_signals_path, index=False)

    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        git_sha = None

    metrics = result["metrics"]
    metadata = {
        "created_at": datetime.now().isoformat(),
        "strategy_version": STRATEGY_VERSION,
        "git_sha": git_sha,
        "report_date": date_str,
        "config": result["config"],
        "metrics": {
            "total_return": metrics.get("total_return"),
            "ann_return": metrics.get("ann_return"),
            "sharpe": metrics.get("sharpe"),
            "sortino": metrics.get("sortino"),
            "calmar": metrics.get("calmar"),
            "max_drawdown_pct": metrics.get("max_drawdown_pct"),
            "total_trades": metrics.get("total_trades"),
            "win_rate": metrics.get("win_rate"),
            "profit_factor": metrics.get("profit_factor"),
        },
        "component_metrics": {
            "swing": result["swing"]["metrics"],
            "day": result["day"]["metrics"],
        },
        "comparisons": result["comparisons"],
        "gate": result["gate"],
        "artifacts": {
            "equity": equity_path,
            "trades": trades_path,
            "swing_signals": swing_signals_path,
            "day_signals": day_signals_path,
        },
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
    with open(gate_path, "w", encoding="utf-8") as f:
        json.dump({
            "created_at": metadata["created_at"],
            "strategy_version": STRATEGY_VERSION,
            "gate": result["gate"],
            "comparisons": result["comparisons"],
            "pass": result["gate"]["pass"],
        }, f, indent=2, ensure_ascii=False, default=str)


def parse_args():
    parser = argparse.ArgumentParser(description="HK Master Strategy report")
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument("--static-pool", action="store_true")
    parser.add_argument("--days", type=int, default=1200)
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--capital", type=float, default=200_000)
    parser.add_argument("--slippage", type=float, default=0.001)
    parser.add_argument("--buy-cost", type=float, default=DEFAULT_BUY_COST)
    parser.add_argument("--sell-cost", type=float, default=DEFAULT_SELL_COST)

    parser.add_argument("--day-overlay-scale", type=float, default=8.142)

    parser.add_argument("--swing-universe", type=int, default=50)
    parser.add_argument("--swing-top-k", type=int, default=12)
    parser.add_argument("--swing-threshold", type=float, default=2.0)
    parser.add_argument("--swing-position-size", type=float, default=0.12)
    parser.add_argument("--swing-tp-atr", type=float, default=3.0)
    parser.add_argument("--swing-sl-atr", type=float, default=3.0)
    parser.add_argument("--swing-hold-days", type=int, default=20)
    parser.add_argument("--swing-ma-period", type=int, default=60)
    parser.add_argument("--swing-rsi-weight", type=float, default=1.5)
    parser.add_argument("--swing-breakout-weight", type=float, default=0.0)
    parser.add_argument("--swing-gap-filter", type=float, default=0.0)
    parser.add_argument("--swing-dd-pause-pct", type=float, default=0.10)
    parser.add_argument("--swing-dd-pause-days", type=int, default=5)
    parser.add_argument("--swing-consec-loss-limit", type=int, default=3)
    parser.add_argument("--swing-consec-loss-pause", type=int, default=5)
    parser.add_argument("--swing-sector-max-pct", type=float, default=0.75)

    parser.add_argument("--day-universe", type=int, default=35)
    parser.add_argument("--day-scan-k", type=int, default=4)
    parser.add_argument("--day-top-k", type=int, default=4)
    parser.add_argument("--day-threshold", type=float, default=2.9)
    parser.add_argument("--day-position-size", type=float, default=0.08)
    parser.add_argument("--day-tp-atr", type=float, default=1.3)
    parser.add_argument("--day-sl-atr", type=float, default=0.8)
    parser.add_argument("--day-gap-min", type=float, default=0.03)
    parser.add_argument("--day-gap-max", type=float, default=0.075)
    parser.add_argument("--day-gap-bonus", type=float, default=0.2)
    parser.add_argument("--day-atr-period", type=int, default=20)
    parser.add_argument("--day-rsi-weight", type=float, default=1.0)
    parser.add_argument("--day-breakout-weight", type=float, default=0.0)
    parser.add_argument("--day-ma-period", type=int, default=30)
    parser.add_argument("--day-direction", choices=["long", "short"], default="short")
    parser.add_argument("--day-ambiguous", choices=["stop-first", "profit-first", "close"], default="stop-first")
    return parser.parse_args()


def main():
    args = parse_args()
    result = run_strategy(args)
    date_str = result["latest_date"].strftime("%Y%m%d")
    print("📈 產出 master_strategy_chart.png 與 master_strategy_report.html...")
    plot_equity(result, args.capital)
    generate_html(result)
    write_artifacts(date_str, result)
    if result["gate"]["pass"]:
        print(f"✅ Master gate 通過：主策略勝過 {DEFAULT_BENCHMARK_LABEL} 與 {SECONDARY_BENCHMARK_LABEL}")
    else:
        print("⚠️ Master gate 未通過")
    print("✅ Master strategy 報告已生成：master_strategy_report.html")


if __name__ == "__main__":
    main()
