#!/usr/bin/env python3
"""
AI 港股 Benchmark Beater Day Trade Companion。

這不是取代 Benchmark Beater 中線長倉策略，而是用同一套強勢股分數
產生日內 short gap-up mean reversion 執行頁。
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

from day_trade_report import (
    compute_atr,
    latest_daytrade_signals,
    next_trading_day,
    run_daytrade_backtest,
)
from strategy.ai_strategy import fetch_panel_data, build_liquid_universe, engineer_features
from strategy.benchmark import fetch_benchmark
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


STRATEGY_VERSION = "benchmark-beater-day-trade-v1"


def pct(value):
    return float(value) * 100


def format_pct(value, digits=1):
    return f"{pct(value):+.{digits}f}%"


def compare_with_benchmark(equity_df, benchmark_equity, label):
    if benchmark_equity is None or len(benchmark_equity) <= 20:
        return None
    common = equity_df.index.intersection(benchmark_equity.index)
    if len(common) <= 20:
        return None
    strat = equity_df.loc[common, "Equity"]
    bench = benchmark_equity.loc[common] / benchmark_equity.loc[common].iloc[0] * float(strat.iloc[0])
    strat_m = compute_risk_metrics(pd.DataFrame({"Equity": strat}, index=common), pd.DataFrame(), float(strat.iloc[0]))
    bench_m = compute_risk_metrics(pd.DataFrame({"Equity": bench}, index=common), pd.DataFrame(), float(strat.iloc[0]))
    return {
        "label": label,
        "start": common[0].strftime("%Y-%m-%d"),
        "end": common[-1].strftime("%Y-%m-%d"),
        "strategy_total": float(strat.iloc[-1] / strat.iloc[0] - 1),
        "benchmark_total": float(bench.iloc[-1] / bench.iloc[0] - 1),
        "strategy_ann": float(strat_m["ann_return"]),
        "benchmark_ann": float(bench_m["ann_return"]),
        "strategy_sharpe": float(strat_m["sharpe"]),
        "benchmark_sharpe": float(bench_m["sharpe"]),
        "strategy_mdd": float(strat_m["max_drawdown_pct"]),
        "benchmark_mdd": float(bench_m["max_drawdown_pct"]),
    }


def run_strategy(args):
    tickers = args.tickers if args.tickers else (DEFAULT_TICKERS if args.static_pool else EXTENDED_TICKERS)
    use_dynamic = not args.static_pool and not args.tickers

    print("=" * 64)
    print("⚡ Benchmark Beater Day Trade Companion")
    print("=" * 64)
    print(f"   股池: {'動態 Universe Top-' + str(args.universe_size) if use_dynamic else '靜態 ' + str(len(tickers)) + ' 檔'}")
    print(f"   分數: Mom20×3 + Trend(MA{args.ma_period})×1 + RSI×{args.rsi_weight}")
    print(f"   方向: {args.direction.upper()}  Gap: {args.gap_min*100:+.1f}%~{args.gap_max*100:+.1f}%")
    print(f"   Scan/Top-K: {args.scan_k}/{args.top_k}  TP/SL: ATR×{args.tp_atr}/{args.sl_atr}")
    print("=" * 64)

    close_df, open_df, high_df, low_df, vol_df = fetch_panel_data(
        tickers,
        days=args.days,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    universe_mask = build_liquid_universe(close_df, vol_df, top_n=args.universe_size) if use_dynamic else None
    total_score, ma_df, _, _ = engineer_features(
        close_df,
        vol_df,
        universe_mask,
        ma_period=args.ma_period,
        rsi_weight=args.rsi_weight,
        breakout_weight=args.breakout_weight,
    )
    config = {
        "initial_capital": args.capital,
        "universe_size": args.universe_size,
        "scan_k": args.scan_k,
        "top_k": args.top_k,
        "threshold": args.threshold,
        "position_size": args.position_size,
        "tp_atr": args.tp_atr,
        "sl_atr": args.sl_atr,
        "gap_min": args.gap_min,
        "gap_max": args.gap_max,
        "gap_bonus": args.gap_bonus,
        "atr_period": args.atr_period,
        "rsi_weight": args.rsi_weight,
        "breakout_weight": args.breakout_weight,
        "ma_period": args.ma_period,
        "slippage": args.slippage,
        "buy_cost": args.buy_cost,
        "sell_cost": args.sell_cost,
        "direction": args.direction,
        "ambiguous": args.ambiguous,
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
    metrics = compute_risk_metrics(equity_df, trades_df, args.capital)
    print(format_metrics_summary(metrics))

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
    print("📊 載入 Benchmark...")
    benchmark_equity = fetch_benchmark(DEFAULT_BENCHMARK, days=args.days, start_date=args.start_date, end_date=args.end_date)
    benchmark2_equity = fetch_benchmark(SECONDARY_BENCHMARK, days=args.days, start_date=args.start_date, end_date=args.end_date)
    return {
        "trades_df": trades_df,
        "equity_df": equity_df,
        "metrics": metrics,
        "signals": signals,
        "latest_date": latest_date,
        "config": config,
        "benchmark_equity": benchmark_equity,
        "benchmark2_equity": benchmark2_equity,
        "comparisons": {
            DEFAULT_BENCHMARK_LABEL: compare_with_benchmark(equity_df, benchmark_equity, DEFAULT_BENCHMARK_LABEL),
            SECONDARY_BENCHMARK_LABEL: compare_with_benchmark(equity_df, benchmark2_equity, SECONDARY_BENCHMARK_LABEL),
        },
    }


def plot_equity(result):
    equity_df = result["equity_df"]
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), gridspec_kw={"height_ratios": [3, 1]})
    ax1, ax2 = axes
    for ax in axes:
        ax.set_facecolor("#ffffff")
    ax1.plot(equity_df.index, equity_df["Equity"], color="#007aff", lw=2, label="BB Day Trade")
    ax1.axhline(result["config"]["initial_capital"], color="#8e8e93", linestyle="--", alpha=0.6)
    for bench, label, color, style in [
        (result["benchmark_equity"], DEFAULT_BENCHMARK_LABEL, "#ff9500", "--"),
        (result["benchmark2_equity"], SECONDARY_BENCHMARK_LABEL, "#34c759", "-."),
    ]:
        if bench is None or len(bench) == 0:
            continue
        common = equity_df.index.intersection(bench.index)
        if len(common) > 0:
            bench_eq = bench.loc[common] / bench.loc[common].iloc[0] * equity_df.loc[common, "Equity"].iloc[0]
            ax1.plot(common, bench_eq, color=color, linestyle=style, lw=1.4, label=f"{label} Buy & Hold")
    ax1.set_title("Benchmark Beater Day Trade Companion", fontsize=14, fontweight="bold")
    ax1.set_ylabel(f"Portfolio Value ({CURRENCY})")
    ax1.grid(alpha=0.08)
    ax1.legend(fontsize=9, loc="upper left")

    dd = equity_df["Equity"] / equity_df["Equity"].cummax() - 1
    ax2.fill_between(dd.index, 0, dd * 100, color="#ff3b30", alpha=0.35)
    ax2.plot(dd.index, dd * 100, color="#ff3b30", lw=1)
    ax2.set_ylabel("Drawdown (%)")
    ax2.grid(alpha=0.08)
    fig.tight_layout()
    fig.savefig("benchmark_beater_day_trade_chart.png", dpi=150, bbox_inches="tight", facecolor="#f5f5f7")
    plt.close(fig)


def write_artifacts(date_str, result):
    os.makedirs("artifacts", exist_ok=True)
    equity_path = f"artifacts/bb_day_equity_{date_str}.csv"
    trades_path = f"artifacts/bb_day_trades_{date_str}.csv"
    signals_path = f"artifacts/bb_day_signals_{date_str}.csv"
    metadata_path = f"artifacts/bb_day_metadata_{date_str}.json"
    orders_path = f"artifacts/bb_day_orders_{date_str}.json"

    result["equity_df"].to_csv(equity_path)
    result["trades_df"].to_csv(trades_path, index=False)
    pd.DataFrame(result["signals"]).to_csv(signals_path, index=False)
    with open(orders_path, "w", encoding="utf-8") as f:
        json.dump({"orders": result["signals"]}, f, indent=2, ensure_ascii=False, default=str)

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
        "comparisons": result["comparisons"],
        "artifacts": {
            "equity": equity_path,
            "trades": trades_path,
            "signals": signals_path,
            "orders": orders_path,
        },
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)


def benchmark_cards(comparisons):
    html = []
    for label, comp in comparisons.items():
        if not comp:
            continue
        edge = comp["strategy_total"] - comp["benchmark_total"]
        color = "#00ff00" if edge > 0 else "#ff4444"
        html.append(f"""
        <div class="stats">
            <div class="stat-card benchmark"><div class="label">{label} 總報酬</div><div class="value">{format_pct(comp['benchmark_total'])}</div></div>
            <div class="stat-card benchmark"><div class="label">{label} Sharpe</div><div class="value">{comp['benchmark_sharpe']:.2f}</div></div>
            <div class="stat-card benchmark"><div class="label">{label} 最大回撤</div><div class="value">{pct(comp['benchmark_mdd']):.1f}%</div></div>
            <div class="stat-card" style="border-left-color:{color}"><div class="label">策略總超額 vs {label}</div><div class="value" style="color:{color}">{format_pct(edge)}</div></div>
        </div>""")
    return "\n".join(html)


def generate_html(result):
    metrics = result["metrics"]
    config = result["config"]
    report_date = result["latest_date"].strftime("%Y-%m-%d")
    next_date = next_trading_day(result["latest_date"])
    total_color = "#00ff00" if metrics["total_return"] > 0 else "#ff4444"
    sharpe_color = "#00ff00" if metrics["sharpe"] > 0.7 else ("#ffab00" if metrics["sharpe"] > 0 else "#ff4444")
    dd_color = "#ff4444" if metrics["max_drawdown_pct"] < -0.15 else "#ffab00"

    signal_rows = ""
    for item in result["signals"]:
        active = item["status"].startswith("建議")
        opacity = "1" if active else ("0.65" if "候選" in item["status"] else "0.45")
        status_color = "#ffab00" if active else "#a8b3bd"
        plan = "-"
        if active:
            plan = (
                f'<b>估算進場:</b> {item["ref_entry"]:.2f}<br>'
                f'<b>停利:</b> <span style="color:#00ff00">{item["tp"]:.2f}</span> ({item["tp_pct"]:.1f}%)<br>'
                f'<b>停損:</b> <span style="color:#ff4444">{item["sl"]:.2f}</span> ({item["sl_pct"]:.1f}%)'
            )
        signal_rows += (
            f'<tr style="opacity:{opacity}"><td>{item["ticker"]}</td><td>{item["score"]:.2f}</td>'
            f'<td>{item["price"]:.2f}</td><td>{item["trigger"]}</td>'
            f'<td style="color:{status_color};font-weight:bold;">{item["status"]}</td><td>{plan}</td></tr>\n'
        )
    if not signal_rows:
        signal_rows = '<tr><td colspan="6" style="color:#a8b3bd;">今日沒有符合 watchlist 條件的標的。</td></tr>'

    recent_rows = ""
    if not result["trades_df"].empty:
        recent = result["trades_df"].sort_values("Entry_Date", ascending=False).head(40)
        for _, tr in recent.iterrows():
            color = "#00ff00" if tr["Return_Pct"] > 0 else "#ff4444"
            recent_rows += (
                f'<tr><td>{pd.Timestamp(tr["Entry_Date"]).strftime("%Y-%m-%d")}</td>'
                f'<td>{tr["Ticker"]}</td><td>{tr["Rank"]}</td><td>{tr["Entry_Price"]:.2f}</td>'
                f'<td>{tr["Exit_Price"]:.2f}</td><td>{tr["Reason"]}</td>'
                f'<td style="color:{color};font-weight:bold;">{tr["Return_Pct"]*100:+.2f}%</td></tr>\n'
            )

    monthly = result["equity_df"]["Equity"].resample("ME").last().pct_change().dropna()
    monthly_rows = ""
    for idx, val in monthly.tail(24).items():
        color = "#00ff00" if val > 0 else "#ff4444"
        monthly_rows += f'<tr><td>{idx.strftime("%Y-%m")}</td><td style="color:{color};font-weight:bold;">{val*100:+.2f}%</td></tr>\n'

    html = f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Benchmark Beater Day Trade — {report_date}</title>
    <style>
        * {{ box-sizing:border-box; }}
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
        .value {{ font-size:1.55rem; font-weight:750; }}
        table {{ width:100%; border-collapse:collapse; background:#111820; border:1px solid #25313d; border-radius:8px; overflow:hidden; margin-top:12px; }}
        th,td {{ padding:10px 12px; border-bottom:1px solid #25313d; text-align:left; vertical-align:top; }}
        th {{ color:#9fb0bf; background:#17212b; font-size:.85rem; }}
        td {{ color:#edf5fb; }}
        img {{ width:100%; border-radius:8px; border:1px solid #25313d; background:#fff; }}
        .grid-2 {{ display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr); gap:16px; }}
        .disclaimer {{ margin-top:28px; color:#a8b3bd; background:#111820; border:1px solid #25313d; border-radius:8px; padding:14px; font-size:.92rem; }}
        @media (max-width:800px) {{ .container {{ padding:20px 10px 44px; }} .grid-2 {{ grid-template-columns:1fr; }} table {{ font-size:.86rem; }} th,td {{ padding:8px; }} .value {{ font-size:1.25rem; }} }}
    </style>
</head>
<body>
<div class="container">
    <h1>⚡ Benchmark Beater Day Trade</h1>
    <p class="subtitle">
        報表日期：{report_date} ｜ 下一交易日：{next_date}
        <br>
        <span class="badge">Companion to Benchmark Beater v1</span>
        <span class="badge">{config['direction'].upper()}</span>
        <span class="badge">Universe-{config['universe_size']}</span>
        <span class="badge">MA{config['ma_period']} / RSI×{config['rsi_weight']}</span>
        <span class="badge">Scan-{config['scan_k']} / Top-{config['top_k']}</span>
        <span class="badge">Gap {config['gap_min']*100:+.1f}%~{config['gap_max']*100:+.1f}%</span>
        <span class="badge">ATR×{config['tp_atr']}/{config['sl_atr']}</span>
    </p>

    <h2>績效總覽</h2>
    <div class="stats">
        <div class="stat-card"><div class="label">策略總報酬率</div><div class="value" style="color:{total_color};">{format_pct(metrics['total_return'])}</div></div>
        <div class="stat-card"><div class="label">年化報酬率</div><div class="value" style="color:{total_color};">{format_pct(metrics['ann_return'])}</div></div>
        <div class="stat-card"><div class="label">Sharpe Ratio</div><div class="value" style="color:{sharpe_color};">{metrics['sharpe']:.2f}</div></div>
        <div class="stat-card"><div class="label">最大回撤</div><div class="value" style="color:{dd_color};">{pct(metrics['max_drawdown_pct']):.1f}%</div></div>
        <div class="stat-card"><div class="label">交易數</div><div class="value">{metrics['total_trades']}</div></div>
        <div class="stat-card"><div class="label">勝率</div><div class="value">{pct(metrics['win_rate']):.1f}%</div></div>
        <div class="stat-card"><div class="label">Profit Factor</div><div class="value">{metrics['profit_factor']:.2f}</div></div>
        <div class="stat-card"><div class="label">平均每筆</div><div class="value">{format_pct(metrics['avg_return'], 2)}</div></div>
    </div>

    <h2>Benchmark 參考</h2>
    <p class="section-note">這是低曝險日內 companion，不是用來替代中線 benchmark gate；benchmark 僅作參考。</p>
    {benchmark_cards(result['comparisons'])}

    <h2>下一交易日 Day Trade 執行單</h2>
    <table>
        <thead><tr><th>股票</th><th>Score</th><th>昨收</th><th>開盤觸發</th><th>狀態</th><th>日內計畫</th></tr></thead>
        <tbody>{signal_rows}</tbody>
    </table>

    <h2>資金曲線</h2>
    <img src="benchmark_beater_day_trade_chart.png" alt="Benchmark beater day trade equity curve">

    <div class="grid-2">
        <div>
            <h2>近 24 個月</h2>
            <table><thead><tr><th>月份</th><th>報酬</th></tr></thead><tbody>{monthly_rows}</tbody></table>
        </div>
        <div>
            <h2>近期交易</h2>
            <table><thead><tr><th>日期</th><th>股票</th><th>Rank</th><th>進場</th><th>出場</th><th>原因</th><th>報酬</th></tr></thead><tbody>{recent_rows}</tbody></table>
        </div>
    </div>

    <div class="disclaimer">
        <b>方法論：</b>本頁使用 Benchmark Beater 的強勢股分數，但日內執行採 short gap-up mean reversion。
        測試顯示同向 long day trade 在此資料窗明顯失效，因此沒有採用。Short 版本未納入借券費與可借券性，實盤前必須另行確認。
        本報表僅供研究與技術交流，不構成投資建議。
    </div>
</div>
</body>
</html>"""
    with open("benchmark_beater_day_trade_report.html", "w", encoding="utf-8") as f:
        f.write(html)


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Beater Day Trade Companion")
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument("--static-pool", action="store_true")
    parser.add_argument("--days", type=int, default=1200)
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--capital", type=float, default=200_000)
    parser.add_argument("--universe-size", type=int, default=60)
    parser.add_argument("--scan-k", type=int, default=12)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=2.0)
    parser.add_argument("--position-size", type=float, default=0.08)
    parser.add_argument("--tp-atr", type=float, default=1.0)
    parser.add_argument("--sl-atr", type=float, default=0.75)
    parser.add_argument("--gap-min", type=float, default=0.02)
    parser.add_argument("--gap-max", type=float, default=0.08)
    parser.add_argument("--gap-bonus", type=float, default=0.2)
    parser.add_argument("--atr-period", type=int, default=20)
    parser.add_argument("--rsi-weight", type=float, default=1.5)
    parser.add_argument("--breakout-weight", type=float, default=0.0)
    parser.add_argument("--ma-period", type=int, default=60)
    parser.add_argument("--slippage", type=float, default=0.001)
    parser.add_argument("--buy-cost", type=float, default=DEFAULT_BUY_COST)
    parser.add_argument("--sell-cost", type=float, default=DEFAULT_SELL_COST)
    parser.add_argument("--direction", choices=["long", "short"], default="short")
    parser.add_argument("--ambiguous", choices=["stop-first", "profit-first", "close"], default="stop-first")
    return parser.parse_args()


def main():
    args = parse_args()
    result = run_strategy(args)
    date_str = result["latest_date"].strftime("%Y%m%d")
    print("📈 產出 benchmark_beater_day_trade_chart.png 與 benchmark_beater_day_trade_report.html...")
    plot_equity(result)
    generate_html(result)
    write_artifacts(date_str, result)
    print("✅ Benchmark Beater Day Trade 報告已生成：benchmark_beater_day_trade_report.html")


if __name__ == "__main__":
    main()
