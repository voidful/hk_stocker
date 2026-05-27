#!/usr/bin/env python3
"""
AI 港股 Benchmark Beater 報告頁。

目標：
- 使用中線長倉動量輪動，而不是低曝險 day trade。
- 明確以 2800.HK 與 2828.HK buy-and-hold 作為 gate。
- 只有策略在共存期間勝過 benchmark，報表才標示 gate pass。
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

from strategy.ai_strategy import fetch_panel_data, build_liquid_universe, engineer_features
from strategy.benchmark import fetch_benchmark, equal_weight_benchmark
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


STRATEGY_VERSION = "benchmark-beater-v1.1"


def pct(value):
    return float(value) * 100


def format_pct(value, digits=1):
    return f"{pct(value):+.{digits}f}%"


def normalize_equity(series, initial_capital):
    return series / series.iloc[0] * initial_capital


def compare_with_benchmark(equity_df, benchmark_equity, label, initial_capital):
    if benchmark_equity is None or len(benchmark_equity) <= 20:
        return None
    common = equity_df.index.intersection(benchmark_equity.index)
    if len(common) <= 20:
        return None

    strat = equity_df.loc[common, "Equity"]
    bench = normalize_equity(benchmark_equity.loc[common], float(strat.iloc[0]))
    strat_metrics = compute_risk_metrics(
        pd.DataFrame({"Equity": strat}, index=common),
        pd.DataFrame(),
        float(strat.iloc[0]),
    )
    bench_metrics = compute_risk_metrics(
        pd.DataFrame({"Equity": bench}, index=common),
        pd.DataFrame(),
        float(strat.iloc[0]),
    )
    strat_total = strat.iloc[-1] / strat.iloc[0] - 1
    bench_total = bench.iloc[-1] / bench.iloc[0] - 1
    return {
        "label": label,
        "start": common[0].strftime("%Y-%m-%d"),
        "end": common[-1].strftime("%Y-%m-%d"),
        "strategy_total": float(strat_total),
        "benchmark_total": float(bench_total),
        "total_excess": float(strat_total - bench_total),
        "strategy_ann": float(strat_metrics["ann_return"]),
        "benchmark_ann": float(bench_metrics["ann_return"]),
        "ann_alpha": float(strat_metrics["ann_return"] - bench_metrics["ann_return"]),
        "strategy_sharpe": float(strat_metrics["sharpe"]),
        "benchmark_sharpe": float(bench_metrics["sharpe"]),
        "strategy_mdd": float(strat_metrics["max_drawdown_pct"]),
        "benchmark_mdd": float(bench_metrics["max_drawdown_pct"]),
        "beats_total": bool(strat_total > bench_total),
        "beats_ann": bool(strat_metrics["ann_return"] > bench_metrics["ann_return"]),
        "beats_sharpe": bool(strat_metrics["sharpe"] > bench_metrics["sharpe"]),
    }


def latest_signals(total_score, close_df, ma_df, universe_mask, config, limit=20):
    latest_date = total_score.index[-1]
    rows = []
    scores = total_score.loc[latest_date].dropna().sort_values(ascending=False)
    for ticker, score in scores.items():
        if len(rows) >= limit:
            break
        if ticker not in close_df.columns or ticker not in ma_df.columns:
            continue
        if universe_mask is not None:
            try:
                if not bool(universe_mask.loc[latest_date, ticker]):
                    continue
            except Exception:
                continue
        price = close_df[ticker].dropna().iloc[-1] if not close_df[ticker].dropna().empty else None
        ma = ma_df[ticker].dropna().iloc[-1] if not ma_df[ticker].dropna().empty else None
        if price is None or ma is None or price <= 0:
            continue
        above_ma = price > ma
        qualified = float(score) >= config["threshold"] and above_ma
        rows.append({
            "ticker": ticker,
            "score": float(score),
            "price": float(price),
            "ma": float(ma),
            "above_ma": bool(above_ma),
            "status": "候選" if qualified else ("低於門檻" if float(score) < config["threshold"] else "低於 MA"),
        })
    return latest_date, rows


def run_strategy(args):
    tickers = args.tickers if args.tickers else (DEFAULT_TICKERS if args.static_pool else EXTENDED_TICKERS)
    use_dynamic = not args.static_pool and not args.tickers

    print("=" * 64)
    print("🏆 AI 港股 Benchmark Beater v1.1")
    print("=" * 64)
    print(f"   股池: {'動態 Universe Top-' + str(args.universe_size) if use_dynamic else '靜態 ' + str(len(tickers)) + ' 檔'}")
    print(f"   目標: 打贏 {DEFAULT_BENCHMARK_LABEL} / {SECONDARY_BENCHMARK_LABEL} Buy & Hold")
    print(f"   選股: Mom20×3 + Trend(MA{args.ma_period})×1 + RSI×{args.rsi_weight}")
    print(f"   Top-K: {args.top_k}  Threshold: {args.threshold}  Hold: {args.hold_days}D")
    print(f"   TP/SL: ATR×{args.tp_atr}/{args.sl_atr}  每筆: {args.position_size*100:.1f}%")
    print("=" * 64)

    close_df, open_df, high_df, low_df, vol_df = fetch_panel_data(
        tickers,
        days=args.days,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    universe_mask = build_liquid_universe(close_df, vol_df, top_n=args.universe_size) if use_dynamic else None

    total_score, ma_df, _atr_df, _ = engineer_features(
        close_df,
        vol_df,
        universe_mask,
        ma_period=args.ma_period,
        rsi_weight=args.rsi_weight,
        breakout_weight=args.breakout_weight,
    )

    backtester = EventDrivenBacktester(
        initial_capital=args.capital,
        position_size=args.position_size,
        tp_sl_mode="atr",
        tp_atr_mult=args.tp_atr,
        sl_atr_mult=args.sl_atr,
        max_hold_days=args.hold_days,
        slippage=args.slippage,
        buy_cost=args.buy_cost,
        sell_cost=args.sell_cost,
        gap_filter_atr=args.gap_filter,
        regime_filter=False,
        dd_pause_pct=args.dd_pause_pct,
        dd_pause_days=args.dd_pause_days,
        consec_loss_limit=args.consec_loss_limit,
        consec_loss_pause=args.consec_loss_pause,
        sector_max_pct=args.sector_max_pct,
        gap_aware_sizing=True,
        breadth_regime=True,
    )
    # Important: let EventDrivenBacktester compute True Range ATR from OHLC.
    trades_df, equity_df = backtester.run(
        total_score,
        close_df,
        open_df,
        high_df,
        low_df,
        ma_df,
        top_k=args.top_k,
        threshold=args.threshold,
        vol_df=vol_df,
        universe_mask=universe_mask,
    )

    metrics = compute_risk_metrics(equity_df, trades_df, args.capital)
    print(format_metrics_summary(metrics))

    print("📊 載入 Benchmark...")
    benchmark_equity = fetch_benchmark(DEFAULT_BENCHMARK, days=args.days, start_date=args.start_date, end_date=args.end_date)
    benchmark2_equity = fetch_benchmark(SECONDARY_BENCHMARK, days=args.days, start_date=args.start_date, end_date=args.end_date)
    ew_equity = equal_weight_benchmark(close_df)

    config = {
        "initial_capital": args.capital,
        "universe_size": args.universe_size,
        "top_k": args.top_k,
        "threshold": args.threshold,
        "position_size": args.position_size,
        "tp_atr": args.tp_atr,
        "sl_atr": args.sl_atr,
        "hold_days": args.hold_days,
        "ma_period": args.ma_period,
        "rsi_weight": args.rsi_weight,
        "breakout_weight": args.breakout_weight,
        "gap_filter": args.gap_filter,
        "slippage": args.slippage,
        "buy_cost": args.buy_cost,
        "sell_cost": args.sell_cost,
        "dd_pause_pct": args.dd_pause_pct,
        "dd_pause_days": args.dd_pause_days,
        "consec_loss_limit": args.consec_loss_limit,
        "consec_loss_pause": args.consec_loss_pause,
        "sector_max_pct": args.sector_max_pct,
        "gap_aware_sizing": True,
        "breadth_regime": True,
    }
    comparisons = {
        DEFAULT_BENCHMARK_LABEL: compare_with_benchmark(equity_df, benchmark_equity, DEFAULT_BENCHMARK_LABEL, args.capital),
        SECONDARY_BENCHMARK_LABEL: compare_with_benchmark(equity_df, benchmark2_equity, SECONDARY_BENCHMARK_LABEL, args.capital),
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
    latest_date, signals = latest_signals(total_score, close_df, ma_df, universe_mask, config)
    return {
        "trades_df": trades_df,
        "equity_df": equity_df,
        "metrics": metrics,
        "benchmark_equity": benchmark_equity,
        "benchmark2_equity": benchmark2_equity,
        "ew_equity": ew_equity,
        "comparisons": comparisons,
        "gate": gate,
        "latest_date": latest_date,
        "signals": signals,
        "config": config,
    }


def plot_equity(equity_df, benchmark_equity, benchmark2_equity, initial_capital):
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), gridspec_kw={"height_ratios": [3, 1]})
    ax1, ax2 = axes
    for ax in axes:
        ax.set_facecolor("#ffffff")

    ax1.plot(equity_df.index, equity_df["Equity"], color="#007aff", lw=2.0, label="Benchmark Beater")
    if benchmark_equity is not None and len(benchmark_equity) > 0:
        common = equity_df.index.intersection(benchmark_equity.index)
        if len(common) > 0:
            bench = normalize_equity(benchmark_equity.loc[common], initial_capital)
            ax1.plot(common, bench, color="#ff9500", lw=1.5, linestyle="--", label=f"{DEFAULT_BENCHMARK_LABEL} Buy & Hold")
    if benchmark2_equity is not None and len(benchmark2_equity) > 0:
        common = equity_df.index.intersection(benchmark2_equity.index)
        if len(common) > 0:
            bench2 = normalize_equity(benchmark2_equity.loc[common], equity_df.loc[common, "Equity"].iloc[0])
            ax1.plot(common, bench2, color="#34c759", lw=1.4, linestyle="-.", label=f"{SECONDARY_BENCHMARK_LABEL} Buy & Hold")

    ax1.set_title("HK Benchmark Beater Equity Curve", fontsize=14, fontweight="bold", color="#1d1d1f")
    ax1.set_ylabel(f"Portfolio Value ({CURRENCY})", fontsize=11)
    ax1.grid(alpha=0.08, color="#000000")
    ax1.legend(fontsize=9, loc="upper left")

    equity = equity_df["Equity"]
    drawdown = equity / equity.cummax() - 1
    ax2.fill_between(drawdown.index, 0, drawdown * 100, color="#ff3b30", alpha=0.35)
    ax2.plot(drawdown.index, drawdown * 100, color="#ff3b30", lw=1)
    ax2.set_ylabel("Drawdown (%)", fontsize=10)
    ax2.grid(alpha=0.08, color="#000000")
    fig.tight_layout()
    fig.savefig("benchmark_beater_chart.png", dpi=150, bbox_inches="tight", facecolor="#f5f5f7")
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
            <div class="stat-card" style="border-left-color:{color}"><div class="label">策略總超額 vs {label}</div><div class="value" style="color:{color}">{format_pct(comp['total_excess'])}</div></div>
            <div class="stat-card" style="border-left-color:{color}"><div class="label">策略年化 alpha vs {label}</div><div class="value" style="color:{color}">{format_pct(comp['ann_alpha'])}</div></div>
        </div>""")
    return "\n".join(blocks)


def write_artifacts(date_str, result):
    os.makedirs("artifacts", exist_ok=True)
    equity_path = f"artifacts/bb_equity_{date_str}.csv"
    trades_path = f"artifacts/bb_trades_{date_str}.csv"
    signals_path = f"artifacts/bb_signals_{date_str}.csv"
    metadata_path = f"artifacts/bb_metadata_{date_str}.json"
    gate_path = f"artifacts/bb_gate_{date_str}.json"

    result["equity_df"].to_csv(equity_path)
    result["trades_df"].to_csv(trades_path, index=False)
    pd.DataFrame(result["signals"]).to_csv(signals_path, index=False)

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
        "gate": result["gate"],
        "artifacts": {
            "equity": equity_path,
            "trades": trades_path,
            "signals": signals_path,
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


def generate_html(result):
    metrics = result["metrics"]
    config = result["config"]
    latest_date = result["latest_date"]
    report_date = latest_date.strftime("%Y-%m-%d")
    gate_color = "#00ff00" if result["gate"]["pass"] else "#ff4444"
    total_ret_color = "#00ff00" if metrics["total_return"] > 0 else "#ff4444"
    sharpe_color = "#00ff00" if metrics["sharpe"] > 1 else "#ffab00"
    dd_color = "#ff4444" if metrics["max_drawdown_pct"] < -0.20 else "#ffab00"

    signal_rows = ""
    for idx, item in enumerate(result["signals"][:15], 1):
        qualified = item["status"] == "候選"
        opacity = "1" if qualified else "0.5"
        status_color = "#00ff00" if qualified else "#a8b3bd"
        signal_rows += (
            f'<tr style="opacity:{opacity}"><td>{idx}</td><td>{item["ticker"]}</td>'
            f'<td>{item["score"]:.2f}</td><td>{item["price"]:.2f}</td>'
            f'<td>{item["ma"]:.2f}</td><td style="color:{status_color};font-weight:bold;">{item["status"]}</td></tr>\n'
        )

    recent_rows = ""
    if not result["trades_df"].empty:
        recent = result["trades_df"].sort_values("Exit_Date", ascending=False).head(40)
        for _, tr in recent.iterrows():
            ret = tr["Return_Pct"]
            color = "#00ff00" if ret > 0 else "#ff4444"
            recent_rows += (
                f'<tr><td>{tr["Entry_Date"]}</td><td>{tr["Exit_Date"]}</td><td>{tr["Ticker"]}</td>'
                f'<td>{tr["Entry_Price"]:.2f}</td><td>{tr["Exit_Price"]:.2f}</td>'
                f'<td>{tr["Reason"]}</td><td>{tr["Days_Held"]}</td>'
                f'<td style="color:{color};font-weight:bold;">{ret*100:+.2f}%</td></tr>\n'
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
    <title>AI 港股 Benchmark Beater 報告 — {report_date}</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:#0b0f14; color:#f5f5f7; line-height:1.55; }}
        .container {{ max-width:1320px; margin:0 auto; padding:28px 18px 56px; }}
        h1 {{ font-size:2rem; margin:0 0 8px; letter-spacing:0; }}
        h2 {{ margin-top:28px; color:#ffffff; border-bottom:1px solid #25313d; padding-bottom:8px; }}
        .subtitle {{ color:#a8b3bd; margin-bottom:18px; }}
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
        .section-note {{ color:#a8b3bd; margin:8px 0 12px; }}
        .disclaimer {{ margin-top:28px; color:#a8b3bd; background:#111820; border:1px solid #25313d; border-radius:8px; padding:14px; font-size:.92rem; }}
        @media (max-width:800px) {{ .container {{ padding:20px 10px 44px; }} .grid-2 {{ grid-template-columns:1fr; }} table {{ font-size:.86rem; }} th,td {{ padding:8px; }} .value {{ font-size:1.25rem; }} }}
    </style>
</head>
<body>
<div class="container">
    <h1>🏆 AI 港股 Benchmark Beater 報告</h1>
    <p class="subtitle">
        報表日期：{report_date}
        <br>
        <span class="badge">Gate: <b style="color:{gate_color};">{'PASS' if result['gate']['pass'] else 'FAIL'}</b></span>
        <span class="badge">Universe-{config['universe_size']}</span>
        <span class="badge">Top-{config['top_k']}</span>
        <span class="badge">Mom20×3 + Trend×1 + RSI×{config['rsi_weight']}</span>
        <span class="badge">ATR×{config['tp_atr']}/{config['sl_atr']}</span>
        <span class="badge">Hold {config['hold_days']}D</span>
        <span class="badge">每筆 {config['position_size']*100:.0f}%</span>
    </p>

    <h2>績效總覽</h2>
    <div class="stats">
        <div class="stat-card"><div class="label">策略總報酬率</div><div class="value" style="color:{total_ret_color};">{format_pct(metrics['total_return'])}</div></div>
        <div class="stat-card"><div class="label">年化報酬率</div><div class="value" style="color:{total_ret_color};">{format_pct(metrics['ann_return'])}</div></div>
        <div class="stat-card"><div class="label">Sharpe Ratio</div><div class="value" style="color:{sharpe_color};">{metrics['sharpe']:.2f}</div></div>
        <div class="stat-card"><div class="label">最大回撤</div><div class="value" style="color:{dd_color};">{pct(metrics['max_drawdown_pct']):.1f}%</div></div>
        <div class="stat-card"><div class="label">Calmar</div><div class="value">{metrics['calmar']:.2f}</div></div>
        <div class="stat-card"><div class="label">完成交易數</div><div class="value">{metrics['total_trades']}</div></div>
        <div class="stat-card"><div class="label">勝率</div><div class="value">{pct(metrics['win_rate']):.1f}%</div></div>
        <div class="stat-card"><div class="label">Profit Factor</div><div class="value">{metrics['profit_factor']:.2f}</div></div>
    </div>

    <h2>Benchmark Gate</h2>
    <p class="section-note">Gate 要求策略在共存期間同時勝過 benchmark 的總報酬、年化報酬與 Sharpe。</p>
    {comparison_cards(result['comparisons'])}

    <h2>資金曲線</h2>
    <img src="benchmark_beater_chart.png" alt="HK benchmark beater equity curve">

    <div class="grid-2">
        <div>
            <h2>下一交易日 Watchlist</h2>
            <table>
                <thead><tr><th>#</th><th>股票</th><th>Score</th><th>昨收</th><th>MA</th><th>狀態</th></tr></thead>
                <tbody>{signal_rows}</tbody>
            </table>
        </div>
        <div>
            <h2>近 24 個月</h2>
            <table>
                <thead><tr><th>月份</th><th>報酬</th></tr></thead>
                <tbody>{monthly_rows}</tbody>
            </table>
        </div>
    </div>

    <h2>近期交易</h2>
    <table>
        <thead><tr><th>進場</th><th>出場</th><th>股票</th><th>進場價</th><th>出場價</th><th>原因</th><th>天數</th><th>報酬</th></tr></thead>
        <tbody>{recent_rows}</tbody>
    </table>

    <div class="disclaimer">
        <b>方法論：</b>本策略是中線長倉 cross-sectional momentum，使用 t-1 收盤信號、t open 進場、True Range ATR TP/SL、
        港股交易成本與 10bps 滑價。它是為了取得足夠市場曝險來打贏 2800/2828 buy-and-hold；
        不同於 day trade 的低曝險 short mean reversion。本報表僅供研究與技術交流，不構成投資建議。
    </div>
</div>
</body>
</html>
"""
    with open("benchmark_beater_report.html", "w", encoding="utf-8") as f:
        f.write(html)


def parse_args():
    parser = argparse.ArgumentParser(description="AI 港股 Benchmark Beater 報告")
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument("--static-pool", action="store_true")
    parser.add_argument("--days", type=int, default=1200)
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--capital", type=float, default=200_000)
    parser.add_argument("--universe-size", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--threshold", type=float, default=2.0)
    parser.add_argument("--position-size", type=float, default=0.12)
    parser.add_argument("--tp-atr", type=float, default=3.0)
    parser.add_argument("--sl-atr", type=float, default=3.0)
    parser.add_argument("--hold-days", type=int, default=20)
    parser.add_argument("--ma-period", type=int, default=60)
    parser.add_argument("--rsi-weight", type=float, default=1.5)
    parser.add_argument("--breakout-weight", type=float, default=0.0)
    parser.add_argument("--gap-filter", type=float, default=0.0)
    parser.add_argument("--slippage", type=float, default=0.001)
    parser.add_argument("--buy-cost", type=float, default=DEFAULT_BUY_COST)
    parser.add_argument("--sell-cost", type=float, default=DEFAULT_SELL_COST)
    parser.add_argument("--dd-pause-pct", type=float, default=0.10)
    parser.add_argument("--dd-pause-days", type=int, default=5)
    parser.add_argument("--consec-loss-limit", type=int, default=3)
    parser.add_argument("--consec-loss-pause", type=int, default=5)
    parser.add_argument("--sector-max-pct", type=float, default=0.75)
    return parser.parse_args()


def main():
    args = parse_args()
    result = run_strategy(args)
    date_str = result["latest_date"].strftime("%Y%m%d")
    print("📈 產出 benchmark_beater_chart.png 與 benchmark_beater_report.html...")
    plot_equity(result["equity_df"], result["benchmark_equity"], result["benchmark2_equity"], args.capital)
    generate_html(result)
    write_artifacts(date_str, result)
    if result["gate"]["pass"]:
        print(f"✅ Benchmark gate 通過：策略勝過 {DEFAULT_BENCHMARK_LABEL} 與 {SECONDARY_BENCHMARK_LABEL}")
    else:
        print("⚠️ Benchmark gate 未通過，請勿升級為 benchmark-beater")
    print("✅ Benchmark beater 報告已生成：benchmark_beater_report.html")


if __name__ == "__main__":
    main()
