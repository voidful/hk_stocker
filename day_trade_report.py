#!/usr/bin/env python3
"""
AI 港股 Day Trade 報告頁。

定義：
- t-1 收盤後產生信號
- t 日開盤進場
- t 日盤中觸發 TP/SL，否則收盤強制出場

日內觸價使用日線 High/Low 近似；若同日同時碰到 TP 與 SL，預設採保守的 stop-first。
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strategy.ai_strategy import fetch_panel_data, build_liquid_universe, engineer_features
from strategy.benchmark import fetch_benchmark, equal_weight_benchmark
from strategy.market import (
    CURRENCY,
    DEFAULT_BENCHMARK,
    DEFAULT_BENCHMARK_LABEL,
    DEFAULT_BUY_COST,
    DEFAULT_SELL_COST,
    EXCHANGE_CALENDAR,
    SECONDARY_BENCHMARK,
    SECONDARY_BENCHMARK_LABEL,
)
from strategy.risk_metrics import compute_risk_metrics, format_metrics_summary
from strategy.universe import DEFAULT_TICKERS, EXTENDED_TICKERS

try:
    import exchange_calendars as xcals
    HK_CALENDAR = xcals.get_calendar(EXCHANGE_CALENDAR)
    HAS_EXCHANGE_CAL = True
except ImportError:
    HK_CALENDAR = None
    HAS_EXCHANGE_CAL = False


def next_trading_day(from_date):
    if HAS_EXCHANGE_CAL and HK_CALENDAR is not None:
        try:
            from_ts = pd.Timestamp(from_date)
            sessions = HK_CALENDAR.sessions_in_range(
                from_ts + pd.Timedelta(days=1),
                from_ts + pd.Timedelta(days=14),
            )
            if len(sessions) > 0:
                return sessions[0].strftime("%Y-%m-%d")
        except Exception:
            pass
    return (pd.Timestamp(from_date) + timedelta(days=1)).strftime("%Y-%m-%d")


def compute_atr(high_df, low_df, close_df, period=20):
    prev_close = close_df.shift(1)
    tr1 = high_df - low_df
    tr2 = (high_df - prev_close).abs()
    tr3 = (low_df - prev_close).abs()
    true_range = pd.concat(
        [tr1.stack(), tr2.stack(), tr3.stack()],
        axis=1,
    ).max(axis=1).unstack()
    true_range = true_range.reindex(index=close_df.index, columns=close_df.columns)
    return true_range.rolling(period).mean()


def is_tradable(open_v, high_v, low_v, close_v):
    values = (open_v, high_v, low_v, close_v)
    return all(not pd.isna(v) and v > 0 for v in values)


def run_daytrade_backtest(
    total_score,
    close_df,
    open_df,
    high_df,
    low_df,
    ma_60,
    universe_mask,
    config,
):
    initial_capital = config["initial_capital"]
    capital = initial_capital
    equity_rows = []
    trades = []
    atr_df = compute_atr(high_df, low_df, close_df, period=config["atr_period"])
    dates = close_df.index

    for i in range(1, len(dates)):
        date = dates[i]
        signal_date = dates[i - 1]
        day_start_capital = capital
        cash = capital

        day_scores = total_score.iloc[i - 1].dropna().sort_values(ascending=False)
        candidates = []
        for ticker, score in day_scores.items():
            if len(candidates) >= config["scan_k"]:
                break
            if score < config["threshold"]:
                continue
            if ticker not in close_df.columns:
                continue
            if universe_mask is not None:
                try:
                    if not bool(universe_mask[ticker].iloc[i - 1]):
                        continue
                except Exception:
                    continue

            prev_close = close_df[ticker].iloc[i - 1]
            ma = ma_60[ticker].iloc[i - 1]
            open_v = open_df[ticker].iloc[i]
            high_v = high_df[ticker].iloc[i]
            low_v = low_df[ticker].iloc[i]
            close_v = close_df[ticker].iloc[i]
            if pd.isna(prev_close) or pd.isna(ma) or prev_close <= ma:
                continue
            if not is_tradable(open_v, high_v, low_v, close_v):
                continue
            gap = open_v / prev_close - 1
            if gap < config["gap_min"] or gap > config["gap_max"]:
                continue
            adjusted_score = float(score) + config["gap_bonus"] * float(gap * 100)
            candidates.append((ticker, adjusted_score, float(score), float(gap)))

        selected = sorted(candidates, key=lambda x: x[1], reverse=True)[:config["top_k"]]

        for rank, (ticker, adjusted_score, score, gap) in enumerate(selected, 1):
            open_v = float(open_df[ticker].iloc[i])
            high_v = float(high_df[ticker].iloc[i])
            low_v = float(low_df[ticker].iloc[i])
            close_v = float(close_df[ticker].iloc[i])
            atr_v = atr_df[ticker].iloc[i - 1] if ticker in atr_df.columns else np.nan
            if pd.isna(atr_v) or atr_v <= 0:
                continue

            if config["direction"] == "short":
                entry_price = open_v * (1 - config["slippage"])
            else:
                entry_price = open_v * (1 + config["slippage"])
            trade_amount = day_start_capital * config["position_size"]
            buy_value = trade_amount * (1 + config["buy_cost"])
            if cash < buy_value:
                continue

            if config["direction"] == "short":
                tp_price = max(0.01, entry_price - float(atr_v) * config["tp_atr"])
                sl_price = entry_price + float(atr_v) * config["sl_atr"]
                hit_tp = low_v <= tp_price
                hit_sl = high_v >= sl_price
            else:
                tp_price = entry_price + float(atr_v) * config["tp_atr"]
                sl_price = max(0.01, entry_price - float(atr_v) * config["sl_atr"])
                hit_tp = high_v >= tp_price
                hit_sl = low_v <= sl_price

            if hit_tp and hit_sl:
                if config["ambiguous"] == "profit-first":
                    raw_exit = tp_price
                    reason = "TP/SL same day (TP first)"
                elif config["ambiguous"] == "close":
                    raw_exit = close_v
                    reason = "TP/SL same day (Close)"
                else:
                    raw_exit = sl_price
                    reason = "TP/SL same day (SL first)"
            elif hit_tp:
                raw_exit = tp_price
                reason = "Take Profit"
            elif hit_sl:
                raw_exit = sl_price
                reason = "Stop Loss"
            else:
                raw_exit = close_v
                reason = "Day Close"

            if config["direction"] == "short":
                exit_price = raw_exit * (1 + config["slippage"])
            else:
                exit_price = raw_exit * (1 - config["slippage"])
            shares = trade_amount / entry_price
            if config["direction"] == "short":
                entry_cost = trade_amount * (config["sell_cost"] + config["slippage"])
                cover_cost = trade_amount * (config["buy_cost"] + config["slippage"])
                ret_pct = (entry_price - exit_price) / entry_price - config["buy_cost"] - config["sell_cost"]
                pnl = trade_amount * ret_pct
                cash += pnl
            else:
                cash -= buy_value
                revenue = shares * exit_price * (1 - config["sell_cost"])
                pnl = revenue - buy_value
                cash += revenue
                ret_pct = pnl / buy_value if buy_value > 0 else 0

            trades.append({
                "Entry_Date": date,
                "Exit_Date": date,
                "Signal_Date": signal_date,
                "Ticker": ticker,
                "Rank": rank,
                "Score": score,
                "Adjusted_Score": adjusted_score,
                "Gap_Pct": gap,
                "Direction": config["direction"],
                "Entry_Price": entry_price,
                "Exit_Price": exit_price,
                "TP_Price": tp_price,
                "SL_Price": sl_price,
                "Shares": shares,
                "PnL": pnl,
                "Return_Pct": ret_pct,
                "Days_Held": 1,
                "Reason": reason,
            })

        capital = cash
        equity_rows.append({"Date": date, "Equity": capital})

    equity_df = pd.DataFrame(equity_rows).set_index("Date")
    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        for col in ("Entry_Date", "Exit_Date", "Signal_Date"):
            trades_df[col] = pd.to_datetime(trades_df[col])
    return trades_df, equity_df, atr_df


def latest_daytrade_signals(total_score, close_df, high_df, low_df, ma_60, atr_df, universe_mask, config):
    latest_date = total_score.index[-1]
    day_scores = total_score.loc[latest_date].dropna().sort_values(ascending=False)
    rows = []
    candidates = []

    for ticker, score in day_scores.items():
        if ticker not in close_df.columns:
            continue
        if universe_mask is not None:
            try:
                if not bool(universe_mask.loc[latest_date, ticker]):
                    continue
            except Exception:
                continue
        price_s = close_df[ticker].dropna()
        ma_s = ma_60[ticker].dropna()
        atr_s = atr_df[ticker].dropna() if ticker in atr_df.columns else pd.Series(dtype=float)
        price = price_s.iloc[-1] if not price_s.empty else np.nan
        ma = ma_s.iloc[-1] if not ma_s.empty else np.nan
        atr = atr_s.iloc[-1] if not atr_s.empty else np.nan
        if pd.isna(price) or pd.isna(ma) or pd.isna(atr) or price <= 0 or atr <= 0:
            continue
        if score < config["threshold"]:
            continue
        if price <= ma:
            candidates.append((ticker, float(score), float(price), float(atr), f"低於 MA{config['ma_period']}"))
            continue
        candidates.append((ticker, float(score), float(price), float(atr), "候選"))

    tradable = [x for x in candidates if x[4] == "候選"][:config["scan_k"]]
    selected = tradable[:config["top_k"]]
    overflow = tradable[config["top_k"]:config["top_k"] + 5]
    filtered = [x for x in candidates if x[4] != "候選"][:5]

    for rank, (ticker, score, price, atr, status) in enumerate(selected, 1):
        ref_entry = price * (1 + (config["gap_min"] + config["gap_max"]) / 2)
        if config["direction"] == "short":
            tp = max(0.01, ref_entry - atr * config["tp_atr"])
            sl = ref_entry + atr * config["sl_atr"]
            action = "建議 short day trade"
        else:
            tp = ref_entry + atr * config["tp_atr"]
            sl = max(0.01, ref_entry - atr * config["sl_atr"])
            action = "建議 long day trade"
        rows.append({
            "ticker": ticker,
            "rank": rank,
            "score": score,
            "price": price,
            "ref_entry": ref_entry,
            "tp": tp,
            "sl": sl,
            "tp_pct": abs(tp / ref_entry - 1) * 100,
            "sl_pct": abs(sl / ref_entry - 1) * 100,
            "status": action,
            "trigger": f"開盤 gap {config['gap_min']*100:+.1f}% ~ {config['gap_max']*100:+.1f}%",
        })

    for ticker, score, price, atr, _ in overflow:
        rows.append({
            "ticker": ticker,
            "rank": None,
            "score": score,
            "price": price,
            "ref_entry": None,
            "tp": None,
            "sl": None,
            "tp_pct": None,
            "sl_pct": None,
            "status": "候選（超出 Top-K）",
            "trigger": f"開盤 gap {config['gap_min']*100:+.1f}% ~ {config['gap_max']*100:+.1f}%",
        })

    for ticker, score, price, atr, status in filtered:
        rows.append({
            "ticker": ticker,
            "rank": None,
            "score": score,
            "price": price,
            "ref_entry": None,
            "tp": None,
            "sl": None,
            "tp_pct": None,
            "sl_pct": None,
            "status": f"觀望（{status}）",
            "trigger": "-",
        })

    return latest_date, rows


def plot_equity(equity_df, benchmark_equity, benchmark2_equity, initial_capital):
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), gridspec_kw={"height_ratios": [3, 1]})
    ax1, ax2 = axes
    for ax in axes:
        ax.set_facecolor("#ffffff")

    ax1.plot(equity_df.index, equity_df["Equity"], color="#007aff", lw=2, label="Day Trade")
    ax1.axhline(initial_capital, color="#8e8e93", linestyle="--", alpha=0.6, label="Initial Capital")
    if benchmark_equity is not None and len(benchmark_equity) > 0:
        common = equity_df.index.intersection(benchmark_equity.index)
        if len(common) > 0:
            ax1.plot(
                common,
                benchmark_equity.loc[common] * initial_capital,
                color="#ff9500",
                lw=1.5,
                linestyle="--",
                label=f"{DEFAULT_BENCHMARK_LABEL} Buy & Hold",
            )
    if benchmark2_equity is not None and len(benchmark2_equity) > 0:
        common = equity_df.index.intersection(benchmark2_equity.index)
        if len(common) > 0:
            bench2 = benchmark2_equity.loc[common] / benchmark2_equity.loc[common].iloc[0]
            ax1.plot(
                common,
                bench2 * equity_df.loc[common, "Equity"].iloc[0],
                color="#34c759",
                lw=1.4,
                linestyle="-.",
                label=f"{SECONDARY_BENCHMARK_LABEL} Buy & Hold",
            )

    ax1.set_title("HK Day Trade Equity Curve", fontsize=14, fontweight="bold", color="#1d1d1f")
    ax1.set_ylabel(f"Portfolio Value ({CURRENCY})", fontsize=11, color="#1d1d1f")
    ax1.grid(alpha=0.08, color="#000000")
    ax1.legend(fontsize=9, loc="upper left")

    equity = equity_df["Equity"]
    drawdown = equity / equity.cummax() - 1
    ax2.fill_between(drawdown.index, 0, drawdown * 100, color="#ff3b30", alpha=0.35)
    ax2.plot(drawdown.index, drawdown * 100, color="#ff3b30", lw=1)
    ax2.set_ylabel("Drawdown (%)", fontsize=10, color="#1d1d1f")
    ax2.grid(alpha=0.08, color="#000000")
    fig.tight_layout()
    fig.savefig("day_trade_chart.png", dpi=150, bbox_inches="tight", facecolor="#f5f5f7")
    plt.close(fig)


def benchmark_card_html(equity_df, benchmark_equity, label, initial_capital, mode="annual"):
    if benchmark_equity is None or len(benchmark_equity) <= 20:
        return ""

    common = equity_df.index.intersection(benchmark_equity.index)
    if len(common) <= 20:
        return ""

    strat = equity_df.loc[common, "Equity"]
    bench = benchmark_equity.loc[common]
    bench_eq = bench / bench.iloc[0] * float(strat.iloc[0])

    strat_m = compute_risk_metrics(
        pd.DataFrame({"Equity": strat / strat.iloc[0] * strat.iloc[0]}, index=common),
        pd.DataFrame(),
        float(strat.iloc[0]),
    )
    bench_m = compute_risk_metrics(
        pd.DataFrame({"Equity": bench_eq}, index=common),
        pd.DataFrame(),
        float(strat.iloc[0]),
    )

    if mode == "total":
        s_ret = float(strat.iloc[-1] / strat.iloc[0] - 1) * 100
        b_ret = float(bench_eq.iloc[-1] / bench_eq.iloc[0] - 1) * 100
        edge = s_ret - b_ret
        edge_label = f"超額 vs {label} (共存期)"
        edge_value = f"{edge:+.1f}%"
    else:
        edge = (strat_m["ann_return"] - bench_m["ann_return"]) * 100
        edge_label = f"年化 alpha vs {label}"
        edge_value = f"{edge:+.1f}%"

    color = "#00ff00" if edge > 0 else "#ff4444"
    return f"""
    <div class="stats">
        <div class="stat-card benchmark">
            <div class="label">{label} 年化報酬</div>
            <div class="value">{bench_m['ann_return']*100:+.1f}%</div>
        </div>
        <div class="stat-card benchmark">
            <div class="label">{label} 最大回撤</div>
            <div class="value">{bench_m['max_drawdown_pct']*100:.1f}%</div>
        </div>
        <div class="stat-card benchmark">
            <div class="label">{label} Sharpe</div>
            <div class="value">{bench_m['sharpe']:.2f}</div>
        </div>
        <div class="stat-card" style="border-left-color:{color}">
            <div class="label">{edge_label}</div>
            <div class="value" style="color:{color}">{edge_value}</div>
        </div>
    </div>"""


def write_artifacts(date_str, trades_df, equity_df, signals, config, metrics):
    os.makedirs("artifacts", exist_ok=True)
    equity_path = f"artifacts/day_equity_{date_str}.csv"
    trades_path = f"artifacts/day_trades_{date_str}.csv"
    signals_path = f"artifacts/day_signals_{date_str}.csv"
    orders_path = f"artifacts/day_orders_{date_str}.json"
    metadata_path = f"artifacts/day_metadata_{date_str}.json"

    equity_df.to_csv(equity_path)
    trades_df.to_csv(trades_path, index=False)
    pd.DataFrame(signals).to_csv(signals_path, index=False)
    with open(orders_path, "w", encoding="utf-8") as f:
        json.dump({"orders": signals}, f, indent=2, ensure_ascii=False, default=str)

    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        git_sha = None

    metadata = {
        "created_at": datetime.now().isoformat(),
        "strategy_version": "day-trade-v1.1-gated",
        "git_sha": git_sha,
        "report_date": date_str,
        "config": config,
        "metrics": {
            "total_return": metrics.get("total_return"),
            "ann_return": metrics.get("ann_return"),
            "sharpe": metrics.get("sharpe"),
            "sortino": metrics.get("sortino"),
            "calmar": metrics.get("calmar"),
            "max_drawdown_pct": metrics.get("max_drawdown_pct"),
            "total_trades": metrics.get("total_trades"),
        },
        "artifacts": {
            "equity": equity_path,
            "trades": trades_path,
            "signals": signals_path,
            "orders": orders_path,
        },
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)


def generate_html(trades_df, equity_df, metrics, signals, latest_date, config,
                  benchmark_equity=None, benchmark2_equity=None):
    report_date = latest_date.strftime("%Y-%m-%d")
    next_date = next_trading_day(latest_date)
    total_ret_color = "#00ff00" if metrics["total_return"] > 0 else "#ff4444"
    sharpe_color = "#00ff00" if metrics["sharpe"] > 0.7 else ("#ffab00" if metrics["sharpe"] > 0 else "#ff4444")
    dd_color = "#ff4444" if metrics["max_drawdown_pct"] < -0.15 else "#ffab00"

    signal_rows = ""
    for item in signals:
        if item["status"].startswith("建議"):
            side_color = "#ffab00" if "short" in item["status"] else "#00ff00"
            status = f'<span style="color:{side_color};font-weight:bold;">🟢 {item["status"]} #{item["rank"]}</span>'
            plan = (
                f'<b>估算進場:</b> {item["ref_entry"]:.2f}<br>'
                f'<b>停利:</b> <span style="color:#00ff00">{item["tp"]:.2f}</span> ({item["tp_pct"]:.1f}%)<br>'
                f'<b>停損:</b> <span style="color:#ff4444">{item["sl"]:.2f}</span> ({item["sl_pct"]:.1f}%)<br>'
                f'<b>出場:</b> 當日收盤前'
            )
            opacity = "1"
        elif "候選" in item["status"]:
            status = '<span style="color:#ffab00;">🟡 候選</span>'
            plan = "-"
            opacity = "0.65"
        else:
            status = f'<span style="color:#aaaaaa;">⚪ {item["status"]}</span>'
            plan = "-"
            opacity = "0.45"
        signal_rows += (
            f'<tr style="opacity:{opacity}"><td>{item["ticker"]}</td>'
            f'<td>{item["score"]:.2f}</td><td>{item["price"]:.2f}</td>'
            f'<td>{item["trigger"]}</td><td>{status}</td><td>{plan}</td></tr>\n'
        )
    if not signal_rows:
        signal_rows = '<tr><td colspan="6" style="color:#a8b3bd;">今日沒有符合 watchlist 條件的標的。</td></tr>'

    recent_rows = ""
    if not trades_df.empty:
        recent = trades_df.sort_values("Entry_Date", ascending=False).head(40)
        for _, tr in recent.iterrows():
            color = "#00ff00" if tr["Return_Pct"] > 0 else "#ff4444"
            recent_rows += (
                f'<tr><td>{pd.Timestamp(tr["Entry_Date"]).strftime("%Y-%m-%d")}</td>'
                f'<td>{tr["Ticker"]}</td><td>{tr["Rank"]}</td>'
                f'<td>{tr["Entry_Price"]:.2f}</td><td>{tr["Exit_Price"]:.2f}</td>'
                f'<td>{tr["Reason"]}</td>'
                f'<td style="color:{color};font-weight:bold;">{tr["Return_Pct"]*100:+.2f}%</td></tr>\n'
            )

    reason_rows = ""
    if not trades_df.empty:
        for reason, subset in trades_df.groupby("Reason"):
            avg = subset["Return_Pct"].mean() * 100
            color = "#00ff00" if avg > 0 else "#ff4444"
            reason_rows += (
                f'<tr><td>{reason}</td><td>{len(subset)}</td>'
                f'<td style="color:{color};font-weight:bold;">{avg:+.2f}%</td></tr>\n'
            )

    monthly_html = ""
    try:
        monthly = equity_df["Equity"].resample("ME").last().pct_change().dropna()
        monthly_rows = ""
        for idx, val in monthly.tail(24).items():
            color = "#00ff00" if val > 0 else "#ff4444"
            monthly_rows += (
                f'<tr><td>{idx.strftime("%Y-%m")}</td>'
                f'<td style="color:{color};font-weight:bold;">{val*100:+.2f}%</td></tr>\n'
            )
        monthly_html = f"""
        <table>
            <thead><tr><th>月份</th><th>報酬</th></tr></thead>
            <tbody>{monthly_rows}</tbody>
        </table>"""
    except Exception:
        monthly_html = "<p class=\"section-note\">月度資料不足</p>"

    benchmark_html = benchmark_card_html(
        equity_df, benchmark_equity, DEFAULT_BENCHMARK_LABEL,
        config["initial_capital"], mode="annual",
    )
    benchmark2_html = benchmark_card_html(
        equity_df, benchmark2_equity, SECONDARY_BENCHMARK_LABEL,
        config["initial_capital"], mode="total",
    )

    html = f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI 港股 Day Trade 報告 — {report_date}</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            margin: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background: #0b0f14;
            color: #f5f5f7;
            line-height: 1.55;
        }}
        .container {{ max-width: 1320px; margin: 0 auto; padding: 28px 18px 56px; }}
        h1 {{ font-size: 2rem; margin: 0 0 8px; letter-spacing: 0; }}
        h2 {{ margin-top: 28px; color: #ffffff; border-bottom: 1px solid #25313d; padding-bottom: 8px; }}
        .subtitle {{ color: #a8b3bd; margin-bottom: 18px; }}
        .badge {{
            display: inline-block;
            margin: 4px 6px 4px 0;
            padding: 5px 9px;
            border-radius: 6px;
            background: #17212b;
            color: #dbe7f0;
            border: 1px solid #25313d;
            font-size: 0.9rem;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
            gap: 12px;
            margin: 14px 0;
        }}
        .stat-card {{
            background: #111820;
            border: 1px solid #25313d;
            border-left: 4px solid #007aff;
            border-radius: 8px;
            padding: 14px;
            min-height: 86px;
        }}
        .stat-card.benchmark {{ border-left-color: #ff9500; }}
        .label {{ color: #98a6b3; font-size: 0.82rem; margin-bottom: 6px; }}
        .value {{ font-size: 1.55rem; font-weight: 750; color: #f5f5f7; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: #111820;
            border: 1px solid #25313d;
            border-radius: 8px;
            overflow: hidden;
            margin-top: 12px;
        }}
        th, td {{ padding: 10px 12px; border-bottom: 1px solid #25313d; text-align: left; vertical-align: top; }}
        th {{ color: #9fb0bf; background: #17212b; font-size: 0.85rem; }}
        td {{ color: #edf5fb; }}
        .section-note {{ color: #a8b3bd; margin: 8px 0 12px; }}
        img {{ width: 100%; max-width: 100%; border-radius: 8px; border: 1px solid #25313d; background: #fff; }}
        .grid-2 {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 16px; }}
        .disclaimer {{
            margin-top: 28px;
            color: #a8b3bd;
            background: #111820;
            border: 1px solid #25313d;
            border-radius: 8px;
            padding: 14px;
            font-size: 0.92rem;
        }}
        @media (max-width: 800px) {{
            .container {{ padding: 20px 10px 44px; }}
            .grid-2 {{ grid-template-columns: 1fr; }}
            table {{ font-size: 0.86rem; }}
            th, td {{ padding: 8px; }}
            .value {{ font-size: 1.25rem; }}
        }}
    </style>
</head>
<body>
<div class="container">
    <h1>⚡ AI 港股 Day Trade 報告</h1>
    <p class="subtitle">
        報表日期：{report_date} ｜ 下一交易日：{next_date}
        <br>
        <span class="badge">t-1 watchlist → t open trigger → t close exit</span>
        <span class="badge">{config['direction'].upper()}</span>
        <span class="badge">Universe-{config['universe_size']}</span>
        <span class="badge">Scan-{config['scan_k']} / Top-{config['top_k']}</span>
        <span class="badge">Gap {config['gap_min']*100:+.1f}%~{config['gap_max']*100:+.1f}%</span>
        <span class="badge">ATR×{config['tp_atr']}/{config['sl_atr']}</span>
        <span class="badge">RSI×{config['rsi_weight']}</span>
        <span class="badge">每筆 {config['position_size']*100:.0f}%</span>
    </p>

    <h2>績效總覽</h2>
    <div class="stats">
        <div class="stat-card"><div class="label">策略總報酬率</div><div class="value" style="color:{total_ret_color};">{metrics['total_return']*100:+.1f}%</div></div>
        <div class="stat-card"><div class="label">年化報酬率</div><div class="value" style="color:{total_ret_color};">{metrics['ann_return']*100:+.1f}%</div></div>
        <div class="stat-card"><div class="label">Sharpe Ratio</div><div class="value" style="color:{sharpe_color};">{metrics['sharpe']:.2f}</div></div>
        <div class="stat-card"><div class="label">最大回撤</div><div class="value" style="color:{dd_color};">{metrics['max_drawdown_pct']*100:.1f}%</div></div>
        <div class="stat-card"><div class="label">完成交易數</div><div class="value">{metrics['total_trades']}</div></div>
        <div class="stat-card"><div class="label">勝率</div><div class="value">{metrics['win_rate']*100:.1f}%</div></div>
        <div class="stat-card"><div class="label">Profit Factor</div><div class="value">{metrics['profit_factor']:.2f}</div></div>
        <div class="stat-card"><div class="label">平均每筆</div><div class="value">{metrics['avg_return']*100:+.2f}%</div></div>
    </div>

    <h2>下一交易日 Day Trade 執行單</h2>
    <p class="section-note">
        以 {report_date} 收盤資料產生 watchlist；{next_date} 開盤後只有符合 gap 條件才執行。
        預設為 short day trade，需確認標的可融券、可借券或有對應衍生工具。
    </p>
    <table>
        <thead><tr><th>股票代號</th><th>AI 評分</th><th>昨收參考</th><th>開盤觸發</th><th>狀態</th><th>日內執行計畫</th></tr></thead>
        <tbody>{signal_rows}</tbody>
    </table>

    <h2>資金曲線 vs Benchmark</h2>
{benchmark_html}
{benchmark2_html}
    <img src="day_trade_chart.png" alt="HK day trade equity curve">

    <div class="grid-2">
        <div>
            <h2>出場原因</h2>
            <table>
                <thead><tr><th>原因</th><th>筆數</th><th>平均報酬</th></tr></thead>
                <tbody>{reason_rows}</tbody>
            </table>
        </div>
        <div>
            <h2>近 24 個月</h2>
            {monthly_html}
        </div>
    </div>

    <h2>近期 Day Trades</h2>
    <table>
        <thead><tr><th>日期</th><th>股票</th><th>Rank</th><th>進場</th><th>出場</th><th>原因</th><th>報酬</th></tr></thead>
        <tbody>{recent_rows}</tbody>
    </table>

    <div class="disclaimer">
        <b>方法論：</b>信號使用前一日收盤後可得資料；開盤 gap 條件成立才進場，並加入滑價與港股交易成本；
        日內 TP/SL 由日線 High/Low 近似，若同日同時觸及 TP/SL，採用 {config['ambiguous']}。
        Short 版本未納入借券可得性與借券費，實盤前必須另外檢查。
        本報表僅供研究與技術交流，不構成投資建議。
    </div>
</div>
</body>
</html>
"""

    with open("day_trade_report.html", "w", encoding="utf-8") as f:
        f.write(html)


def parse_args():
    parser = argparse.ArgumentParser(description="AI 港股 Day Trade 報告")
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument("--static-pool", action="store_true")
    parser.add_argument("--days", type=int, default=1200)
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--capital", type=float, default=200_000)
    parser.add_argument("--universe-size", type=int, default=35)
    parser.add_argument("--scan-k", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=2.9)
    parser.add_argument("--position-size", type=float, default=0.08)
    parser.add_argument("--tp-atr", type=float, default=1.3)
    parser.add_argument("--sl-atr", type=float, default=0.8)
    parser.add_argument("--gap-min", type=float, default=0.03)
    parser.add_argument("--gap-max", type=float, default=0.075)
    parser.add_argument("--gap-bonus", type=float, default=0.2)
    parser.add_argument("--atr-period", type=int, default=20)
    parser.add_argument("--rsi-weight", type=float, default=1.0)
    parser.add_argument("--breakout-weight", type=float, default=0.0)
    parser.add_argument("--ma-period", type=int, default=30)
    parser.add_argument("--slippage", type=float, default=0.001)
    parser.add_argument("--buy-cost", type=float, default=DEFAULT_BUY_COST)
    parser.add_argument("--sell-cost", type=float, default=DEFAULT_SELL_COST)
    parser.add_argument(
        "--direction",
        choices=["long", "short"],
        default="short",
        help="Day trade 方向，預設 short gap-up mean reversion",
    )
    parser.add_argument(
        "--ambiguous",
        choices=["stop-first", "profit-first", "close"],
        default="stop-first",
        help="同日同時觸及 TP/SL 時的假設，預設採保守 stop-first",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    tickers = args.tickers if args.tickers else (DEFAULT_TICKERS if args.static_pool else EXTENDED_TICKERS)
    use_dynamic = not args.static_pool and not args.tickers

    print("=" * 60)
    print("⚡ AI 港股 Day Trade 報告")
    print("=" * 60)
    print(f"   股池: {'動態 Universe Top-' + str(args.universe_size) if use_dynamic else '靜態 ' + str(len(tickers)) + ' 檔'}")
    print(
        f"   方向: {args.direction.upper()}  Gap: {args.gap_min*100:+.1f}%~{args.gap_max*100:+.1f}%  "
        f"Scan/Top-K: {args.scan_k}/{args.top_k}"
    )
    print(f"   TP/SL: ATR×{args.tp_atr}/{args.sl_atr}  每筆: {args.position_size*100:.1f}%")
    print(f"   成本: 買 {args.buy_cost*100:.3f}% 賣 {args.sell_cost*100:.3f}%  滑價: {args.slippage*100:.2f}%")
    print("=" * 60)

    close_df, open_df, high_df, low_df, vol_df = fetch_panel_data(
        tickers,
        days=args.days,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    universe_mask = build_liquid_universe(close_df, vol_df, top_n=args.universe_size) if use_dynamic else None

    total_score, ma_60, _, _ = engineer_features(
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
        ma_60,
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
        ma_60,
        atr_df,
        universe_mask,
        config,
    )

    print("📊 載入 Benchmark...")
    benchmark_equity = fetch_benchmark(DEFAULT_BENCHMARK, days=args.days, start_date=args.start_date, end_date=args.end_date)
    benchmark2_equity = fetch_benchmark(SECONDARY_BENCHMARK, days=args.days, start_date=args.start_date, end_date=args.end_date)
    _ = equal_weight_benchmark(close_df)

    print("📈 產出 day_trade_chart.png 與 day_trade_report.html...")
    plot_equity(equity_df, benchmark_equity, benchmark2_equity, args.capital)
    generate_html(trades_df, equity_df, metrics, signals, latest_date, config, benchmark_equity, benchmark2_equity)
    write_artifacts(latest_date.strftime("%Y%m%d"), trades_df, equity_df, signals, config, metrics)

    print("✅ Day trade 報告已生成：day_trade_report.html")


if __name__ == "__main__":
    main()
