#!/usr/bin/env python3
"""One-shot optimizer for HK Stocker production defaults.

The script downloads HK data once, sweeps a bounded set of strategy parameters,
and ranks candidates against the 2800.HK buy-and-hold baseline.
"""

import argparse
import contextlib
import itertools
import json
import math
import os
from datetime import datetime

import numpy as np
import pandas as pd

from ai_report import EXTENDED_TICKERS
from strategy.ai_strategy import fetch_panel_data, build_liquid_universe, engineer_features
from strategy.benchmark import fetch_benchmark
from strategy.event_backtest import EventDrivenBacktester
from strategy.evaluation import slice_evaluation_window
from strategy.market import DEFAULT_BENCHMARK, DEFAULT_BUY_COST, DEFAULT_SELL_COST
from strategy.risk_metrics import compute_risk_metrics


DEFAULT_OUT = "artifacts/hk_optimization_results.csv"


def benchmark_metrics(days, start_date=None, end_date=None, initial_capital=200_000):
    bench = fetch_benchmark(
        DEFAULT_BENCHMARK, days=days, start_date=start_date, end_date=end_date
    )
    equity = pd.DataFrame({"Equity": bench * initial_capital}, index=bench.index)
    return compute_risk_metrics(equity, pd.DataFrame(), initial_capital), bench


def make_score(close_df, vol_df, universe_mask, market_close, feature_cfg):
    with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
        total_score, ma_60, _, _ = engineer_features(
            close_df,
            vol_df,
            universe_mask,
            ma_period=feature_cfg["ma_period"],
            residual_momentum=feature_cfg["residual_momentum"],
            trend_quality=feature_cfg["trend_quality"],
            liq_stability=feature_cfg["liq_stability"],
            liq_mode=feature_cfg["liq_mode"],
            market_close=market_close,
            rsi_weight=feature_cfg["rsi_weight"],
            breakout_weight=feature_cfg["breakout_weight"],
            rev_momentum_weight=feature_cfg["rev_momentum_weight"],
        )
    return total_score, ma_60


def run_candidate(
    close_df,
    open_df,
    high_df,
    low_df,
    vol_df,
    total_score,
    ma_60,
    universe_mask,
    market_close,
    params,
    initial_capital,
    eval_start=None,
):
    backtester = EventDrivenBacktester(
        max_hold_days=params["hold_days"],
        initial_capital=initial_capital,
        position_size=params["position_size"],
        tp_sl_mode="atr",
        tp_atr_mult=params["tp_atr"],
        sl_atr_mult=params["sl_atr"],
        trailing_stop=params["trailing_stop"],
        trailing_atr_mult=params["trailing_atr"],
        regime_filter=params["regime_filter"],
        regime_graduated=params["regime_graduated"],
        regime_floor=params["regime_floor"],
        gap_filter_atr=params["gap_filter"],
        slippage=params["slippage"],
        vol_parity=params["vol_parity"],
        dynamic_risk=params["dynamic_risk"],
        corr_filter=params["corr_filter"],
        sector_max_pct=params["sector_max_pct"],
        breadth_regime=params["breadth_regime"],
        gap_aware_sizing=params["gap_aware_sizing"],
        dynamic_topk=params["dynamic_topk"],
        dynamic_gap_filter=params["dynamic_gap_filter"],
        dynamic_corr_filter=params["dynamic_corr_filter"],
        sector_flow_tilt=params["sector_flow_tilt"],
        tilt_strength=params["tilt_strength"],
        tilt_windows=params["tilt_windows"],
        buy_cost=DEFAULT_BUY_COST,
        sell_cost=DEFAULT_SELL_COST,
    )
    with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
        trades_df, equity_df = backtester.run(
            total_score,
            close_df,
            open_df,
            high_df,
            low_df,
            ma_60,
            top_k=params["top_k"],
            threshold=params["threshold"],
            market_close=market_close if params["regime_filter"] else None,
            vol_df=vol_df,
            universe_mask=universe_mask,
        )
    report_equity, report_trades = slice_evaluation_window(
        equity_df, trades_df, eval_start=eval_start, initial_capital=initial_capital
    )
    metrics = compute_risk_metrics(report_equity, report_trades, initial_capital)
    return metrics


def search_space(mode):
    feature_grid = [
        {
            "feature_name": "base",
            "ma_period": 60,
            "residual_momentum": False,
            "trend_quality": False,
            "liq_stability": False,
            "liq_mode": "raw",
            "rsi_weight": 0.0,
            "breakout_weight": 0.0,
            "rev_momentum_weight": 0.0,
        },
        {
            "feature_name": "rsi",
            "ma_period": 60,
            "residual_momentum": False,
            "trend_quality": False,
            "liq_stability": False,
            "liq_mode": "raw",
            "rsi_weight": 1.0,
            "breakout_weight": 0.0,
            "rev_momentum_weight": 0.0,
        },
        {
            "feature_name": "breakout",
            "ma_period": 60,
            "residual_momentum": False,
            "trend_quality": False,
            "liq_stability": False,
            "liq_mode": "raw",
            "rsi_weight": 0.0,
            "breakout_weight": 1.0,
            "rev_momentum_weight": 0.0,
        },
        {
            "feature_name": "rsi+breakout",
            "ma_period": 60,
            "residual_momentum": False,
            "trend_quality": False,
            "liq_stability": False,
            "liq_mode": "raw",
            "rsi_weight": 0.8,
            "breakout_weight": 0.8,
            "rev_momentum_weight": 0.0,
        },
        {
            "feature_name": "residual",
            "ma_period": 60,
            "residual_momentum": True,
            "trend_quality": False,
            "liq_stability": False,
            "liq_mode": "raw",
            "rsi_weight": 0.0,
            "breakout_weight": 0.0,
            "rev_momentum_weight": 0.0,
        },
        {
            "feature_name": "trendq",
            "ma_period": 60,
            "residual_momentum": False,
            "trend_quality": True,
            "liq_stability": False,
            "liq_mode": "raw",
            "rsi_weight": 0.0,
            "breakout_weight": 0.0,
            "rev_momentum_weight": 0.0,
        },
    ]

    if mode == "quick":
        universe_sizes = [40, 60, 80]
        param_grid = list(itertools.product(
            [4, 7, 10],          # top_k
            [10, 15, 20, 30],    # hold_days
            [3.0, 4.0, 5.0],     # tp_atr
            [2.0, 3.0],          # sl_atr
            [1.5, 2.0],          # threshold
            [0.0, 1.5],          # gap_filter
            [False, True],       # regime_filter
        ))
    else:
        universe_sizes = [30, 40, 50, 60, 80, 100]
        param_grid = list(itertools.product(
            [3, 5, 7, 10, 12],
            [8, 10, 15, 20, 30, 40],
            [2.5, 3.0, 4.0, 5.0, 6.0],
            [1.5, 2.0, 2.5, 3.0, 3.5],
            [1.2, 1.5, 1.8, 2.0, 2.3],
            [0.0, 1.0, 1.5, 2.0],
            [False, True],
        ))

    return feature_grid, universe_sizes, param_grid


def params_from_tuple(values):
    top_k, hold_days, tp_atr, sl_atr, threshold, gap_filter, regime_filter = values
    return {
        "top_k": top_k,
        "hold_days": hold_days,
        "tp_atr": tp_atr,
        "sl_atr": sl_atr,
        "threshold": threshold,
        "gap_filter": gap_filter,
        "regime_filter": regime_filter,
        "regime_graduated": True,
        "regime_floor": 0.10,
        "position_size": 0.10,
        "trailing_stop": False,
        "trailing_atr": 2.0,
        "slippage": 0.001,
        "vol_parity": False,
        "dynamic_risk": False,
        "corr_filter": 0.8,
        "sector_max_pct": 0.75,
        "breadth_regime": True,
        "gap_aware_sizing": True,
        "dynamic_topk": False,
        "dynamic_gap_filter": False,
        "dynamic_corr_filter": False,
        "sector_flow_tilt": False,
        "tilt_strength": 1.0,
        "tilt_windows": [10, 15, 20],
    }


def score_candidate(metrics, baseline_metrics):
    ann_edge = metrics["ann_return"] - baseline_metrics["ann_return"]
    sharpe_edge = metrics["sharpe"] - baseline_metrics["sharpe"]
    mdd_penalty = max(0.0, abs(metrics["max_drawdown_pct"]) - abs(baseline_metrics["max_drawdown_pct"]))
    return ann_edge * 3.0 + sharpe_edge * 0.8 - mdd_penalty * 0.6


def main():
    parser = argparse.ArgumentParser(description="Optimize HK Stocker parameters")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--days", type=int, default=1200)
    parser.add_argument("--capital", type=float, default=200_000)
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--eval-start", type=str, default=None)
    parser.add_argument("--oos-start", type=str, default="2025-01-01")
    parser.add_argument("--out", type=str, default=DEFAULT_OUT)
    parser.add_argument("--max-candidates", type=int, default=240)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print(f"🔎 HK strategy optimizer — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   mode={args.mode}, days={args.days}, max_candidates={args.max_candidates}")

    close_df, open_df, high_df, low_df, vol_df = fetch_panel_data(
        EXTENDED_TICKERS,
        days=args.days,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    baseline_metrics, bench = benchmark_metrics(
        args.days, start_date=args.start_date, end_date=args.end_date,
        initial_capital=args.capital,
    )
    market_close = bench * bench.iloc[0] if len(bench) else None
    print(
        "   baseline 2800.HK: "
        f"ann={baseline_metrics['ann_return']*100:+.2f}% "
        f"sharpe={baseline_metrics['sharpe']:.3f} "
        f"mdd={baseline_metrics['max_drawdown_pct']*100:.1f}%"
    )

    feature_grid, universe_sizes, param_grid = search_space(args.mode)
    all_jobs = [
        (universe_size, feature_idx, values)
        for universe_size in universe_sizes
        for feature_idx in range(len(feature_grid))
        for values in param_grid
    ]
    if args.max_candidates and len(all_jobs) > args.max_candidates:
        chosen = np.linspace(0, len(all_jobs) - 1, args.max_candidates, dtype=int)
        jobs = [all_jobs[i] for i in chosen]
    else:
        jobs = all_jobs

    rows = []
    score_cache = {}
    universe_cache = {}

    for evaluated, (universe_size, feature_idx, values) in enumerate(jobs, start=1):
        if universe_size not in universe_cache:
            universe_cache[universe_size] = build_liquid_universe(
                close_df, vol_df, top_n=universe_size
            )
        universe_mask = universe_cache[universe_size]
        feature_cfg = feature_grid[feature_idx]
        score_key = (universe_size, feature_cfg["feature_name"])
        if score_key not in score_cache:
            score_cache[score_key] = make_score(
                close_df, vol_df, universe_mask, market_close, feature_cfg
            )
        total_score, ma_60 = score_cache[score_key]

        params = params_from_tuple(values)
        metrics = run_candidate(
            close_df, open_df, high_df, low_df, vol_df,
            total_score, ma_60, universe_mask, market_close,
            params, args.capital, eval_start=args.eval_start,
        )
        row = {
            "feature": feature_cfg["feature_name"],
            "universe_size": universe_size,
            **params,
            "ann": metrics["ann_return"] * 100,
            "sharpe": metrics["sharpe"],
            "mdd": metrics["max_drawdown_pct"] * 100,
            "calmar": metrics["calmar"],
            "trades": metrics["total_trades"],
            "win_rate": metrics["win_rate"] * 100,
            "pf": metrics["profit_factor"],
            "score": score_candidate(metrics, baseline_metrics),
            "beats_ann": metrics["ann_return"] > baseline_metrics["ann_return"],
            "beats_sharpe": metrics["sharpe"] > baseline_metrics["sharpe"],
            "beats_mdd": abs(metrics["max_drawdown_pct"]) <= abs(baseline_metrics["max_drawdown_pct"]),
        }
        rows.append(row)
        if evaluated % 25 == 0:
            best = max(rows, key=lambda r: r["score"])
            print(
                f"   {evaluated:>4d}/{len(jobs)} checked | best {best['feature']} "
                f"u={best['universe_size']} k={best['top_k']} hold={best['hold_days']} "
                f"ann={best['ann']:+.1f}% sh={best['sharpe']:.2f} mdd={best['mdd']:.1f}%"
            )

    df = pd.DataFrame(rows).sort_values("score", ascending=False)
    top = df.head(20).copy()

    oos_rows = []
    print(f"\n🧪 OOS sanity check from {args.oos_start} for top {len(top)}")
    for _, row in top.iterrows():
        universe_mask = build_liquid_universe(close_df, vol_df, top_n=int(row["universe_size"]))
        feature_cfg = next(f for f in feature_grid if f["feature_name"] == row["feature"])
        total_score, ma_60 = make_score(close_df, vol_df, universe_mask, market_close, feature_cfg)
        params = params_from_tuple((
            int(row["top_k"]),
            int(row["hold_days"]),
            float(row["tp_atr"]),
            float(row["sl_atr"]),
            float(row["threshold"]),
            float(row["gap_filter"]),
            bool(row["regime_filter"]),
        ))
        metrics = run_candidate(
            close_df, open_df, high_df, low_df, vol_df,
            total_score, ma_60, universe_mask, market_close,
            params, args.capital, eval_start=args.oos_start,
        )

        bench_oos = bench.loc[pd.Timestamp(args.oos_start):]
        if len(bench_oos) > 1:
            bench_oos_eq = pd.DataFrame(
                {"Equity": bench_oos / bench_oos.iloc[0] * args.capital},
                index=bench_oos.index,
            )
            bench_oos_metrics = compute_risk_metrics(
                bench_oos_eq, pd.DataFrame(), args.capital
            )
        else:
            bench_oos_metrics = baseline_metrics

        oos_rows.append({
            **row.to_dict(),
            "oos_ann": metrics["ann_return"] * 100,
            "oos_sharpe": metrics["sharpe"],
            "oos_mdd": metrics["max_drawdown_pct"] * 100,
            "oos_trades": metrics["total_trades"],
            "oos_beats_ann": metrics["ann_return"] > bench_oos_metrics["ann_return"],
            "oos_beats_sharpe": metrics["sharpe"] > bench_oos_metrics["sharpe"],
            "oos_benchmark_ann": bench_oos_metrics["ann_return"] * 100,
            "oos_benchmark_sharpe": bench_oos_metrics["sharpe"],
        })

    out_df = pd.DataFrame(oos_rows).sort_values(
        ["oos_beats_ann", "oos_beats_sharpe", "score"],
        ascending=[False, False, False],
    )
    out_df.to_csv(args.out, index=False)

    print(f"\n🏁 Results saved: {args.out}")
    display_cols = [
        "feature", "universe_size", "top_k", "hold_days", "tp_atr", "sl_atr",
        "threshold", "gap_filter", "regime_filter", "ann", "sharpe", "mdd",
        "oos_ann", "oos_sharpe", "oos_mdd", "oos_benchmark_ann",
        "oos_benchmark_sharpe",
    ]
    print(out_df[display_cols].head(10).to_string(index=False))

    best = out_df.iloc[0].to_dict()
    with open(args.out.replace(".csv", "_best.json"), "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2, ensure_ascii=False, default=str)
    print(f"✅ Best config JSON: {args.out.replace('.csv', '_best.json')}")


if __name__ == "__main__":
    main()
