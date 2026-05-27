#!/usr/bin/env python3
"""Bounded optimizer for the combined HK master strategy."""

import argparse
import contextlib
import json
import os
from datetime import datetime
from types import SimpleNamespace

import pandas as pd

from benchmark_beater_report import compare_with_benchmark
from master_strategy_report import combine_equity, combine_trades, run_day_overlay, run_swing_book
from strategy.ai_strategy import fetch_panel_data
from strategy.benchmark import fetch_benchmark
from strategy.market import (
    DEFAULT_BENCHMARK,
    DEFAULT_BENCHMARK_LABEL,
    DEFAULT_BUY_COST,
    DEFAULT_SELL_COST,
    SECONDARY_BENCHMARK,
    SECONDARY_BENCHMARK_LABEL,
)
from strategy.risk_metrics import compute_risk_metrics
from strategy.universe import EXTENDED_TICKERS


OUT_CSV = "artifacts/master_optimization_results.csv"
OUT_BEST = "artifacts/master_optimization_best.json"
OUT_SUMMARY = "artifacts/master_optimization_summary.json"


def base_args(args):
    return SimpleNamespace(
        tickers=None,
        static_pool=False,
        days=args.days,
        start_date=args.start_date,
        end_date=args.end_date,
        capital=args.capital,
        slippage=args.slippage,
        buy_cost=DEFAULT_BUY_COST,
        sell_cost=DEFAULT_SELL_COST,
        day_overlay_scale=0.0,
        swing_universe=50,
        swing_top_k=12,
        swing_threshold=2.0,
        swing_position_size=0.12,
        swing_tp_atr=3.0,
        swing_sl_atr=3.0,
        swing_hold_days=20,
        swing_ma_period=60,
        swing_rsi_weight=1.5,
        swing_breakout_weight=0.0,
        swing_gap_filter=0.0,
        swing_dd_pause_pct=0.10,
        swing_dd_pause_days=5,
        swing_consec_loss_limit=3,
        swing_consec_loss_pause=5,
        swing_sector_max_pct=0.75,
        day_universe=35,
        day_scan_k=4,
        day_top_k=4,
        day_threshold=2.9,
        day_position_size=0.08,
        day_tp_atr=1.3,
        day_sl_atr=0.8,
        day_gap_min=0.03,
        day_gap_max=0.075,
        day_gap_bonus=0.2,
        day_atr_period=20,
        day_rsi_weight=1.0,
        day_breakout_weight=0.0,
        day_ma_period=30,
        day_direction="short",
        day_ambiguous="stop-first",
    )


def day_variants(args):
    current = base_args(args)
    yield "day_trade_v1.1", current

    companion = base_args(args)
    companion.day_universe = 60
    companion.day_scan_k = 12
    companion.day_top_k = 4
    companion.day_threshold = 2.0
    companion.day_position_size = 0.08
    companion.day_tp_atr = 1.0
    companion.day_sl_atr = 0.75
    companion.day_gap_min = 0.02
    companion.day_gap_max = 0.08
    companion.day_rsi_weight = 1.5
    companion.day_ma_period = 60
    yield "bb_day_companion", companion

    conservative = base_args(args)
    conservative.day_position_size = 0.06
    conservative.day_gap_min = 0.03
    conservative.day_gap_max = 0.075
    yield "day_trade_conservative", conservative

    wider_gap = base_args(args)
    wider_gap.day_gap_min = 0.02
    wider_gap.day_gap_max = 0.08
    yield "day_trade_wider_gap", wider_gap


def gate_from_comparisons(comparisons):
    checks = []
    for comp in comparisons.values():
        checks.extend([
            bool(comp and comp["beats_total"]),
            bool(comp and comp["beats_ann"]),
            bool(comp and comp["beats_sharpe"]),
        ])
    return bool(checks and all(checks))


def objective(row, incumbent):
    mdd_penalty = max(0.0, abs(row["mdd"]) - abs(incumbent["mdd"]) - 0.03) * 2.0
    return (
        row["ann"] * 2.0
        + row["sharpe"] * 1.0
        + row["calmar"] * 0.30
        + min(row["total_excess_2800"], row["total_excess_2828"]) * 0.15
        - mdd_penalty
    )


def evaluate(scale, day_name, day_args, swing, day, benchmark_2800, benchmark_2828, capital):
    day_args.day_overlay_scale = scale
    equity_df = combine_equity(swing, day, day_args)
    trades_df = combine_trades(swing, day, day_args)
    metrics = compute_risk_metrics(equity_df[["Equity"]], trades_df, capital)
    comparisons = {
        DEFAULT_BENCHMARK_LABEL: compare_with_benchmark(
            equity_df[["Equity"]], benchmark_2800, DEFAULT_BENCHMARK_LABEL, capital
        ),
        SECONDARY_BENCHMARK_LABEL: compare_with_benchmark(
            equity_df[["Equity"]], benchmark_2828, SECONDARY_BENCHMARK_LABEL, capital
        ),
    }
    comp_2800 = comparisons[DEFAULT_BENCHMARK_LABEL]
    comp_2828 = comparisons[SECONDARY_BENCHMARK_LABEL]
    return {
        "day_variant": day_name,
        "day_overlay_scale": float(scale),
        "ann": float(metrics["ann_return"]),
        "sharpe": float(metrics["sharpe"]),
        "sortino": float(metrics["sortino"]),
        "calmar": float(metrics["calmar"]),
        "mdd": float(metrics["max_drawdown_pct"]),
        "total_return": float(metrics["total_return"]),
        "trades": int(metrics["total_trades"]),
        "win_rate": float(metrics["win_rate"]),
        "profit_factor": float(metrics["profit_factor"]),
        "gate": gate_from_comparisons(comparisons),
        "total_excess_2800": float(comp_2800["total_excess"]) if comp_2800 else None,
        "ann_alpha_2800": float(comp_2800["ann_alpha"]) if comp_2800 else None,
        "total_excess_2828": float(comp_2828["total_excess"]) if comp_2828 else None,
        "ann_alpha_2828": float(comp_2828["ann_alpha"]) if comp_2828 else None,
        "day_config": day["config"],
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Optimize HK master strategy")
    parser.add_argument("--days", type=int, default=1200)
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--capital", type=float, default=200_000)
    parser.add_argument("--slippage", type=float, default=0.001)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs("artifacts", exist_ok=True)

    print("🔎 Master optimizer loading data...")
    close_df, open_df, high_df, low_df, vol_df = fetch_panel_data(
        EXTENDED_TICKERS,
        days=args.days,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    benchmark_2800 = fetch_benchmark(DEFAULT_BENCHMARK, days=args.days, start_date=args.start_date, end_date=args.end_date)
    benchmark_2828 = fetch_benchmark(SECONDARY_BENCHMARK, days=args.days, start_date=args.start_date, end_date=args.end_date)

    with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
        swing = run_swing_book(close_df, open_df, high_df, low_df, vol_df, base_args(args))

    rows = []
    day_cache = {}
    for name, candidate_args in day_variants(args):
        with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
            day = run_day_overlay(close_df, open_df, high_df, low_df, vol_df, candidate_args)
        day_cache[name] = day

        for scale in [x * 0.5 for x in range(0, 25)]:
            rows.append(evaluate(scale, name, candidate_args, swing, day, benchmark_2800, benchmark_2828, args.capital))

    incumbent = next(r for r in rows if r["day_variant"] == "day_trade_v1.1" and r["day_overlay_scale"] == 0.0)
    for row in rows:
        row["score"] = objective(row, incumbent)

    coarse_best = max([r for r in rows if r["gate"]], key=lambda r: r["score"])
    print(
        "Coarse best: "
        f"{coarse_best['day_variant']} scale={coarse_best['day_overlay_scale']:.2f} "
        f"ann={coarse_best['ann']*100:+.2f}% "
        f"sharpe={coarse_best['sharpe']:.3f} "
        f"mdd={coarse_best['mdd']*100:.1f}%"
    )

    fine_scales = [
        round(coarse_best["day_overlay_scale"] - 1.0 + x * 0.1, 2)
        for x in range(0, 21)
        if coarse_best["day_overlay_scale"] - 1.0 + x * 0.1 >= 0
    ]
    best_day = day_cache[coarse_best["day_variant"]]
    best_args = next(a for n, a in day_variants(args) if n == coarse_best["day_variant"])
    existing = {(r["day_variant"], r["day_overlay_scale"]) for r in rows}
    for scale in fine_scales:
        key = (coarse_best["day_variant"], float(scale))
        if key in existing:
            continue
        row = evaluate(scale, coarse_best["day_variant"], best_args, swing, best_day, benchmark_2800, benchmark_2828, args.capital)
        row["score"] = objective(row, incumbent)
        rows.append(row)

    fine_best = max([r for r in rows if r["gate"]], key=lambda r: r["score"])

    final_scales = [
        round(fine_best["day_overlay_scale"] - 0.5 + x * 0.02, 2)
        for x in range(0, 51)
        if fine_best["day_overlay_scale"] - 0.5 + x * 0.02 >= 0
    ]
    existing = {(r["day_variant"], r["day_overlay_scale"]) for r in rows}
    final_added = 0
    for scale in final_scales:
        key = (fine_best["day_variant"], float(scale))
        if key in existing:
            continue
        row = evaluate(scale, fine_best["day_variant"], best_args, swing, best_day, benchmark_2800, benchmark_2828, args.capital)
        row["score"] = objective(row, incumbent)
        rows.append(row)
        final_added += 1

    best = max([r for r in rows if r["gate"]], key=lambda r: r["score"])
    pre_micro_best = best
    micro_scales = [
        round(pre_micro_best["day_overlay_scale"] - 0.10 + x * 0.005, 3)
        for x in range(0, 41)
        if pre_micro_best["day_overlay_scale"] - 0.10 + x * 0.005 >= 0
    ]
    existing = {(r["day_variant"], r["day_overlay_scale"]) for r in rows}
    micro_added = 0
    for scale in micro_scales:
        key = (pre_micro_best["day_variant"], float(scale))
        if key in existing:
            continue
        row = evaluate(scale, pre_micro_best["day_variant"], best_args, swing, best_day, benchmark_2800, benchmark_2828, args.capital)
        row["score"] = objective(row, incumbent)
        rows.append(row)
        micro_added += 1

    best = max([r for r in rows if r["gate"]], key=lambda r: r["score"])
    pre_nano_best = best
    nano_scales = [
        round(pre_nano_best["day_overlay_scale"] - 0.02 + x * 0.001, 3)
        for x in range(0, 41)
        if pre_nano_best["day_overlay_scale"] - 0.02 + x * 0.001 >= 0
    ]
    existing = {(r["day_variant"], r["day_overlay_scale"]) for r in rows}
    nano_added = 0
    for scale in nano_scales:
        key = (pre_nano_best["day_variant"], float(scale))
        if key in existing:
            continue
        row = evaluate(scale, pre_nano_best["day_variant"], best_args, swing, best_day, benchmark_2800, benchmark_2828, args.capital)
        row["score"] = objective(row, incumbent)
        rows.append(row)
        nano_added += 1

    best = max([r for r in rows if r["gate"]], key=lambda r: r["score"])
    material_delta = 0.0001
    if best["score"] > pre_nano_best["score"] + material_delta:
        stop_reason = "nano_refinement_upgrade_found"
    else:
        stop_reason = "nano_refinement_no_material_upgrade"

    df = pd.DataFrame(rows).sort_values("score", ascending=False)
    df.to_csv(OUT_CSV, index=False)
    with open(OUT_BEST, "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2, ensure_ascii=False, default=str)
    summary = {
        "created_at": datetime.now().isoformat(),
        "best": best,
        "incumbent": incumbent,
        "coarse_best": coarse_best,
        "fine_best": fine_best,
        "pre_micro_best": pre_micro_best,
        "pre_nano_best": pre_nano_best,
        "stop_reason": stop_reason,
        "candidates_evaluated": len(rows),
        "final_refinement_added": final_added,
        "micro_refinement_added": micro_added,
        "nano_refinement_added": nano_added,
        "material_delta": material_delta,
        "out_csv": OUT_CSV,
    }
    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    print(
        "Champion: "
        f"{best['day_variant']} scale={best['day_overlay_scale']:.2f} "
        f"ann={best['ann']*100:+.2f}% "
        f"sharpe={best['sharpe']:.3f} "
        f"mdd={best['mdd']*100:.1f}% "
        f"score={best['score']:.3f}"
    )
    print(f"Stop reason: {stop_reason}")
    print(f"Saved: {OUT_CSV}")
    print(f"Saved: {OUT_BEST}")
    print(f"Saved: {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
