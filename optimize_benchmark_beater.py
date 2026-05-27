#!/usr/bin/env python3
"""Targeted optimizer for the HK Benchmark Beater strategy.

The search is deliberately bounded and incumbent-aware:
- fetch one Yahoo snapshot and compare every candidate on that same data,
- require the 2800/2828 benchmark gate to pass,
- only accept a candidate when it beats the current champion's composite score,
- stop after a full round produces no accepted upgrade.
"""

import argparse
import contextlib
import json
import os
import random
from dataclasses import asdict, dataclass, replace
from datetime import datetime

import pandas as pd

from benchmark_beater_report import compare_with_benchmark
from strategy.ai_strategy import build_liquid_universe, engineer_features, fetch_panel_data
from strategy.benchmark import fetch_benchmark
from strategy.event_backtest import EventDrivenBacktester
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


OUT_CSV = "artifacts/bb_optimization_results.csv"
OUT_BEST = "artifacts/bb_optimization_best.json"
OUT_SUMMARY = "artifacts/bb_optimization_summary.json"


@dataclass(frozen=True)
class BBConfig:
    universe_size: int = 60
    top_k: int = 12
    threshold: float = 2.0
    position_size: float = 0.10
    tp_atr: float = 3.0
    sl_atr: float = 3.0
    hold_days: int = 20
    ma_period: int = 60
    rsi_weight: float = 1.5
    breakout_weight: float = 0.0
    gap_filter: float = 0.0
    dd_pause_pct: float = 0.10
    dd_pause_days: int = 5
    consec_loss_limit: int = 3
    consec_loss_pause: int = 5
    sector_max_pct: float = 0.75


PARAM_SPACE = {
    "universe_size": [40, 50, 60, 70, 80, 90],
    "top_k": [6, 8, 10, 12, 14, 16],
    "threshold": [1.6, 1.8, 2.0, 2.2, 2.4, 2.6],
    "position_size": [0.08, 0.09, 0.10, 0.11, 0.12, 0.13],
    "tp_atr": [2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
    "sl_atr": [2.0, 2.5, 3.0, 3.5, 4.0],
    "hold_days": [15, 18, 20, 22, 25, 30],
    "ma_period": [45, 50, 60, 70, 80],
    "rsi_weight": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5],
    "breakout_weight": [0.0, 0.5, 1.0],
    "gap_filter": [0.0, 0.5, 1.0, 1.5, 2.0],
    "dd_pause_pct": [0.08, 0.10, 0.12, 0.15],
    "dd_pause_days": [3, 5, 8],
    "consec_loss_limit": [2, 3, 4],
    "consec_loss_pause": [3, 5, 8],
    "sector_max_pct": [0.60, 0.75, 1.0],
}


def gate_from_comparisons(comparisons):
    checks = []
    for comp in comparisons.values():
        checks.extend([
            bool(comp and comp["beats_total"]),
            bool(comp and comp["beats_ann"]),
            bool(comp and comp["beats_sharpe"]),
        ])
    return bool(checks and all(checks))


def score_row(metrics, comparisons, incumbent_metrics):
    """Composite ranking score.

    Annual return and Sharpe dominate. Calmar and benchmark excess are tie
    breakers. Drawdown is penalized only after a small allowed research band
    above the incumbent, so the optimizer cannot win by simply concentrating
    until risk explodes.
    """
    min_total_excess = min(
        comp["total_excess"] for comp in comparisons.values() if comp is not None
    )
    incumbent_mdd = abs(float(incumbent_metrics["max_drawdown_pct"]))
    mdd = abs(float(metrics["max_drawdown_pct"]))
    over_mdd_band = max(0.0, mdd - incumbent_mdd - 0.03)
    trade_penalty = 0.0 if metrics["total_trades"] >= 300 else 0.40

    return (
        float(metrics["ann_return"]) * 2.5
        + float(metrics["sharpe"]) * 0.65
        + float(metrics["calmar"]) * 0.20
        + min_total_excess * 0.25
        - over_mdd_band * 1.40
        - trade_penalty
    )


class Optimizer:
    def __init__(self, args):
        self.args = args
        self.capital = args.capital
        self.universe_cache = {}
        self.score_cache = {}
        self.rows = []
        self.seen = set()

        self.close_df = None
        self.open_df = None
        self.high_df = None
        self.low_df = None
        self.vol_df = None
        self.benchmark_2800 = None
        self.benchmark_2828 = None
        self.incumbent_metrics = None

    def load_data(self):
        self.close_df, self.open_df, self.high_df, self.low_df, self.vol_df = fetch_panel_data(
            EXTENDED_TICKERS,
            days=self.args.days,
            start_date=self.args.start_date,
            end_date=self.args.end_date,
        )
        self.benchmark_2800 = fetch_benchmark(
            DEFAULT_BENCHMARK,
            days=self.args.days,
            start_date=self.args.start_date,
            end_date=self.args.end_date,
        )
        self.benchmark_2828 = fetch_benchmark(
            SECONDARY_BENCHMARK,
            days=self.args.days,
            start_date=self.args.start_date,
            end_date=self.args.end_date,
        )

    def universe_mask(self, universe_size):
        if universe_size not in self.universe_cache:
            self.universe_cache[universe_size] = build_liquid_universe(
                self.close_df,
                self.vol_df,
                top_n=universe_size,
            )
        return self.universe_cache[universe_size]

    def score_matrix(self, cfg):
        key = (
            cfg.universe_size,
            cfg.ma_period,
            cfg.rsi_weight,
            cfg.breakout_weight,
        )
        if key not in self.score_cache:
            mask = self.universe_mask(cfg.universe_size)
            with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
                total_score, ma_df, _, _ = engineer_features(
                    self.close_df,
                    self.vol_df,
                    mask,
                    ma_period=cfg.ma_period,
                    rsi_weight=cfg.rsi_weight,
                    breakout_weight=cfg.breakout_weight,
                )
            self.score_cache[key] = (total_score, ma_df)
        return self.score_cache[key]

    def run_candidate(self, cfg, origin):
        key = tuple(asdict(cfg).items())
        if key in self.seen:
            return None
        self.seen.add(key)

        total_score, ma_df = self.score_matrix(cfg)
        mask = self.universe_mask(cfg.universe_size)
        backtester = EventDrivenBacktester(
            initial_capital=self.capital,
            position_size=cfg.position_size,
            tp_sl_mode="atr",
            tp_atr_mult=cfg.tp_atr,
            sl_atr_mult=cfg.sl_atr,
            max_hold_days=cfg.hold_days,
            slippage=self.args.slippage,
            buy_cost=DEFAULT_BUY_COST,
            sell_cost=DEFAULT_SELL_COST,
            gap_filter_atr=cfg.gap_filter,
            dd_pause_pct=cfg.dd_pause_pct,
            dd_pause_days=cfg.dd_pause_days,
            consec_loss_limit=cfg.consec_loss_limit,
            consec_loss_pause=cfg.consec_loss_pause,
            sector_max_pct=cfg.sector_max_pct,
            gap_aware_sizing=True,
            breadth_regime=True,
        )
        with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
            trades_df, equity_df = backtester.run(
                total_score,
                self.close_df,
                self.open_df,
                self.high_df,
                self.low_df,
                ma_df,
                top_k=cfg.top_k,
                threshold=cfg.threshold,
                vol_df=self.vol_df,
                universe_mask=mask,
            )

        metrics = compute_risk_metrics(equity_df, trades_df, self.capital)
        comparisons = {
            DEFAULT_BENCHMARK_LABEL: compare_with_benchmark(
                equity_df,
                self.benchmark_2800,
                DEFAULT_BENCHMARK_LABEL,
                self.capital,
            ),
            SECONDARY_BENCHMARK_LABEL: compare_with_benchmark(
                equity_df,
                self.benchmark_2828,
                SECONDARY_BENCHMARK_LABEL,
                self.capital,
            ),
        }
        gate = gate_from_comparisons(comparisons)
        incumbent_metrics = self.incumbent_metrics or {
            "max_drawdown_pct": metrics["max_drawdown_pct"],
        }
        score = score_row(metrics, comparisons, incumbent_metrics)
        if not gate:
            score -= 5.0

        comp_2800 = comparisons[DEFAULT_BENCHMARK_LABEL]
        comp_2828 = comparisons[SECONDARY_BENCHMARK_LABEL]
        row = {
            "origin": origin,
            **asdict(cfg),
            "ann": float(metrics["ann_return"]),
            "sharpe": float(metrics["sharpe"]),
            "sortino": float(metrics["sortino"]),
            "calmar": float(metrics["calmar"]),
            "mdd": float(metrics["max_drawdown_pct"]),
            "trades": int(metrics["total_trades"]),
            "win_rate": float(metrics["win_rate"]),
            "profit_factor": float(metrics["profit_factor"]),
            "gate": gate,
            "score": float(score),
            "total_excess_2800": float(comp_2800["total_excess"]) if comp_2800 else None,
            "ann_alpha_2800": float(comp_2800["ann_alpha"]) if comp_2800 else None,
            "total_excess_2828": float(comp_2828["total_excess"]) if comp_2828 else None,
            "ann_alpha_2828": float(comp_2828["ann_alpha"]) if comp_2828 else None,
        }
        self.rows.append(row)
        return row

    def accepted_upgrade(self, candidate, champion):
        if candidate is None or not candidate["gate"]:
            return False
        if candidate["trades"] < 300:
            return False
        incumbent_mdd = abs(float(self.incumbent_metrics["max_drawdown_pct"]))
        if abs(candidate["mdd"]) > incumbent_mdd + self.args.max_mdd_relax:
            return False
        return candidate["score"] > champion["score"] + self.args.min_delta

    def single_dimension_candidates(self, champion_cfg, round_idx):
        for dim, values in PARAM_SPACE.items():
            for value in values:
                if getattr(champion_cfg, dim) == value:
                    continue
                yield replace(champion_cfg, **{dim: value}), f"round{round_idx}:{dim}"

    def random_candidates(self, champion_cfg):
        rng = random.Random(self.args.seed)
        dims = list(PARAM_SPACE)
        for _ in range(self.args.random_candidates):
            cfg = champion_cfg
            changed = rng.sample(dims, k=rng.randint(2, min(5, len(dims))))
            updates = {dim: rng.choice(PARAM_SPACE[dim]) for dim in changed}
            cfg = replace(cfg, **updates)
            yield cfg, "interaction_random"

    def optimize(self):
        self.load_data()
        incumbent_cfg = BBConfig()
        incumbent = self.run_candidate(incumbent_cfg, "incumbent")
        self.incumbent_metrics = {
            "ann_return": incumbent["ann"],
            "sharpe": incumbent["sharpe"],
            "calmar": incumbent["calmar"],
            "max_drawdown_pct": incumbent["mdd"],
        }
        incumbent["score"] = score_row(
            {
                "ann_return": incumbent["ann"],
                "sharpe": incumbent["sharpe"],
                "calmar": incumbent["calmar"],
                "max_drawdown_pct": incumbent["mdd"],
                "total_trades": incumbent["trades"],
            },
            {
                DEFAULT_BENCHMARK_LABEL: {
                    "total_excess": incumbent["total_excess_2800"],
                },
                SECONDARY_BENCHMARK_LABEL: {
                    "total_excess": incumbent["total_excess_2828"],
                },
            },
            self.incumbent_metrics,
        )

        champion = incumbent
        champion_cfg = incumbent_cfg
        print(
            "Incumbent: "
            f"ann={champion['ann']*100:+.2f}% "
            f"sharpe={champion['sharpe']:.3f} "
            f"mdd={champion['mdd']*100:.1f}% "
            f"score={champion['score']:.3f}"
        )

        stop_reason = "max_rounds"
        for round_idx in range(1, self.args.max_rounds + 1):
            round_best = champion
            round_best_cfg = champion_cfg
            checked = 0
            for cfg, origin in self.single_dimension_candidates(champion_cfg, round_idx):
                row = self.run_candidate(cfg, origin)
                checked += 1
                if self.accepted_upgrade(row, round_best):
                    round_best = row
                    round_best_cfg = cfg
            print(
                f"Round {round_idx}: checked={checked}, "
                f"best ann={round_best['ann']*100:+.2f}% "
                f"sharpe={round_best['sharpe']:.3f} "
                f"mdd={round_best['mdd']*100:.1f}% "
                f"score={round_best['score']:.3f}"
            )
            if round_best["score"] <= champion["score"] + self.args.min_delta:
                stop_reason = f"no_single_dimension_upgrade_round_{round_idx}"
                break
            champion = round_best
            champion_cfg = round_best_cfg

        interaction_best = champion
        interaction_best_cfg = champion_cfg
        checked = 0
        for cfg, origin in self.random_candidates(champion_cfg):
            row = self.run_candidate(cfg, origin)
            checked += 1
            if self.accepted_upgrade(row, interaction_best):
                interaction_best = row
                interaction_best_cfg = cfg
        print(
            f"Interaction random: checked={checked}, "
            f"best ann={interaction_best['ann']*100:+.2f}% "
            f"sharpe={interaction_best['sharpe']:.3f} "
            f"mdd={interaction_best['mdd']*100:.1f}% "
            f"score={interaction_best['score']:.3f}"
        )
        if interaction_best["score"] > champion["score"] + self.args.min_delta:
            champion = interaction_best
            champion_cfg = interaction_best_cfg
            stop_reason = "interaction_upgrade_found_then_final_sensitivity"

            final_best = champion
            final_cfg = champion_cfg
            checked = 0
            for cfg, origin in self.single_dimension_candidates(champion_cfg, "final"):
                row = self.run_candidate(cfg, origin)
                checked += 1
                if self.accepted_upgrade(row, final_best):
                    final_best = row
                    final_cfg = cfg
            print(
                f"Final sensitivity: checked={checked}, "
                f"best ann={final_best['ann']*100:+.2f}% "
                f"sharpe={final_best['sharpe']:.3f} "
                f"mdd={final_best['mdd']*100:.1f}% "
                f"score={final_best['score']:.3f}"
            )
            if final_best["score"] > champion["score"] + self.args.min_delta:
                champion = final_best
                champion_cfg = final_cfg
                stop_reason = "final_sensitivity_upgrade_found"
            else:
                stop_reason = "no_final_sensitivity_upgrade"

        self.write_outputs(champion, incumbent, stop_reason)
        return champion, incumbent, stop_reason

    def write_outputs(self, champion, incumbent, stop_reason):
        os.makedirs("artifacts", exist_ok=True)
        df = pd.DataFrame(self.rows).sort_values("score", ascending=False)
        df.to_csv(OUT_CSV, index=False)
        payload = {
            "created_at": datetime.now().isoformat(),
            "best": champion,
            "incumbent": incumbent,
            "stop_reason": stop_reason,
            "candidates_evaluated": len(self.rows),
            "out_csv": OUT_CSV,
        }
        with open(OUT_BEST, "w", encoding="utf-8") as f:
            json.dump(champion, f, indent=2, ensure_ascii=False, default=str)
        with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
        print(f"Saved: {OUT_CSV}")
        print(f"Saved: {OUT_BEST}")
        print(f"Saved: {OUT_SUMMARY}")


def parse_args():
    parser = argparse.ArgumentParser(description="Optimize Benchmark Beater parameters")
    parser.add_argument("--days", type=int, default=1200)
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--capital", type=float, default=200_000)
    parser.add_argument("--slippage", type=float, default=0.001)
    parser.add_argument("--max-rounds", type=int, default=4)
    parser.add_argument("--random-candidates", type=int, default=80)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--min-delta", type=float, default=0.01)
    parser.add_argument("--max-mdd-relax", type=float, default=0.04)
    return parser.parse_args()


def main():
    args = parse_args()
    optimizer = Optimizer(args)
    best, incumbent, stop_reason = optimizer.optimize()
    print("\nChampion")
    print(json.dumps(best, indent=2, ensure_ascii=False, default=str))
    print("\nIncumbent")
    print(json.dumps(incumbent, indent=2, ensure_ascii=False, default=str))
    print(f"\nStop reason: {stop_reason}")


if __name__ == "__main__":
    main()
