# HK Stocker v9.1 — AI 量化交易系統（Master Strategy）

這是 `voidful/tw_stocker` 的港股版本，保留原本的 v8.5 Momentum + Sector Rotation v2 雙策略架構，並把市場層換成港股：

- Yahoo Finance ticker 使用 `.HK`，內部代號統一四碼，例如 `0700`、`9988`、`0005`
- 預設 benchmark 改為 `2800.HK`（盈富基金），第二 benchmark 為 `2828.HK`
- 交易日曆改為 `XHKG`
- 交易成本預設改為港股費率近似值：買賣各約 `0.1127%`，不含券商佣金
- 板塊分類改為港股代表股映射：互聯網/科技、金融、地產、消費、醫藥、能源/原材料、汽車/工業、電訊/公用

> 最新港股 artifacts：`report_date=2026-05-27`。Master v1 年化 `45.69%`、Sharpe `1.67`、最大回撤 `-16.5%`；相對 `2800.HK` 共存期總超額 `+147.2%`，相對 `2828.HK` 共存期總超額 `+99.2%`。Master = Benchmark Beater v1.1 core + Day Trade v1.1 intraday overlay。

## 雙策略架構

| | v8.5 Momentum | Sector Rotation v2 |
|---|:---:|:---:|
| 邏輯 | 個股 cross-sectional ranking | 先選板塊 → 板塊內排名 |
| Regime | 預設停用；仍顯示 `2800.HK` 診斷 | SPY + VIX + SOX |
| 選股因子 | Mom(20d)×3 + Trend(60MA)×1 + RSI×1.5 | 板塊 flow(10/15/20d) + 板塊內動量 |
| 角色 | 穩健底倉 | 積極追蹤 |

## 快速開始

```bash
pip install -r requirements.txt

# v8.5 Momentum
python ai_report.py --show-inst

# Benchmark Beater：專門對標 2800/2828 buy-and-hold
python benchmark_beater_report.py

# Master Strategy：Benchmark Beater core + Day Trade overlay
python master_strategy_report.py
python optimize_master_strategy.py

# Benchmark Beater bounded optimizer
python optimize_benchmark_beater.py

# Benchmark Beater 的獨立 Day Trade companion
python benchmark_beater_day_trade_report.py

# Day Trade 掃描報告
python day_trade_report.py

# Day Trade v1.1 預設：short gap-up mean reversion
# Universe-35 / MA30 / RSI×1.0 / Scan-4 / Top-4 / Threshold 2.9
# Gap +3.0%~+7.5% / ATR TP 1.3、SL 0.8 / 每筆 8%

# Sector Rotation v2
python sector_rotation_report.py
python sector_rotation_report.py --start-date 2019-01-01
python sector_rotation_report.py --compare

# 驗證工具
python walk_forward.py
python walk_forward_nested.py --quick
python monte_carlo.py --equity artifacts/equity_YYYYMMDD.csv --runs 2000 --block-size 20
python crisis_test.py

# 研究審計 / 多重測試修正
python sweep.py --quick
python factor_grid_search.py --mode ablation
python -m validation.deflated_sharpe --equity artifacts/equity_YYYYMMDD.csv --trials 20
python -m research.experiment_registry --latest 20

# Paper Trading
python paper_trade.py signals --enrich
python paper_trade.py hardstop
```

## 專案結構

```text
hk_stocker/
├── ai_report.py                  # v8.5 主程式 + CLI + HTML 報表
├── master_strategy_report.py     # 主策略：中線 core + 日內 overlay
├── optimize_master_strategy.py   # Master bounded optimizer
├── benchmark_beater_report.py    # 對標 2800/2828 的中線長倉報告
├── optimize_benchmark_beater.py  # Benchmark Beater bounded optimizer
├── benchmark_beater_day_trade_report.py
│                                  # Benchmark Beater 強勢股日內 companion
├── day_trade_report.py           # 日內 short gap-up 掃描 + 報告
├── sector_rotation_report.py     # 板塊輪動 v2 回測 + 報告
├── deep_crisis_test.py           # 歷史危機壓測 + benchmark 對照
├── crisis_test.py                # 基礎危機壓力測試
├── walk_forward.py               # Anchored OOS 穩定性驗證
├── walk_forward_nested.py        # Nested train→select→test research gate
├── monte_carlo.py                # Equity-Curve Block Bootstrap
├── sweep.py                      # 季度參數校準 + Telegram 警報
├── paper_trade.py                # Paper Trading 工具
├── strategy/
│   ├── market.py                 # 港股市場設定、ticker 轉換、成本常數
│   ├── universe.py               # 港股預設股池
│   ├── ai_strategy.py            # yfinance 港股資料下載 + 特徵工程
│   ├── event_backtest.py         # 事件驅動回測引擎
│   ├── sector_flow.py            # 港股板塊分類與資金流
│   ├── sector_rotation_backtest.py
│   └── benchmark.py              # Benchmark (`2800.HK` / EW)
├── research/
└── validation/
```

## 港股版注意事項

港股沒有台股「三大法人」同構資料源；目前 `institutional_flow` 和 `news_sentiment` 保留介面但回傳空/中性資料，避免把台股資料混進港股信號。

港股交易成本會依券商、產品、印花稅豁免和結算安排而不同；預設值只包含常見交易所/政府側費用近似，實盤前請用自己的券商費率覆蓋 `--buy-cost` / `--sell-cost`。
