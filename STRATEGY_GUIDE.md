# HK Stocker v8.5 — 策略說明

本文件是港股版策略備忘。核心邏輯沿用 `tw_stocker`，但市場參數已切到 HKEX。

## v8.5 Momentum

每日流程：

1. 從港股候選池下載 OHLCV。
2. 以過去 20 日平均成交額建立動態 Universe，預設 Top-60。
3. 計算橫截面排名：20 日動量 ×3 + 60MA 趨勢 ×1 + RSI-20 ×1.5。
4. 預設不使用 `2800.HK` regime filter；報表仍保留大盤診斷。
5. 分數達標且站上 60MA 的個股，按 Top-K 進場。
6. 進場使用 t+1 open；出場用 ATR TP/SL + 最大持倉天數。

Production baseline：

- TP/SL：ATR 3.0 / 3.0
- Hold：20 個交易日
- Top-K：12
- Threshold：2.0
- Gap filter：停用
- Corr filter：停用
- Drawdown pause：權益回撤 10% 暫停新倉 5 天
- Consecutive loss pause：連續停損 3 次暫停新倉 5 天
- 成本：買賣各約 0.1127% + 滑價 10bps

最新港股 artifacts（報表日 2026-05-26；注意：Yahoo 重新下載後的資料快照會影響結果）：

- 年化報酬：31.60%
- Sharpe：1.27
- 最大回撤：-19.7%
- vs `2800.HK`：年化 alpha +18.6%
- vs `2828.HK`：共存期總報酬超額 +58.1%

## Master Strategy v1

Master Strategy 報告頁由 `python master_strategy_report.py` 產生，輸出 `master_strategy_report.html`。

參數優化由 `python optimize_master_strategy.py` 產生 artifacts：

- `artifacts/master_optimization_results.csv`
- `artifacts/master_optimization_best.json`
- `artifacts/master_optimization_summary.json`

Master v1 將兩條已驗證策略合成一條主資金曲線：

- Core：Benchmark Beater v1.1 中線長倉
- Overlay：Day Trade v1.1 short gap-up mean reversion
- Overlay scale：8.142
- 停止條件：粗掃、細掃、micro、nano grid 後，`nano_refinement_no_material_upgrade`

最新 artifacts（報表日 2026-05-27）：

- 年化報酬：45.69%
- Sharpe：1.67
- 最大回撤：-16.5%
- 總交易數：776
- vs `2800.HK` 共存期：總超額 +147.2%，年化 alpha +30.5%
- vs `2828.HK` 共存期：總超額 +99.2%，年化 alpha +34.9%

注意：Day Trade overlay scale 8.142 代表日內成交額被放大；回測未納入借券費、融資利息、實際可 short 名單與券商額外限制。這是研究用主策略，不是直接實盤槓桿建議。

## Benchmark Beater v1.1

Benchmark Beater 報告頁由 `python benchmark_beater_report.py` 產生，輸出 `benchmark_beater_report.html`。

參數優化由 `python optimize_benchmark_beater.py` 產生 artifacts：

- `artifacts/bb_optimization_results.csv`
- `artifacts/bb_optimization_best.json`
- `artifacts/bb_optimization_summary.json`

v1.1 的 bounded optimizer 評估 237 個候選後停止於 `no_single_dimension_upgrade_round_3`，代表第三輪單參數敏感性已找不到可接受升級。

這是專門為了打贏 `2800.HK` / `2828.HK` buy-and-hold 而獨立出來的中線長倉策略，核心與 v8.5 Momentum 一致，但報表 gate 明確要求在共存期間勝過兩個 benchmark：

- 動態 Universe：Top-50
- Score：Mom20 × 3 + Trend(MA60) × 1 + RSI20 × 1.5
- Top-K：12
- Threshold：2.0
- TP/SL：True Range ATR 3.0 / 3.0
- Hold：20 個交易日
- 每筆部位：12%
- Gap-aware sizing：開啟
- 回撤暫停：10% 回撤後暫停新倉 5 天
- 連續停損暫停：連續 3 次停損後暫停新倉 5 天

Benchmark gate（報表日 2026-05-26）：

- Full window：年化 31.60%、Sharpe 1.27、最大回撤 -19.7%、470 筆
- vs `2800.HK` 共存期：策略總報酬 +127.3%，benchmark +45.4%，總超額 +81.9%
- vs `2828.HK` 共存期：策略總報酬 +107.5%，benchmark +49.3%，總超額 +58.1%
- Gate：總報酬、年化報酬、Sharpe 全部勝過 `2800.HK` 與 `2828.HK`

注意：此策略為高曝險長倉動量，和 day trade 的低曝險 short mean reversion 不同；它能打贏 benchmark，但回撤也明顯高於 day trade。

## Benchmark Beater Day Trade Companion

Benchmark Beater 的日內 companion 報告頁由 `python benchmark_beater_day_trade_report.py` 產生，輸出 `benchmark_beater_day_trade_report.html`。

這個頁面沿用 Benchmark Beater 的強勢股分數，但日內執行不是同向做多，而是 short gap-up mean reversion。原因是同向 long day trade 在目前資料窗明顯失效；日內版本只能作為短線監控/輔助，不是取代中線 benchmark gate。

- 分數：Mom20 × 3 + Trend(MA60) × 1 + RSI20 × 1.5
- 動態 Universe：Top-60
- 方向：short
- 開盤 gap：+2% 到 +8%
- Scan-K：12，Top-K：4
- TP/SL：ATR 1.0 / 0.75
- 每筆部位：8%
- 當日觸發 TP/SL 或收盤前強制出場

最新 artifacts（報表日 2026-05-26）：

- 年化報酬：1.08%
- Sharpe：0.25
- 最大回撤：-6.3%
- 交易數：939

注意：此 companion 未納入借券費與可借券性；實盤前必須檢查標的可 short 的工具與成本。

## Sector Rotation v2

三層架構：

```text
Layer 1: 美股 Macro Regime
  SPY trend + VIX level → 整體曝險
  SOX → 港股互聯網/科技風險門檻

Layer 2: 港股板塊資金流
  10/15/20d 平均報酬加權排名
  取前 3 板塊

Layer 3: 板塊內選股
  momentum(20d) × 2 + trend(close > MA60) × 1
  每板塊 Top-3
```

港股板塊：

- 互聯網/科技
- 金融
- 地產/收租
- 消費/博彩
- 醫藥/生技
- 能源/原材料
- 汽車/工業
- 電訊/公用

## 驗證門檻

把台股參數搬到港股後，必須重新驗證：

- `python ai_report.py --start-date 2019-01-01 --eval-start 2021-01-01`
- `python sector_rotation_report.py --start-date 2019-01-01 --compare`
- `python walk_forward.py`
- `python walk_forward_nested.py --quick`
- `python deep_crisis_test.py`

README 裡的績效數字應只使用港股 artifacts 更新，不沿用台股歷史結果。

## Day Trade v1.1

Day trade 報告頁由 `python day_trade_report.py` 產生，輸出 `day_trade_report.html`。

目前預設不是把中線動量硬拿來當日內做多，而是使用 short gap-up mean reversion：

- t-1 收盤後產生 watchlist
- t 日開盤若跳高 +3% 到 +7.5%，才允許進場
- 方向：short
- 動態 Universe：Top-35
- MA：30
- RSI 權重：1.0
- Scan-K：4，Top-K：4
- Threshold：2.9
- Gap bonus：0.2
- TP/SL：ATR 1.3 / 0.8
- 每筆部位：8%
- 當日觸發 TP/SL 或收盤前強制出場

v1.1 是經過 gate 後升級的參數，而不是單純挑單次最佳回測：

- 舊 day trade baseline：年化 2.06%、Sharpe 0.50、最大回撤 -5.0%、821 筆
- v1.1 full window：年化 3.00%、Sharpe 1.06、最大回撤 -4.1%、304 筆
- v1.1 2025+ OOS：年化 5.68%、Sharpe 1.77、最大回撤 -1.8%、158 筆
- v1.1 pre-2025：年化為正，避免只吃近期行情
- 驗證紀錄：`artifacts/day_trade_gate_validation.json`、`artifacts/day_trade_optimization_sensitivity.csv`

注意：short day trade 需要確認標的可融券、可借券或有可用衍生工具；目前回測沒有納入借券費與借券可得性。
