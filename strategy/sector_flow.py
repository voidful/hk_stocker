"""
板塊資金流動量傾斜模組 (Sector-Flow Momentum Tilt)

用 10/15/20 天窗口計算各板塊的平均動量，識別資金正在流入的板塊，
動態調整 Top-K 選股的板塊配額，讓策略跟隨資金流方向。

核心概念：
- 計算每個板塊在 universe 內所有股票的平均動量
- 用三個窗口（10d/15d/20d）做加權平均，捕捉不同時間尺度的資金流
- 將 Top-K 的 slot 配額向強勢板塊傾斜
- 當板塊間差異不大時，自動退化為原始純分數排名

v1.0 — 2026-04-09
"""

import pandas as pd
import numpy as np


# =============================================
# 港股板塊分類（以預設 universe 的代表股做明確映射）
# =============================================
SECTOR_MAP = {
    'internet_tech': {
        'tickers': (
            '0700', '9988', '3690', '9618', '1024', '9999', '1810', '9888',
            '9992', '2015', '0268', '0772', '0241', '1833', '6618', '6690',
        ),
        'label': '互聯網/科技',
        'description': '平台、電商、雲服務、軟體、消費科技',
    },
    'financials': {
        'tickers': (
            '0005', '1299', '2318', '2628', '3888', '0388', '0939', '1398',
            '3988', '1288', '3328', '3968', '2388', '0011', '1658',
        ),
        'label': '金融',
        'description': '銀行、保險、交易所、券商',
    },
    'property': {
        'tickers': (
            '0016', '0017', '0083', '0101', '0688', '1109', '1113', '1997',
            '0683', '0823', '0960', '1209',
        ),
        'label': '地產/收租',
        'description': '地產發展、物管、REIT、收租股',
    },
    'consumer': {
        'tickers': (
            '2020', '2331', '2319', '0291', '0669', '1928', '1128', '2282',
            '6862', '9987', '9633', '1876', '0763',
        ),
        'label': '消費/博彩',
        'description': '運動服飾、食品飲料、餐飲、澳門博彩',
    },
    'healthcare': {
        'tickers': (
            '1093', '1177', '2269', '2359', '6160', '1801', '3692', '9926',
            '1548', '9969', '1066', '1515',
        ),
        'label': '醫藥/生技',
        'description': '藥企、CRO、創新藥、醫療服務',
    },
    'energy_materials': {
        'tickers': (
            '0883', '0857', '0386', '1088', '1171', '1898', '0914', '3323',
            '2600', '2899', '3993', '1208', '1772', '0968',
        ),
        'label': '能源/原材料',
        'description': '石油、煤炭、金屬、水泥、光伏材料',
    },
    'autos_industrials': {
        'tickers': (
            '1211', '0175', '9866', '9868', '2333', '2238', '0489', '2018',
            '2382', '2313', '0981', '1816', '1919', '1448',
        ),
        'label': '汽車/工業',
        'description': '新能源車、汽車、硬體、物流、工業製造',
    },
    'telecom_utilities': {
        'tickers': (
            '0941', '0762', '0728', '0002', '0003', '0006', '1038', '0836',
            '0267', '0012', '0853', '0001',
        ),
        'label': '電訊/公用',
        'description': '電訊營運商、公用事業、基建、綜合企業',
    },
}

# 建立快速查表：ticker → sector_name
_TICKER_TO_SECTOR = {}
for sector_name, info in SECTOR_MAP.items():
    for ticker in info['tickers']:
        _TICKER_TO_SECTOR[ticker] = sector_name


def classify_sector(ticker):
    """
    根據港股代號查表判斷板塊。

    Parameters
    ----------
    ticker : str
        港股代號，例如 '0700'

    Returns
    -------
    str
        板塊名稱（如 'internet_tech'），若無法分類則回傳 'other'
    """
    ticker_str = str(ticker).zfill(4)
    return _TICKER_TO_SECTOR.get(ticker_str, 'other')


def compute_sector_flow(close_df, universe_mask=None, windows=None,
                        weights=None):
    """
    計算各板塊在多個時間窗口的動量分數。

    每個板塊的動量 = universe 內該板塊所有股票的平均 return (close[t] / close[t-w] - 1)。
    多窗口加權平均後，做橫向排名產出板塊相對強度。

    Parameters
    ----------
    close_df : pd.DataFrame
        收盤價矩陣 (日期 × 股票代號)
    universe_mask : pd.DataFrame (bool), optional
        動態 Universe 遮罩
    windows : list[int], optional
        動量計算窗口（預設 [10, 15, 20]）
    weights : list[float], optional
        各窗口權重（預設 [0.3, 0.4, 0.3]）

    Returns
    -------
    sector_flow_df : pd.DataFrame
        (日期 × 板塊) 的動量分數矩陣
    sector_composition : dict
        {sector_name: [tickers]} 各板塊的股票組成
    """
    if windows is None:
        windows = [10, 15, 20]
    if weights is None:
        weights = [0.3, 0.4, 0.3]

    assert len(windows) == len(weights), "windows 和 weights 長度必須相同"

    # 1. 分類所有股票到板塊
    sector_tickers = {}
    for ticker in close_df.columns:
        sector = classify_sector(ticker)
        if sector not in sector_tickers:
            sector_tickers[sector] = []
        sector_tickers[sector].append(ticker)

    # 2. 計算每個窗口、每個板塊的平均動量
    sector_names = sorted(sector_tickers.keys())
    dates = close_df.index

    # 預計算各窗口的回報率矩陣
    window_returns = {}
    for w in windows:
        window_returns[w] = close_df / close_df.shift(w) - 1

    # 初始化結果矩陣
    sector_flow_data = {s: np.full(len(dates), np.nan) for s in sector_names}

    for i in range(max(windows), len(dates)):
        for sector, tickers in sector_tickers.items():
            # 取得 universe 內的股票
            if universe_mask is not None:
                valid_tickers = [t for t in tickers
                                 if t in universe_mask.columns
                                 and universe_mask[t].iloc[i]]
            else:
                valid_tickers = tickers

            if not valid_tickers:
                continue

            # 加權平均各窗口的板塊動量
            weighted_flow = 0.0
            total_weight = 0.0
            for w, wt in zip(windows, weights):
                rets = window_returns[w][valid_tickers].iloc[i]
                valid_rets = rets.dropna()
                if len(valid_rets) > 0:
                    weighted_flow += valid_rets.mean() * wt
                    total_weight += wt

            if total_weight > 0:
                sector_flow_data[sector][i] = weighted_flow / total_weight

    sector_flow_df = pd.DataFrame(sector_flow_data, index=dates)

    return sector_flow_df, sector_tickers


def get_sector_slots(sector_scores, top_k=7, tilt_strength=1.0,
                     min_dispersion=0.005):
    """
    根據板塊動量分數分配 Top-K 的 slot 配額。

    Parameters
    ----------
    sector_scores : pd.Series
        當日各板塊的動量分數（index=板塊名）
    top_k : int
        總 slot 數
    tilt_strength : float
        傾斜力度 (0.0=均分, 1.0=全力傾斜)
    min_dispersion : float
        板塊分數標準差低於此值時，退化為均分（無明顯方向）

    Returns
    -------
    dict
        {sector_name: slot_count} 每個板塊的建議 slot 數
    """
    valid_scores = sector_scores.dropna()
    if len(valid_scores) == 0:
        return {}

    # 檢查板塊間是否有明顯差異
    score_std = valid_scores.std()
    if score_std < min_dispersion:
        # 差異太小，退化為不限制（回傳空 dict 讓 caller 用原始邏輯）
        return {}

    # 混合策略：tilt_strength 控制傾斜比例
    # 1. 計算排名（越高越好）
    ranks = valid_scores.rank(ascending=True)  # 1=最差, N=最好
    n_sectors = len(ranks)

    # 2. 將排名轉為權重
    # softmax-like: 讓排名差異轉為 slot 分配
    rank_weights = ranks / ranks.sum()

    # 3. 均分權重
    uniform_weights = pd.Series(1.0 / n_sectors, index=valid_scores.index)

    # 4. 混合
    blended = tilt_strength * rank_weights + (1 - tilt_strength) * uniform_weights
    blended = blended / blended.sum()  # 歸一化

    # 5. 分配 slot（按權重比例分配，取 floor 後把剩餘給最強板塊）
    raw_slots = blended * top_k
    floor_slots = raw_slots.apply(np.floor).astype(int)

    # 確保至少分配完所有 slot
    remaining = top_k - floor_slots.sum()
    if remaining > 0:
        # 按小數部分大小分配剩餘 slot
        fractional = raw_slots - floor_slots
        top_frac = fractional.nlargest(int(remaining))
        for sector in top_frac.index:
            floor_slots[sector] += 1

    return floor_slots.to_dict()


def select_with_sector_tilt(candidates, sector_slots, top_k, slots_available):
    """
    按板塊配額從候選股中選股。

    Parameters
    ----------
    candidates : list[tuple]
        已按分數排序的候選股 [(ticker, score, entry_price), ...]
    sector_slots : dict
        {sector_name: max_slots} 板塊配額
    top_k : int
        目標選股數
    slots_available : int
        實際可用 slot（扣除已持倉）

    Returns
    -------
    list[tuple]
        選中的候選股列表
    """
    effective_k = min(top_k, slots_available)

    if not sector_slots:
        # 無傾斜，退化為原始行為
        return candidates[:effective_k]

    selected = []
    sector_used = {}  # 追蹤各板塊已用 slot

    for ticker, score, entry_price in candidates:
        if len(selected) >= effective_k:
            break

        sector = classify_sector(ticker)
        used = sector_used.get(sector, 0)
        max_allowed = sector_slots.get(sector, 1)  # 未在 map 中的板塊給 1 slot

        if used < max_allowed:
            selected.append((ticker, score, entry_price))
            sector_used[sector] = used + 1

    # 如果因為配額限制導致 slot 沒填滿，用剩餘最高分候選補上
    if len(selected) < effective_k:
        selected_tickers = {s[0] for s in selected}
        for ticker, score, entry_price in candidates:
            if len(selected) >= effective_k:
                break
            if ticker not in selected_tickers:
                selected.append((ticker, score, entry_price))
                selected_tickers.add(ticker)

    return selected
