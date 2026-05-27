"""
港股籌碼因子模組 (Institutional Flow Factor)

tw_stocker 原版使用台股三大法人資料；港股沒有相同欄位與公開資料源。
本模組保留相同函式介面，預設回傳空資料，讓主策略與報表流程維持相容。
"""

import json
import urllib.request
import pandas as pd
import numpy as np
from datetime import datetime
from functools import lru_cache

BASE_URL = None
TIMEOUT = 15


def _fetch_json(url):
    """從 URL 抓 JSON，失敗回 None。"""
    if not url:
        return None
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'hk_stocker/1.0'})
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return json.loads(resp.read().decode('utf-8'))
    except Exception as e:
        print(f"   ⚠️ 抓取失敗 {url}: {e}")
        return None


def fetch_inst_timeseries(ticker):
    """
    抓取單檔股票的三大法人持股時序。

    Parameters
    ----------
    ticker : str
        股票代號，例如 '0700'

    Returns
    -------
    list[dict] or None
        時序資料列表，每筆含 date, foreign_ratio, trust_ratio,
        dealer_ratio, three_inst_ratio, three_inst_ratio_change_20
    """
    return None


def fetch_inst_rankings(window=20, direction='up'):
    """
    抓取三大法人持股變化排名表。

    Parameters
    ----------
    window : int
        變化視窗天數 (5, 20, 60, 120)
    direction : str
        'up' 或 'down'

    Returns
    -------
    list[dict] or None
        排名列表，每筆含 code, name, market, three_inst_ratio, change
    """
    return None


def build_inst_flow_df(tickers, close_df, verbose=True):
    """
    批次抓取多檔股票的籌碼時序，構建 DataFrame 對齊到 close_df。

    Parameters
    ----------
    tickers : list[str]
        股票代號列表
    close_df : pd.DataFrame
        收盤價矩陣 (date × ticker)，用於日期對齊
    verbose : bool
        是否印出進度

    Returns
    -------
    inst_flow_df : pd.DataFrame
        三大法人 20 日持股變化矩陣 (date × ticker)
    inst_ratio_df : pd.DataFrame
        三大法人持股比重矩陣 (date × ticker)
    """
    if verbose:
        print("🏛️ 港股籌碼資料源尚未接入，略過籌碼因子")

    empty = pd.DataFrame(np.nan, index=close_df.index, columns=close_df.columns)
    return empty, empty.copy()


def get_inst_flow_for_signals(tickers, window=20):
    """
    為即時信號取得三大法人籌碼標注。
    用排名表做快速查詢（不需抓全部時序）。

    Parameters
    ----------
    tickers : list[str]
        候選股票代號
    window : int
        變化視窗 (default 20)

    Returns
    -------
    dict
        {ticker: {'change': float, 'ratio': float, 'label': str}}
    """
    up_list = fetch_inst_rankings(window, 'up') or []
    down_list = fetch_inst_rankings(window, 'down') or []

    # 建立查找表
    lookup = {}
    for item in up_list:
        lookup[item['code']] = {
            'change': item.get('change', 0.0),
            'ratio': item.get('three_inst_ratio', 0.0),
        }
    for item in down_list:
        lookup[item['code']] = {
            'change': -abs(item.get('change', 0.0)),
            'ratio': item.get('three_inst_ratio', 0.0),
        }

    result = {}
    for t in tickers:
        if t in lookup:
            info = lookup[t]
            change = info['change']
            if change > 2.0:
                label = '🟢 大買'
            elif change > 0.5:
                label = '🟡 小買'
            elif change < -2.0:
                label = '🔴 大賣'
            elif change < -0.5:
                label = '🟠 小賣'
            else:
                label = '⚪ 中性'
            result[t] = {
                'change': change,
                'ratio': info['ratio'],
                'label': label,
            }
        else:
            result[t] = {
                'change': 0.0,
                'ratio': 0.0,
                'label': '⚪ 無資料',
            }

    return result
