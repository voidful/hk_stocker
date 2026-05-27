"""
新聞情緒因子模組 (News Sentiment Factor)

tw_stocker 原版使用台股新聞情緒資料；港股版先保留相同函式介面，
預設回傳中性標注，避免把台股新聞排名混入港股信號。
"""
import json
import urllib.request

TIMEOUT = 15


def _fetch_text(url, quiet=False):
    """從 URL 抓文字，失敗回 None。"""
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'hk_stocker/1.0'})
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return resp.read().decode('utf-8')
    except Exception as e:
        if not quiet:
            print(f"   ⚠️ 新聞數據抓取失敗 {url}: {e}")
        return None


def _fetch_json(url):
    """從 URL 抓 JSON，失敗回 None。"""
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'hk_stocker/1.0'})
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return json.loads(resp.read().decode('utf-8'))
    except Exception as e:
        print(f"   ⚠️ 新聞數據抓取失敗 {url}: {e}")
        return None


def fetch_news_leaderboard(days=5):
    """
    抓取新聞情緒排名 CSV。

    Parameters
    ----------
    days : int
        時間視窗 (1, 3, 5, 10, 30, 60)

    Returns
    -------
    list[dict] or None
        排名列表，每筆含 ticker, name, score 等
    """
    return None


def get_news_sentiment_for_signals(tickers, days=5):
    """
    為即時信號取得新聞情緒標注。

    Parameters
    ----------
    tickers : list[str]
        候選股票代號
    days : int
        時間視窗 (default 5 = 近 5 天)

    Returns
    -------
    dict
        {ticker: {'score': float, 'label': str}}
    """
    leaderboard = fetch_news_leaderboard(days)

    lookup = {}
    if leaderboard:
        for item in leaderboard:
            t = item.get('ticker', '')
            if t:
                lookup[t] = item.get('score', 0.0)

    result = {}
    for t in tickers:
        score = lookup.get(t, 0.0)
        if score > 3.0:
            label = '🟢 強正面'
        elif score > 1.0:
            label = '🟡 正面'
        elif score < -3.0:
            label = '🔴 強負面'
        elif score < -1.0:
            label = '🟠 負面'
        else:
            label = '⚪ 中性'

        result[t] = {
            'score': score,
            'label': label,
        }

    return result
