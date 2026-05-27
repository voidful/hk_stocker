"""Hong Kong market configuration shared by the strategy modules."""

MARKET_NAME_ZH = "港股"
MARKET_NAME_EN = "Hong Kong stocks"
PROJECT_NAME = "HK Stocker"
YAHOO_SUFFIX = ".HK"
EXCHANGE_CALENDAR = "XHKG"
CURRENCY = "HKD"

# Tracker Fund of Hong Kong; tradable Hang Seng proxy on HKEX.
DEFAULT_BENCHMARK = "2800"
DEFAULT_BENCHMARK_LABEL = "2800.HK"

# Hang Seng China Enterprises ETF; secondary benchmark roughly analogous to the
# Extra benchmark panel roughly analogous to the Taiwan version.
SECONDARY_BENCHMARK = "2828"
SECONDARY_BENCHMARK_LABEL = "2828.HK"

# Approximate exchange-side round-trip model, excluding broker commission.
# Per side: stamp duty 0.10% + SFC 0.0027% + AFRC 0.00015%
# + HKEX trading fee 0.00565% + CCASS settlement 0.0042%.
HKEX_SIDE_COST = 0.001127
DEFAULT_BUY_COST = HKEX_SIDE_COST
DEFAULT_SELL_COST = HKEX_SIDE_COST


def normalize_ticker(ticker):
    """Return a canonical HKEX stock code used inside this project."""
    ticker_str = str(ticker).strip().upper()
    if ticker_str.endswith(YAHOO_SUFFIX):
        ticker_str = ticker_str[:-len(YAHOO_SUFFIX)]
    if ticker_str.startswith("^"):
        return ticker_str
    if ticker_str.isdigit() and len(ticker_str) <= 4:
        return ticker_str.zfill(4)
    return ticker_str


def to_yahoo_symbol(ticker):
    """Convert a project ticker such as 0700 into Yahoo Finance 0700.HK."""
    ticker_str = str(ticker).strip().upper()
    if ticker_str.startswith("^") or ticker_str.endswith(YAHOO_SUFFIX):
        return ticker_str
    return f"{normalize_ticker(ticker_str)}{YAHOO_SUFFIX}"


def from_yahoo_symbol(symbol):
    """Convert a Yahoo Finance symbol back into the project ticker format."""
    symbol_str = str(symbol).strip().upper()
    if symbol_str.endswith(YAHOO_SUFFIX):
        symbol_str = symbol_str[:-len(YAHOO_SUFFIX)]
    return normalize_ticker(symbol_str)
