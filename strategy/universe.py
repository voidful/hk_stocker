"""Default Hong Kong stock universes."""

from strategy.market import normalize_ticker


DEFAULT_TICKERS = [
    "0700", "9988", "3690", "9618", "1024",
    "0005", "1299", "0939", "1398", "3988",
    "0883", "0941", "1211", "2020",
]


EXTENDED_TICKERS = [
    # Internet / platform / software
    "0700", "9988", "3690", "9618", "1024", "9999", "1810", "9888",
    "9992", "2015", "0268", "0772", "0241", "1833", "6618", "6690",
    # Financials / exchanges / insurers
    "0005", "1299", "2318", "2628", "3888", "0388", "0939", "1398",
    "3988", "1288", "3328", "3968", "2388", "0011", "1658",
    # Property / landlords / REITs
    "0016", "0017", "0083", "0101", "0688", "1109", "1113", "1997",
    "0683", "0823", "0960", "1209",
    # Consumer / Macau / apparel / food
    "2020", "2331", "2319", "0291", "0669", "1928", "1128", "2282",
    "6862", "9987", "9633", "1876", "0763",
    # Healthcare / biotech
    "1093", "1177", "2269", "2359", "6160", "1801", "3692", "9926",
    "1548", "9969", "1066", "1515",
    # Energy / materials
    "0883", "0857", "0386", "1088", "1171", "1898", "0914", "3323",
    "2600", "2899", "3993", "1208", "1772", "0968",
    # Autos / industrials / hardware
    "1211", "0175", "9866", "9868", "2333", "2238", "0489", "2018",
    "2382", "2313", "0981", "1816", "1919", "1448",
    # Telecom / utilities / infrastructure
    "0941", "0762", "0728", "0002", "0003", "0006", "1038", "0836",
    "0267", "0012", "0853", "0001",
]


DEFAULT_TICKERS = list(dict.fromkeys(normalize_ticker(t) for t in DEFAULT_TICKERS))
EXTENDED_TICKERS = list(dict.fromkeys(normalize_ticker(t) for t in EXTENDED_TICKERS))
