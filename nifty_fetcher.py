# nifty_fetcher.py

import pandas as pd

# ---- Fallback Static NIFTY 100 List (Safe Version) ----
FALLBACK_NIFTY_100 = [
    "RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS",
    "ITC.NS","SBIN.NS","BHARTIARTL.NS","KOTAKBANK.NS",
    "LT.NS","AXISBANK.NS","ASIANPAINT.NS","MARUTI.NS",
    "TATAMOTORS.NS","BAJFINANCE.NS","WIPRO.NS",
    "HCLTECH.NS","SUNPHARMA.NS","VEDL.NS",
    "NBCC.NS","BEL.NS"
]

def get_nifty_100_from_wikipedia():
    try:
        url = "https://en.wikipedia.org/wiki/NIFTY_100"
        tables = pd.read_html(url)

        for table in tables:
            if "Symbol" in table.columns:
                symbols = table["Symbol"].dropna().tolist()
                return [s + ".NS" for s in symbols]

        return FALLBACK_NIFTY_100

    except:
        # If Wikipedia blocked → use fallback list
        return FALLBACK_NIFTY_100
