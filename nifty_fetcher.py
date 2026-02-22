# nifty_fetcher.py

import pandas as pd

def get_nifty_100_from_wikipedia():
    url = "https://en.wikipedia.org/wiki/NIFTY_100"
    tables = pd.read_html(url)

    for table in tables:
        if "Symbol" in table.columns:
            symbols = table["Symbol"].tolist()
            return [s + ".NS" for s in symbols]

    return []
