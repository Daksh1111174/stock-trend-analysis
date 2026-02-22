# sector_analysis.py

import yfinance as yf
import pandas as pd

def get_sector_data(tickers):

    sector_data = []

    for t in tickers:
        try:
            stock = yf.Ticker(t)
            info = stock.info
            sector = info.get("sector", "Unknown")
            market_cap = info.get("marketCap", 0)

            sector_data.append({
                "Ticker": t,
                "Sector": sector,
                "MarketCap": market_cap
            })
        except:
            continue

    df = pd.DataFrame(sector_data)
    return df
