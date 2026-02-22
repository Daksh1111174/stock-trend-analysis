# fundamentals.py

import yfinance as yf

def get_fundamentals(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info

    fundamentals = {
        "Market Cap": info.get("marketCap"),
        "PE Ratio": info.get("trailingPE"),
        "EPS": info.get("trailingEps"),
        "Revenue": info.get("totalRevenue"),
        "Debt to Equity": info.get("debtToEquity")
    }

    return fundamentals
