# data_loader.py

import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker: str, start_date="2015-01-01"):
    """
    Fetch stock data safely for Streamlit Cloud
    """
    try:
        data = yf.download(
            ticker,
            start=start_date,
            progress=False,
            threads=False
        )

        if data is None or data.empty:
            return None

        data.reset_index(inplace=True)
        return data

    except Exception as e:
        print("Data Fetch Error:", e)
        return None
