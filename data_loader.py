# data_loader.py

import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker: str, start_date="2015-01-01"):
    """
    Fetch historical stock data using yfinance
    """
    try:
        data = yf.download(ticker, start=start_date)
        if data.empty:
            raise ValueError("No data found for ticker.")
        data.reset_index(inplace=True)
        return data
    except Exception as e:
        raise RuntimeError(f"Error fetching data: {e}")
