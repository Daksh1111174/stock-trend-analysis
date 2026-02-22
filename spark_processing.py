# spark_processing.py
# Streamlit Cloud Compatible (No Spark)

import pandas as pd
import numpy as np

def process_stock_data(df):

    # Moving averages
    df["MA20"] = df["Close"].rolling(window=20).mean()
    df["MA50"] = df["Close"].rolling(window=50).mean()

    # Daily return
    df["Daily_Return"] = (df["Close"] - df["Open"]) / df["Open"]

    # Volatility
    df["Volatility"] = df["Daily_Return"].rolling(window=20).std()

    # Trend
    df["Trend"] = np.where(df["MA20"] > df["MA50"], "Bullish", "Bearish")

    return df
