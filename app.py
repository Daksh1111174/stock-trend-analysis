# app.py

import streamlit as st
import matplotlib.pyplot as plt
from data_loader import fetch_stock_data
from spark_processing import process_stock_data

st.set_page_config(page_title="Stock Trend Analysis", layout="wide")

st.title("📈 Stock Market Trend Analysis")

ticker = st.text_input("Enter Stock Ticker", "AAPL")

if st.button("Analyze"):

    with st.spinner("Fetching Data..."):
        data = fetch_stock_data(ticker)

    with st.spinner("Processing Data..."):
        final_df = process_stock_data(data)

    st.subheader("Stock Price & Moving Averages")

    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(final_df["Date"], final_df["Close"], label="Close Price")
    ax.plot(final_df["Date"], final_df["MA20"], label="MA20")
    ax.plot(final_df["Date"], final_df["MA50"], label="MA50")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Key Metrics")
    col1, col2, col3 = st.columns(3)

    col1.metric("Latest Price", round(final_df["Close"].iloc[-1],2))
    col2.metric("Volatility (20d)", round(final_df["Volatility"].iloc[-1],4))
    col3.metric("Current Trend", final_df["Trend"].iloc[-1])
