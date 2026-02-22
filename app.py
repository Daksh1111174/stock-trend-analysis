import streamlit as st
import yfinance as yf
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window
import matplotlib.pyplot as plt

st.title("📈 Big Data Stock Trend Analysis")

ticker = st.text_input("Enter Stock Ticker", "AAPL")

if st.button("Analyze"):

    data = yf.download(ticker, start="2015-01-01")
    data.reset_index(inplace=True)

    spark = SparkSession.builder.appName("StockApp").getOrCreate()
    spark_df = spark.createDataFrame(data)

    windowSpec20 = Window.orderBy("Date").rowsBetween(-20, 0)
    windowSpec50 = Window.orderBy("Date").rowsBetween(-50, 0)

    spark_df = spark_df.withColumn("MA20", avg("Close").over(windowSpec20))
    spark_df = spark_df.withColumn("MA50", avg("Close").over(windowSpec50))

    final_df = spark_df.toPandas()

    st.subheader("Stock Price with Moving Averages")

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(final_df['Date'], final_df['Close'], label='Close Price')
    ax.plot(final_df['Date'], final_df['MA20'], label='MA20')
    ax.plot(final_df['Date'], final_df['MA50'], label='MA50')
    ax.legend()

    st.pyplot(fig)

    latest_price = final_df['Close'].iloc[-1]
    st.metric("Latest Price", round(latest_price,2))
