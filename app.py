# app.py

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime

from risk_models import monte_carlo_simulation
from lstm_model import train_lstm
from fundamentals import get_fundamentals
from pdf_report import generate_pdf
from options_analysis import get_option_chain

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(page_title="🇮🇳 Indian Stock Risk & AI Analyzer", layout="wide")

# ---------------------------------------------------
# UI STYLE
# ---------------------------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
    color: white;
}
.block-container {
    background: rgba(255,255,255,0.08);
    padding: 25px;
    border-radius: 20px;
    backdrop-filter: blur(15px);
}
</style>
""", unsafe_allow_html=True)

st.title("🇮🇳 Indian Stock Risk & AI Analyzer")
st.caption("Monte Carlo • Risk Metrics • AI Forecast • Fundamentals")

# ---------------------------------------------------
# STOCK SELECTION (Single Only)
# ---------------------------------------------------
INDIAN_STOCKS = {
    "Reliance Industries": "RELIANCE.NS",
    "Vedanta": "VEDL.NS",
    "Tata Motors": "TATAMOTORS.NS",
    "NBCC": "NBCC.NS",
    "Bharat Electronics": "BEL.NS",
    "NIFTY 50 Index": "^NSEI"
}

selected_stock_name = st.selectbox(
    "Select Stock",
    list(INDIAN_STOCKS.keys())
)

ticker = INDIAN_STOCKS[selected_stock_name]
start_date = st.date_input("Start Date", datetime(2018,1,1))

# ---------------------------------------------------
# RUN ANALYSIS
# ---------------------------------------------------
if st.button("🚀 Analyze Stock"):

    data = yf.download(ticker, start=start_date, progress=False)

    if data is None or data.empty:
        st.error("No data found for selected stock.")
        st.stop()

    close_prices = data["Close"]

    # Ensure Series
    if isinstance(close_prices, pd.DataFrame):
        close_prices = close_prices.squeeze()

    daily_returns = close_prices.pct_change().dropna()

    if len(daily_returns) < 2:
        st.warning("Not enough data to calculate risk metrics.")
        st.stop()

    # ---------------------------------------------------
    # PRICE CHART
    # ---------------------------------------------------
    st.subheader("📈 Price Chart")

    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(
        x=data.index,
        y=close_prices,
        mode='lines',
        name="Close Price"
    ))

    fig_price.update_layout(template="plotly_dark")
    st.plotly_chart(fig_price, use_container_width=True)

    # ---------------------------------------------------
    # SAFE RISK METRICS
    # ---------------------------------------------------
    st.subheader("📊 Risk Metrics")

    std_dev = float(daily_returns.std())
    mean_return = float(daily_returns.mean())

    volatility = std_dev * np.sqrt(252)
    VaR = float(np.percentile(daily_returns, 5))

    if std_dev > 0 and not np.isnan(std_dev):
        sharpe = (mean_return / std_dev) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Replace invalid values safely
    if np.isnan(volatility):
        volatility = 0.0
    if np.isnan(VaR):
        VaR = 0.0
    if np.isnan(sharpe):
        sharpe = 0.0

    col1, col2, col3 = st.columns(3)
    col1.metric("Annual Volatility", f"{volatility:.2f}")
    col2.metric("Value at Risk (95%)", f"{VaR:.4f}")
    col3.metric("Sharpe Ratio", f"{sharpe:.2f}")

    # ---------------------------------------------------
    # MONTE CARLO (CLEAN VERSION)
    # ---------------------------------------------------
    st.subheader("🎲 Monte Carlo Simulation (1 Year Projection)")

    S0 = float(close_prices.iloc[-1])
    mu = mean_return * 252
    sigma = std_dev * np.sqrt(252)

    simulations = monte_carlo_simulation(S0, mu, sigma)

    mean_path = np.mean(simulations, axis=1)
    upper_band = np.percentile(simulations, 95, axis=1)
    lower_band = np.percentile(simulations, 5, axis=1)

    fig_mc = go.Figure()

    fig_mc.add_trace(go.Scatter(
        y=mean_path,
        mode='lines',
        name='Expected Path'
    ))

    fig_mc.add_trace(go.Scatter(
        y=upper_band,
        mode='lines',
        name='95% Upper Bound',
        line=dict(dash='dash')
    ))

    fig_mc.add_trace(go.Scatter(
        y=lower_band,
        mode='lines',
        name='5% Lower Bound',
        line=dict(dash='dash')
    ))

    fig_mc.update_layout(template="plotly_dark")
    st.plotly_chart(fig_mc, use_container_width=True)

    # ---------------------------------------------------
    # LSTM FORECAST (SAFE)
    # ---------------------------------------------------
    st.subheader("🤖 AI Forecast")

    try:
        model, scaler = train_lstm(close_prices.values)

        last_60 = close_prices.values[-60:]
        last_60_scaled = scaler.transform(last_60.reshape(-1,1))
        X_test = np.array([last_60_scaled])

        prediction_scaled = model.predict(X_test)
        prediction = scaler.inverse_transform(prediction_scaled)

        st.metric("Next Day Predicted Price", f"₹{float(prediction[0][0]):.2f}")

    except:
        st.warning("LSTM model unavailable in cloud environment.")

    # ---------------------------------------------------
    # OPTION CHAIN (SAFE)
    # ---------------------------------------------------
    st.subheader("📈 Option Chain")

    try:
        result = get_option_chain(ticker)

        if result is not None:
            calls, puts, expiry = result
        else:
            calls, puts, expiry = None, None, None

        if calls is not None and not calls.empty:
            st.write(f"Nearest Expiry: {expiry}")
            st.dataframe(calls.head())
        else:
            st.warning("Option data unavailable.")

    except:
        st.warning("Option API blocked.")

    # ---------------------------------------------------
    # FUNDAMENTALS
    # ---------------------------------------------------
    st.subheader("🏢 Company Fundamentals")

    try:
        fundamentals = get_fundamentals(ticker)

        st.dataframe(pd.DataFrame(
            fundamentals.items(),
            columns=["Metric","Value"]
        ))
    except:
        st.warning("Fundamental data unavailable.")

    # ---------------------------------------------------
    # PDF REPORT
    # ---------------------------------------------------
    st.subheader("📄 Executive Report")

    try:
        metrics_dict = {
            "Stock": ticker,
            "Volatility": volatility,
            "VaR": VaR,
            "Sharpe": sharpe
        }

        generate_pdf("report.pdf", metrics_dict)

        with open("report.pdf", "rb") as f:
            st.download_button(
                "Download Report",
                f,
                file_name="Stock_Report.pdf"
            )
    except:
        st.warning("PDF generation failed.")

    st.success("Analysis Completed Successfully 🚀")
