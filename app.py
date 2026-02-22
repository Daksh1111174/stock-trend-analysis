# app.py

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

from portfolio import get_portfolio_data, optimize_portfolio
from risk_models import monte_carlo_simulation
from lstm_model import train_lstm
from fundamentals import get_fundamentals
from pdf_report import generate_pdf

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(page_title="Stock Analytics Pro", layout="wide")

# ---------------------------------------------------
# GLASSMORPHISM UI
# ---------------------------------------------------

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg,#1f1c2c,#928dab);
    color: white;
}
.block-container {
    background: rgba(255,255,255,0.1);
    padding: 20px;
    border-radius: 20px;
    backdrop-filter: blur(15px);
}
</style>
""", unsafe_allow_html=True)

st.title("📊 Stock Analytics Pro")
st.caption("Portfolio Optimization • Risk Modeling • Deep Learning Forecasting")

# ---------------------------------------------------
# INPUT SECTION
# ---------------------------------------------------

tickers_input = st.text_input("Enter Stock Tickers (comma separated)", "AAPL,MSFT,TSLA")
start_date = st.date_input("Start Date", datetime(2018,1,1))

tickers = [t.strip() for t in tickers_input.split(",")]

# ---------------------------------------------------
# MAIN ANALYSIS BUTTON
# ---------------------------------------------------

if st.button("Run Full Analysis"):

    # ---------------------------------------------------
    # FETCH DATA
    # ---------------------------------------------------
    data, returns = get_portfolio_data(tickers, start_date)

    if data.empty:
        st.error("Invalid tickers or no data found.")
        st.stop()

    # ---------------------------------------------------
    # PORTFOLIO OPTIMIZATION
    # ---------------------------------------------------

    optimal_weights, results = optimize_portfolio(returns)

    st.subheader("📈 Portfolio Optimization (Max Sharpe)")

    weights_df = pd.DataFrame({
        "Stock": tickers,
        "Optimal Weight": optimal_weights
    })

    st.dataframe(weights_df)

    # Efficient Frontier Plot
    ef_fig = go.Figure()
    ef_fig.add_trace(go.Scatter(
        x=results[1],
        y=results[0],
        mode='markers',
        marker=dict(
            size=5,
            color=results[2],
            colorscale='Viridis',
            showscale=True
        )
    ))

    ef_fig.update_layout(
        title="Efficient Frontier",
        xaxis_title="Volatility",
        yaxis_title="Expected Return",
        template="plotly_dark"
    )

    st.plotly_chart(ef_fig, use_container_width=True)

    # ---------------------------------------------------
    # RISK METRICS (Single First Stock)
    # ---------------------------------------------------

    stock = tickers[0]
    close_prices = data[stock]
    daily_returns = close_prices.pct_change().dropna()

    VaR = np.percentile(daily_returns, 5)
    sharpe_ratio = (daily_returns.mean()/daily_returns.std()) * np.sqrt(252)

    col1, col2 = st.columns(2)
    col1.metric("Value at Risk (95%)", f"{VaR:.4f}")
    col2.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")

    # ---------------------------------------------------
    # MONTE CARLO SIMULATION
    # ---------------------------------------------------

    st.subheader("🎲 Monte Carlo Simulation (1 Year Projection)")

    S0 = close_prices.iloc[-1]
    mu = daily_returns.mean() * 252
    sigma = daily_returns.std() * np.sqrt(252)

    simulations = monte_carlo_simulation(S0, mu, sigma)

    mc_fig = go.Figure()

    for i in range(50):  # plot 50 paths only
        mc_fig.add_trace(go.Scatter(
            y=simulations[:,i],
            mode='lines',
            line=dict(width=1),
            showlegend=False
        ))

    mc_fig.update_layout(
        template="plotly_dark",
        title="Monte Carlo Price Simulation"
    )

    st.plotly_chart(mc_fig, use_container_width=True)

    # ---------------------------------------------------
    # LSTM FORECAST
    # ---------------------------------------------------

    st.subheader("🤖 LSTM Deep Learning Forecast")

    model, scaler = train_lstm(close_prices.values)

    last_60 = close_prices.values[-60:]
    last_60_scaled = scaler.transform(last_60.reshape(-1,1))

    X_test = np.array([last_60_scaled])
    prediction_scaled = model.predict(X_test)
    prediction = scaler.inverse_transform(prediction_scaled)

    st.metric("Next Day Predicted Price", f"${prediction[0][0]:.2f}")

    # ---------------------------------------------------
    # COMPANY FUNDAMENTALS
    # ---------------------------------------------------

    st.subheader("🏢 Company Fundamentals")

    fundamentals = get_fundamentals(stock)

    fundamentals_df = pd.DataFrame(
        fundamentals.items(),
        columns=["Metric","Value"]
    )

    st.dataframe(fundamentals_df)

    # ---------------------------------------------------
    # EXECUTIVE PDF REPORT
    # ---------------------------------------------------

    st.subheader("📄 Download Executive Report")

    metrics_dict = {
        "Stock": stock,
        "VaR (95%)": VaR,
        "Sharpe Ratio": sharpe_ratio,
        "Predicted Price": prediction[0][0]
    }

    generate_pdf("report.pdf", metrics_dict)

    with open("report.pdf", "rb") as f:
        st.download_button(
            "Download PDF Report",
            f,
            file_name="Stock_Report.pdf"
        )

    st.success("Analysis Complete 🚀")
