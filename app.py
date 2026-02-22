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
from nifty_fetcher import get_nifty_100_from_wikipedia
from sector_analysis import get_sector_data
from fii_dii import get_fii_dii_data
from options_analysis import get_option_chain

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(page_title="🇮🇳 Indian Market Quant Pro", layout="wide")

# ---------------------------------------------------
# PREMIUM GLASS UI
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

st.title("🇮🇳 Indian Market Quant Pro")
st.caption("NIFTY Analytics • Portfolio Optimization • Risk Modeling • AI Forecasting")

# ---------------------------------------------------
# STOCK SELECTION
# ---------------------------------------------------
st.subheader("📌 Stock Selection")

mode = st.radio("Choose Mode", ["Indian Stocks", "Custom Input", "Auto NIFTY 100"])

INDIAN_STOCKS = {
    "Reliance Industries": "RELIANCE.NS",
    "Vedanta": "VEDL.NS",
    "Tata Motors": "TATAMOTORS.NS",
    "NBCC": "NBCC.NS",
    "Bharat Electronics": "BEL.NS",
    "NIFTY 50 Index": "^NSEI"
}

if mode == "Indian Stocks":
    selected = st.multiselect(
        "Select Companies",
        list(INDIAN_STOCKS.keys()),
        default=["Reliance Industries", "Tata Motors"]
    )
    tickers = [INDIAN_STOCKS[s] for s in selected]

elif mode == "Auto NIFTY 100":
    tickers = get_nifty_100_from_wikipedia()
    st.success(f"Loaded {len(tickers)} NIFTY 100 stocks")

else:
    custom_input = st.text_input(
        "Enter NSE Tickers (.NS required)",
        "RELIANCE.NS,TATAMOTORS.NS"
    )
    tickers = [t.strip() for t in custom_input.split(",")]

start_date = st.date_input("Start Date", datetime(2018,1,1))

# ---------------------------------------------------
# RUN ANALYSIS
# ---------------------------------------------------
if st.button("🚀 Run Full Quant Analysis"):

    if not tickers:
        st.warning("Please select stocks.")
        st.stop()

    data, returns = get_portfolio_data(tickers, start_date)

    if data.empty:
        st.error("Invalid tickers or no data.")
        st.stop()

    # ---------------------------------------------------
    # PORTFOLIO OPTIMIZATION
    # ---------------------------------------------------
    st.subheader("📊 Portfolio Optimization (Efficient Frontier)")

    optimal_weights, results = optimize_portfolio(returns)

    weights_df = pd.DataFrame({
        "Stock": tickers,
        "Optimal Weight": optimal_weights
    })

    st.dataframe(weights_df)

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
        template="plotly_dark",
        xaxis_title="Volatility",
        yaxis_title="Expected Return"
    )

    st.plotly_chart(ef_fig, use_container_width=True)

    # ---------------------------------------------------
    # RISK METRICS (FIRST STOCK)
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
    st.subheader("🎲 Monte Carlo Simulation")

    S0 = close_prices.iloc[-1]
    mu = daily_returns.mean() * 252
    sigma = daily_returns.std() * np.sqrt(252)

    simulations = monte_carlo_simulation(S0, mu, sigma)

    mc_fig = go.Figure()
    for i in range(50):
        mc_fig.add_trace(go.Scatter(
            y=simulations[:,i],
            mode='lines',
            line=dict(width=1),
            showlegend=False
        ))

    mc_fig.update_layout(template="plotly_dark")
    st.plotly_chart(mc_fig, use_container_width=True)

    # ---------------------------------------------------
    # LSTM FORECAST
    # ---------------------------------------------------
    st.subheader("🤖 AI Forecast (LSTM)")

    model, scaler = train_lstm(close_prices.values)

    last_60 = close_prices.values[-60:]
    last_60_scaled = scaler.transform(last_60.reshape(-1,1))
    X_test = np.array([last_60_scaled])

    prediction_scaled = model.predict(X_test)
    prediction = scaler.inverse_transform(prediction_scaled)

    st.metric("Next Day Predicted Price", f"₹{prediction[0][0]:.2f}")

    # ---------------------------------------------------
    # SECTOR HEATMAP
    # ---------------------------------------------------
    st.subheader("🏭 Sector-wise Market Cap Heatmap")

    sector_df = get_sector_data(tickers)

    if not sector_df.empty:
        fig = px.treemap(
            sector_df,
            path=["Sector","Ticker"],
            values="MarketCap",
            color="MarketCap",
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------
    # FII / DII DATA
    # ---------------------------------------------------
    st.subheader("💰 FII / DII Activity")

    fii_df = get_fii_dii_data()

    if not fii_df.empty:
        st.dataframe(fii_df.head())
    else:
        st.warning("FII/DII data unavailable.")

    # ---------------------------------------------------
    # OPTION CHAIN
    # ---------------------------------------------------
    st.subheader("📈 NIFTY Option Chain")

    calls, puts, expiry = get_option_chain("^NSEI")

    if calls is not None:
        st.write(f"Nearest Expiry: {expiry}")

        col1, col2 = st.columns(2)

        with col1:
            st.write("Top Call OI")
            st.dataframe(calls.sort_values("openInterest", ascending=False).head())

        with col2:
            st.write("Top Put OI")
            st.dataframe(puts.sort_values("openInterest", ascending=False).head())

    # ---------------------------------------------------
    # FUNDAMENTALS
    # ---------------------------------------------------
    st.subheader("🏢 Company Fundamentals")

    fundamentals = get_fundamentals(stock)
    st.dataframe(pd.DataFrame(
        fundamentals.items(),
        columns=["Metric","Value"]
    ))

    # ---------------------------------------------------
    # PDF REPORT
    # ---------------------------------------------------
    st.subheader("📄 Download Executive Report")

    metrics_dict = {
        "Stock": stock,
        "VaR": VaR,
        "Sharpe Ratio": sharpe_ratio,
        "Predicted Price": prediction[0][0]
    }

    generate_pdf("report.pdf", metrics_dict)

    with open("report.pdf", "rb") as f:
        st.download_button(
            "Download PDF",
            f,
            file_name="Indian_Market_Report.pdf"
        )

    st.success("Quant Analysis Completed 🚀")
