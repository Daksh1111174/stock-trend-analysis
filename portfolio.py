# portfolio.py

import numpy as np
import pandas as pd
import yfinance as yf

def get_portfolio_data(tickers, start_date):

    data = yf.download(tickers, start=start_date, progress=False)["Close"]

    # If single stock selected, convert to DataFrame
    if isinstance(data, pd.Series):
        data = data.to_frame()

    # Forward fill missing values
    data = data.fillna(method="ffill")

    # Drop columns with too many NaNs
    data = data.dropna(axis=1, thresh=int(len(data)*0.7))

    returns = data.pct_change().dropna()

    return data, returns


def optimize_portfolio(returns, simulations=3000):

    if returns.shape[1] < 2:
        # Not enough assets for frontier
        return np.array([1.0]), np.zeros((3, 1))

    num_assets = returns.shape[1]

    results = np.zeros((3, simulations))
    weights_record = []

    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252

    for i in range(simulations):

        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        weights_record.append(weights)

        portfolio_return = np.dot(weights, mean_returns)

        portfolio_std = np.sqrt(
            np.dot(weights.T, np.dot(cov_matrix, weights))
        )

        sharpe_ratio = (
            portfolio_return / portfolio_std
            if portfolio_std != 0 else 0
        )

        results[0, i] = portfolio_return
        results[1, i] = portfolio_std
        results[2, i] = sharpe_ratio

    max_sharpe_idx = np.argmax(results[2])
    optimal_weights = weights_record[max_sharpe_idx]

    return optimal_weights, results
