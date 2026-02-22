# portfolio.py

import numpy as np
import pandas as pd
import yfinance as yf

def get_portfolio_data(tickers, start_date):
    data = yf.download(tickers, start=start_date, progress=False)["Close"]
    returns = data.pct_change().dropna()
    return data, returns

def optimize_portfolio(returns, simulations=5000):
    num_assets = returns.shape[1]
    results = np.zeros((3, simulations))
    weights_record = []

    for i in range(simulations):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        weights_record.append(weights)

        portfolio_return = np.sum(returns.mean() * weights) * 252
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(returns.cov()*252, weights)))
        sharpe_ratio = portfolio_return / portfolio_std

        results[0,i] = portfolio_return
        results[1,i] = portfolio_std
        results[2,i] = sharpe_ratio

    max_sharpe_idx = np.argmax(results[2])
    optimal_weights = weights_record[max_sharpe_idx]

    return optimal_weights, results
