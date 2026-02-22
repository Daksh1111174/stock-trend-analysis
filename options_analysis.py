# options_analysis.py

import yfinance as yf
import pandas as pd

def get_option_chain(ticker="^NSEI"):

    stock = yf.Ticker(ticker)

    expiries = stock.options
    if not expiries:
        return None

    expiry = expiries[0]
    opt = stock.option_chain(expiry)

    calls = opt.calls
    puts = opt.puts

    return calls, puts, expiry
