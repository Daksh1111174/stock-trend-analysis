# options_analysis.py

import yfinance as yf
import pandas as pd

def get_option_chain(ticker="^NSEI"):

    try:
        stock = yf.Ticker(ticker)

        expiries = stock.options
        if not expiries:
            return None, None, None

        expiry = expiries[0]
        opt = stock.option_chain(expiry)

        calls = opt.calls
        puts = opt.puts

        return calls, puts, expiry

    except Exception as e:
        print("Option Chain Error:", e)
        return None, None, None
