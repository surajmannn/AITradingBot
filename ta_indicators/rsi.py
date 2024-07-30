""" Functionality for RSI Technical Indicator using ta library """

import ta
import time
import pandas as pd
import yfinance as yf
import yahoo_fin.stock_info as si

# Function returns ADX, DI+, DI- values. Takes security, trading interval, and period as parameters
def rsi(ticker, interval, period_length):

    # 1m interval max 7d period on API call
    if (interval in ('1m', '2m', '5m', '10m', '15m')):
        data_period = '1d'
    else:
        data_period = '1mo'

    # security to be used
    symbol = ticker

    # set window period from input parameter
    window = period_length

    # Retrieve security data from yfinance api
    stock_data = yf.download(symbol, interval=interval, period=data_period, progress=False)

    # Get RSI values from RSI function in ta library
    rsi = ta.momentum.rsi(stock_data['Close'], window)

    # Get latest RSI value
    current_rsi = rsi[-1]

    return current_rsi