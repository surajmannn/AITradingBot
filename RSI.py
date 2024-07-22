""" Functionality for RSI Technical Indicator """

import yfinance as yf
import time

# Function returns RSI values, takes trading interval, look back period and start date as parameters
def RSI(ticker, interval, period_length):

    # 1m interval max 7d period on API call
    if (interval in ('1m', '2m', '5m', '10m', '15m')):
        data_period = '2d'
    else:
        data_period = '30d'

    # Set the ticker security 
    security = ticker

    # Set the RSI window length
    window_length = period_length

    # Retrieve the stock data from Yahoo Finance
    stock_data = yf.download(security, interval=interval, period=data_period, progress=False)

    # Calculate the RSI
    close = stock_data['Close']
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window_length).mean()
    avg_loss = loss.rolling(window_length).mean().abs()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Get latest RSI value and time stamp
    last_rsi = rsi.iloc[-1]
    last_time = rsi.index[-1]
    #(f"\nRSI at {last_time}: {last_rsi:.2f}")

    # return latest RSI Value
    return last_rsi, last_time