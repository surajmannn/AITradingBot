""" RSI Functionality File """

import yfinance as yf
import pandas as pd
import time

def RSI(ticker, period_length):

    # Set the ticker security and date range
    security = ticker
    start_date = pd.Timestamp.today().strftime('%Y-%m-%d')

    # Set the RSI window length
    window_length = period_length

    # Wait until the next minute starts
    current_time = pd.Timestamp.now()
    next_minute = (current_time + pd.Timedelta(minutes=1)).replace(second=0, microsecond=0)
    print("Waiting for the next minute to start....")
    time.sleep((next_minute - current_time).seconds)


    while True:
        # Retrieve the data from Yahoo Finance
        stock_data = yf.download(security, start=start_date, end=None, interval="1m", progress=False)
        #print(stock_data)

        # Calculate the RSI
        close = stock_data['Adj Close']
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window_length).mean()
        avg_loss = loss.rolling(window_length).mean().abs()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        #print("RSI", rsi)
        #print("G: ", gain)
        #print("L: ", loss)
        #print("G: ", avg_gain)
        #print("L: ", avg_loss)
        
        # Print the last RSI value
        last_rsi = rsi.iloc[-1]
        last_time = rsi.index[-1]
        print(f"\nRSI at {last_time}: {last_rsi:.2f}")

        # Wait for one minute before updating the RSI calculation again
        time.sleep(5)

RSI('GBPUSD=X', 14)
#RSI('AAPL', 14)