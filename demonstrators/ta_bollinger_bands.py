""" Demonstrator file for Bollinger Bands indicator using ta library """

import ta
import time
import pandas as pd
import ta.volatility
import yfinance as yf
import yahoo_fin.stock_info as si

# Function returns ADX, DI+, DI- values. Takes security, trading interval, and period as parameters
def bollinger_bands(ticker, interval, period_length):

    # 1m interval max 7d period on API call
    if (interval in ('1m', '2m', '5m', '10m', '15m')):
        data_period = '1d'
    else:
        data_period = '1mo'

    # security to be used
    symbol = ticker

    # set window period from input parameter
    window = period_length

    start_date = pd.Timestamp.today().strftime('%Y-%m-%d')

    # Wait until the next minute starts
    current_time = pd.Timestamp.now()
    next_minute = (current_time + pd.Timedelta(minutes=1)).replace(second=0, microsecond=0)
    print("Waiting for the next minute to start....")
    time.sleep((next_minute - current_time).seconds)

    while True:
        bb_data = pd.DataFrame()
        stock_data = yf.download(symbol, start=start_date, end=None, interval=interval, period=data_period, progress=False)
        bb = ta.volatility.BollingerBands(stock_data['Close'], window)
        bb_data['bb_upper'], bb_data['bb_middle'], bb_data['bb_lower'] = bb.bollinger_hband(), bb.bollinger_mavg(), bb.bollinger_lband()

        # Get the current market price using yahoo_fin
        current_price = si.get_live_price(symbol).round(8)
        print("Current Price: ", current_price)
        print("BB Upper: ", bb_data['bb_upper'][-1])
        print("BB Lower: ", bb_data['bb_lower'][-1], "\n")

        time.sleep(5)


bollinger_bands('GBPUSD=X', '1m', 14)