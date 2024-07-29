""" Functionality for DMI/ADX Technical Indicator """

import yfinance as yf
import numpy as np
import yahoo_fin.stock_info as si
import pandas as pd
import time

# Function returns ADX, DI+, DI- values. Takes security, trading interval, and period as parameters
def dmi_adx(ticker, interval, period_length):

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

        dmiadx_data = pd.DataFrame()
        # Use yfinance to retrieve the stock data and load it into a pandas DataFrame
        stock_data = yf.download(symbol, start=start_date, end=None, interval=interval, period=data_period, progress=False)

        # Calculate True Range (TR)
        dmiadx_data['TR'] = np.maximum(stock_data['High'] - stock_data['Low'], 
                            np.maximum(abs(stock_data['High'] - stock_data['Close'].shift(1)), 
                                        abs(stock_data['Low'] - stock_data['Close'].shift(1))))
        
        # Calculate +DM and -DM
        dmiadx_data['+DM'] = np.where((stock_data['High'] - stock_data['High'].shift(1)) > (stock_data['Low'].shift(1) - stock_data['Low']), 
                            np.maximum(stock_data['High'] - stock_data['High'].shift(1), 0), 0)
        dmiadx_data['-DM'] = np.where((stock_data['Low'].shift(1) - stock_data['Low']) > (stock_data['High'] - stock_data['High'].shift(1)), 
                            np.maximum(stock_data['Low'].shift(1) - stock_data['Low'], 0), 0)
        
        # Smooth the TR, +DM, and -DM
        dmiadx_data['TR_sum'] = dmiadx_data['TR'].rolling(window=window).sum()
        dmiadx_data['+DM_sum'] = dmiadx_data['+DM'].rolling(window=window).sum()
        dmiadx_data['-DM_sum'] = dmiadx_data['-DM'].rolling(window=window).sum()
        
        # Calculate +DI and -DI
        dmiadx_data['+DI'] = 100 * (dmiadx_data['+DM_sum'] / dmiadx_data['TR_sum'])
        dmiadx_data['-DI'] = 100 * (dmiadx_data['-DM_sum'] / dmiadx_data['TR_sum'])
        
        # Calculate DX and ADX
        dmiadx_data['DX'] = 100 * (abs(dmiadx_data['+DI'] - dmiadx_data['-DI']) / (dmiadx_data['+DI'] + dmiadx_data['-DI']))
        dmiadx_data['ADX'] = dmiadx_data['DX'].rolling(window=window).mean()

        # Get the current market price using yahoo_fin
        current_price = si.get_live_price(symbol).round(8)

        print("Current Price: ", current_price)
        print("ADX: ", dmiadx_data['ADX'][-1])
        print("DI+: ", dmiadx_data['+DI'][-1])
        print("DI-: ", dmiadx_data['-DI'][-1], "\n")

        # Wait for one minute before updating the RSI calculation again
        time.sleep(5)


dmi_adx('GBPUSD=X', '1m', 14)