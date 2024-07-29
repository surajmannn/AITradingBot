""" Functionality for DMI/ADX Technical Indicator using ta library"""

import ta
import time
import pandas as pd
import yfinance as yf
import yahoo_fin.stock_info as si

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

    dmiadx_data = pd.DataFrame()
    # Use yfinance to retrieve the stock data and load it into a pandas DataFrame
    stock_data = yf.download(symbol, interval=interval, period=data_period, progress=False)

    # Calculate DMI/ADX using the ta library
    dmiadx_data['ADX'] = ta.trend.adx(stock_data['High'], stock_data['Low'], stock_data['Close'])
    dmiadx_data['+DI'] = ta.trend.adx_pos(stock_data['High'], stock_data['Low'], stock_data['Close'])
    dmiadx_data['-DI'] = ta.trend.adx_neg(stock_data['High'], stock_data['Low'], stock_data['Close'])

    current_adx = dmiadx_data['ADX'][-1]
    current_DI_pos = dmiadx_data['+DI'][-1]
    current_DI_neg = dmiadx_data['-DI'][-1]

    return current_adx, current_DI_pos, current_DI_neg
