""" Functionality for DMI/ADX Technical Indicator """

import yfinance as yf
import numpy as np

# Function returns ADX, DI+, DI- values. Takes security, trading interval, and period as parameters
def dmi_adx(ticker, interval, period_length):

    # 1m interval max 7d period on API call
    if (interval in ('1m', '2m', '5m', '10m', '15m')):
        data_period = '2d'
    else:
        data_period = '30d'

    # security to be used
    symbol = ticker

    # set window period from input parameter
    window = period_length

    # Use yfinance to retrieve the stock data and load it into a pandas DataFrame
    stock_data = yf.download(symbol, interval=interval, period=data_period, progress=False)

    # Calculate True Range (TR)
    true_range = np.maximum(stock_data['high'] - stock_data['low'], 
                          np.maximum(abs(stock_data['high'] - stock_data['close'].shift(1)), 
                                     abs(stock_data['low'] - stock_data['close'].shift(1))))
    
    # Calculate +DM and -DM
    dm_pos = np.where((stock_data['high'] - stock_data['high'].shift(1)) > (stock_data['low'].shift(1) - stock_data['low']), 
                          np.maximum(stock_data['high'] - stock_data['high'].shift(1), 0), 0)
    dm_neg = np.where((stock_data['low'].shift(1) - stock_data['low']) > (stock_data['high'] - stock_data['high'].shift(1)), 
                          np.maximum(stock_data['low'].shift(1) - stock_data['low'], 0), 0)
    
    # Smooth the TR, +DM, and -DM
    smoothed_true_range = true_range.rolling(window=window).sum()
    smoothed_dm_pos = dm_pos.rolling(window=window).sum()
    smoothed_dm_neg = dm_neg.rolling(window=window).sum()
    
    # Calculate +DI and -DI
    DI_pos = 100 * (smoothed_dm_pos / smoothed_true_range)
    DI_neg = 100 * (smoothed_dm_neg / smoothed_true_range)
    
    # Calculate DX and ADX
    dx = 100 * (abs(DI_pos - DI_neg) / (DI_pos + DI_neg))
    adx = dx.rolling(window=window).mean()

    return adx, DI_pos, DI_neg