""" Functionality for Bollinger Bands Technical Indicator using ta library"""

import ta
import time
import pandas as pd
import ta.volatility
import yfinance as yf
import yahoo_fin.stock_info as si

# Function returns bollinger band values. Takes security and trading interval as parameters
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

    # Create pandas data frame for indicator values
    bb_data = pd.DataFrame()

    # Get security data from yfinance api
    stock_data = yf.download(symbol, interval=interval, period=data_period, progress=False)

    # Use Bollinger Band technical indicator from ta library to get upper and lower bands
    bb = ta.volatility.BollingerBands(stock_data['Close'], window)
    bb_data['bb_upper'], bb_data['bb_middle'], bb_data['bb_lower'] = bb.bollinger_hband(), bb.bollinger_mavg(), bb.bollinger_lband()

    # Get latest upper and lower band values
    current_bb_upper = bb_data['bb_upper'][-1]
    current_bb_lower = bb_data['bb_lower'][-1]

    return current_bb_upper, current_bb_lower
