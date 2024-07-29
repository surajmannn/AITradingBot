""" Demonstrator file for DMI(ADX) indicator using ta library """

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

        # Calculate DMI/ADX using the ta library
        dmiadx_data['ADX'] = ta.trend.adx(stock_data['High'], stock_data['Low'], stock_data['Close'])
        dmiadx_data['+DI'] = ta.trend.adx_pos(stock_data['High'], stock_data['Low'], stock_data['Close'])
        dmiadx_data['-DI'] = ta.trend.adx_neg(stock_data['High'], stock_data['Low'], stock_data['Close'])

        # Get the current market price using yahoo_fin
        current_price = si.get_live_price(symbol).round(8)

        print("Current Price: ", current_price)
        print("ADX: ", dmiadx_data['ADX'][-1])
        print("DI+: ", dmiadx_data['+DI'][-1])
        print("DI-: ", dmiadx_data['-DI'][-1], "\n")

        # Wait for one minute before updating the RSI calculation again
        time.sleep(5)


dmi_adx('GBPUSD=X', '1m', 14)

