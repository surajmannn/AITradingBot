""" stock data functionality using Alpha Vantage API / Yahoo_FIN API """

import requests
import pandas as pd
import alpha_vantage.timeseries as timeseries
from yahoo_fin import stock_info
from datetime import datetime

AVKey = "0T2WORTYUXU5Z4XA"

# replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
url = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=IBM&interval=1min&apikey={AVKey}'
r = requests.get(url)
data = r.json()

# gets live price of stock as argument using yahoo_fin API
# https://theautomatic.net/yahoo_fin-documentation/
def get_live_stock_price(stock):
        return stock_info.get_live_price(stock)


# gets historical data of stock as arugment and date range using Alpha Vantage API and returns the data as a panda dataframe
def get_historical_data(stock, start_date):
        api_url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={stock}&apikey={AVKey}&outputsize=full'
        raw_df = requests.get(api_url).json()
        df = pd.DataFrame(raw_df[f'Time Series (Daily)']).T
        df = df.rename(columns = {'1. open': 'open', '2. high': 'high', '3. low': 'low', '4. close': 'close', '5. adjusted close': 'adj close', '6. volume': 'volume'})
        for i in df.columns:
                df[i] = df[i].astype(float)
        df.index = pd.to_datetime(df.index)
        df = df.iloc[::-1].drop(['7. dividend amount', '8. split coefficient'], axis = 1)
        if start_date:
                df = df[df.index >= start_date]
        return df


# gets intraday information for a stock with arguments of stock, intraday interval time and date
def get_intraday_data(stock, interval, start_date):
        api_url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={stock}&interval={interval}&apikey={AVKey}'
        raw_df = requests.get(api_url).json()
        df = pd.DataFrame(raw_df[f'Time Series ({interval})']).T
        df = df.rename(columns = {'1. open': 'open', '2. high': 'high', '3. low': 'low', '4. close': 'close', '5. volume': 'volume'})
        for i in df.columns:
                df[i] = df[i].astype(float)
        df.index = pd.to_datetime(df.index)
        df = df.iloc[::-1]
        if start_date:
                df = df[df.index >= start_date]
        return df


def get_live_updates(stock):
        api_url = f'https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={stock}&apikey={AVKey}'
        raw_df = requests.get(api_url).json()
        attributes = {'attributes':['symbol', 'open', 'high', 'low', 'price', 'volume', 'latest trading day', 'previous close', 'change', 'change percent']}
        attributes_df = pd.DataFrame(attributes)
        values = []
        for i in list(raw_df['Global Quote']):
                values.append(raw_df['Global Quote'][i])
        values_dict = {'values':values}
        values_df = pd.DataFrame(values).rename(columns = {0:'values'})
        frames = [attributes_df, values_df]
        df = pd.concat(frames, axis = 1, join = 'inner').set_index('attributes')
        return df