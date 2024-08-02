""" This file prepares the dataset by adding all technical analysis indicator values to the API dataset for desired period """

from bollinger_bands import *
from dmi_adx import *
from rsi import *
import pandas as pd
import yfinance as yf

# Takes security, desired dataset period, and desired trading interval (i.e, 1m, 5m) as input
def prepare_dataset(ticker, data_period, interval):

    # Retrieve dataset from yahoo finance api
    security_data = yf.download(ticker, interval=interval, period=data_period, progress=False)

    # Add technical indicator values to the dataset
    dataset = pd.DataFrame(security_data)

    # Drop uneeded columns
    dataset.drop(['Open', 'Adj Close', 'Volume'], axis=1, inplace = True)

    # Append technical indicator values to the dataset by calling the functions
    dataset = bollinger_bands(dataset)
    dataset = rsi(dataset)
    dataset = dmi_adx(dataset)

    #print(dataset)

    return dataset

#prepare_dataset('GBPUSD=X', '5d', '1m')