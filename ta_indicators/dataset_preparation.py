""" This file prepares the dataset by adding all technical analysis indicator values to the API dataset for desired period """

from ta_indicators.bollinger_bands import *
from ta_indicators.dmi_adx import *
from ta_indicators.rsi import *
import pandas as pd
import yfinance as yf

# Takes security, desired dataset period, and desired trading interval (i.e, 1m, 5m) as input
def prepare_dataset(ticker, data_period, interval):

    # Retrieve dataset from yahoo finance api
    security_data = yf.download(ticker, interval=interval, period=data_period, progress=False)

    # Add technical indicator values to the dataset
    dataset = pd.DataFrame(security_data)

    # Append technical indicator values to the dataset by calling the functions
    dataset = bollinger_bands(dataset)
    dataset = rsi(dataset)
    dataset = dmi_adx(dataset)

    # Drop uneeded columns
    dataset.drop(['High', 'Close', 'Open', 'Adj Close', 'Volume'], axis=1, inplace = True)

    #print(dataset)

    return dataset


# Yahoo finance API has 5d limit for historical '1m' interval data so this function 
def prepare_training_dataset(ticker, interval, training_range):

    security_data = get_dates_list(ticker)

    # Get last 5 days of security data from api call
    training_data = yf.download(ticker, interval = interval, start=security_data[-6], end=security_data[-1], progress=False)

    # Concatanate previous 5day ranges of security_data (due to API limit) for desired training range per 5 day range (i.e. 3 = 3weeks)
    if (training_range != 1):
        for x in range(1, min(3,training_range)):
            data_period = yf.download(ticker, interval = interval, start=security_data[-6-(x*5)], end=security_data[-1-(x*5)], progress=False)
            training_data = pd.concat([data_period, training_data])

    # Append technical indicator values to the dataset by calling the functions
    training_data = bollinger_bands(training_data)
    training_data = rsi(training_data)
    training_data = dmi_adx(training_data)

    # Drop uneeded columns
    training_data.drop(['High', 'Low', 'Open', 'Adj Close', 'Volume'], axis=1, inplace = True)

    # After applying the technical indicators the first n intervals for the lookback periods of technical analaysis will not have values so dropped
    training_data = training_data.dropna()
    #print(training_data)

    return training_data


# Get list of dates of market days from last month of security data
def get_dates_list(ticker):

    # Retrieve dataset from yahoo finance api
    security_data = pd.DataFrame(yf.download(ticker, interval='1d', period='1mo', progress=False))
    
    # Creates a data frame which contains last 1month of dates in api call format
    security_data = security_data.index

    return security_data


#prepare_dataset('GBPUSD=X', '5d', '1m')
#print(prepare_training_dataset('GBPUSD=X', '1m', 2))