""" This file prepares the dataset by adding all technical analysis indicator values to the API dataset for desired period """

from ta_indicators.bollinger_bands import *
from ta_indicators.dmi_adx import *
from ta_indicators.rsi import *
import pandas as pd
import yfinance as yf

# Takes security, and gets the maximum amount of historical data (1m interval = 30d as per api limit)
def prepare_entire_dataset(ticker, interval):

    # Get list of valid market days from the last month
    dates_list = get_dates_list(ticker)

    # Get the oldest day of security data
    security_data = yf.download(ticker, start=dates_list[4], end=dates_list[5], period='5d', interval=interval, progress=False)

    # Concatanate each following day to the current security data set until the most recent day
    for x in range(5, len(dates_list)-1):
        data_period = yf.download(ticker, start=dates_list[x], end=dates_list[x+1], interval = interval, period='5d', progress=False)
        security_data = pd.concat([security_data, data_period])

    # Append technical indicator values to the dataset by calling the functions
    security_data = bollinger_bands(security_data)
    security_data = rsi(security_data)
    security_data = dmi_adx(security_data)

    # Drop uneeded columns
    security_data.drop(['High', 'Low', 'Open', 'Adj Close', 'Volume'], axis=1, inplace = True)

    # After applying the technical indicators the first n intervals for the lookback periods of technical analaysis will not have values so dropped
    security_data = security_data.dropna()

    return security_data


# Takes security, desired dataset period (1d, 5d), and desired trading interval (i.e, 1m, 5m) as input
def prepare_dataset(ticker, start_date, end_date, data_period, interval):

    # Retrieve dataset from yahoo finance api
    security_data = yf.download(ticker, start=start_date, end=end_date, period=data_period, interval=interval, progress=False)

    # Add technical indicator values to the dataset
    security_data = pd.DataFrame(security_data)

    # Append technical indicator values to the dataset by calling the functions
    security_data = bollinger_bands(security_data)
    security_data = rsi(security_data)
    security_data = dmi_adx(security_data)

    # Drop uneeded columns
    security_data.drop(['High', 'Low', 'Open', 'Adj Close', 'Volume'], axis=1, inplace = True)
    security_data = security_data.dropna()

    return security_data


# Yahoo finance API has 5d limit for historical '1m' interval data so this function 
def prepare_training_dataset(ticker, interval, training_range):

    security_data = get_dates_list(ticker)

    # Get last 5 days of security data from api call
    training_data = yf.download(ticker, interval = interval, start=security_data[-10], end=security_data[-5], progress=False)

    # Concatanate previous 5day ranges of security_data (due to API limit) for desired training range per 5 day range (i.e. 3 = 3weeks)
    if (training_range != 1):
        for x in range(1, min(3,training_range)):
            data_period = yf.download(ticker, interval = interval, start=security_data[-10-(x*5)], end=security_data[-5-(x*5)], progress=False)
            training_data = pd.concat([data_period, training_data])

    # Append technical indicator values to the dataset by calling the functions
    training_data = bollinger_bands(training_data)
    training_data = rsi(training_data)
    training_data = dmi_adx(training_data)

    # Drop uneeded columns
    training_data.drop(['High', 'Low', 'Open', 'Adj Close', 'Volume'], axis=1, inplace = True)

    # After applying the technical indicators the first n intervals for the lookback periods of technical analaysis will not have values so dropped
    training_data = training_data.dropna()

    return training_data


# Get list of dates of market days from last month of security data
def get_dates_list(ticker):

    # Retrieve dataset from yahoo finance api
    security_data = pd.DataFrame(yf.download(ticker, interval='1d', period='1mo', progress=False))
    
    # Creates a data frame which contains last 1month of dates in api call format
    security_data = security_data.index

    return security_data


# This retrieves a list of market days for the desired security data on the inputted trading range
#... e.g. start_date=2024-01-01, end_date=2024-08-09
def prepare_simulation_range_date_list(ticker, start_date, end_date):
    # Retrieve dataset from yahoo finance api
    security_data = pd.DataFrame(yf.download(ticker, start=start_date, end=end_date, interval='1d', progress=False))
    
    # Creates a data frame which contains trading range dates in api call format
    security_data = security_data.index

    return security_data


# Gets latest days security data
def get_current_days_data(ticker, interval):
    # Retrieve dataset from yahoo finance api
    security_data = yf.download(ticker, period='1d', interval=interval, progress=False)

    # Add technical indicator values to the dataset
    security_data = pd.DataFrame(security_data)

    # Append technical indicator values to the dataset by calling the functions
    security_data = bollinger_bands(security_data)
    security_data = rsi(security_data)
    security_data = dmi_adx(security_data)

    # Drop uneeded columns
    security_data.drop(['High', 'Low', 'Open', 'Adj Close', 'Volume'], axis=1, inplace = True)
    security_data = security_data.dropna()

    return security_data