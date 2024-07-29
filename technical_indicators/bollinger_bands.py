""" Functionality for Bollinger Bands Technical Indicator """

import yfinance as yf

# Function returns bollinger band values. Takes security and trading interval as parameters
def bollinger_bands(ticker, interval):

    # 1m interval max 7d period on API call
    if (interval in ('1m', '2m', '5m', '10m', '15m')):
        data_period = '2d'
    else:
        data_period = '30d'

    # security to be used
    symbol = ticker

    # Use yfinance to retrieve the stock data and load it into a pandas DataFrame
    stock_data = yf.download(symbol, interval=interval, period=data_period, progress=False)

    # Calculate the 20-day moving average and standard deviation using the rolling() method
    rolling_mean = stock_data['Close'].rolling(window=20).mean()
    rolling_std = stock_data['Close'].rolling(window=20).std()

    # Calculate the upper and lower bands using the extracted data
    upper_band = rolling_mean + (2 * rolling_std)
    lower_band = rolling_mean - (2 * rolling_std)

    # Get the latest upper and lower band values
    current_lb = lower_band.iloc[-1]
    current_ub = upper_band.iloc[-1]

    # return the current lower and upper band values
    return current_ub, current_lb