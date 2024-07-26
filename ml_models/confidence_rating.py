""" Nerual Network for producing confidence rating on security signals """

# TAKE ALPHA VANTAGE MINUTE DATA, WORK OUT BB AND RSI FOR EACH MINUTE, 
# THEN PRODUCE PERCENTAGE ACCURACY IT WILL CHANGE DIRECTION

# remove pandas warnings
import warnings
warnings.simplefilter(action='ignore')

import pandas as pd
from sklearn.neural_network import MLPRegressor
from alpha_vantage.timeseries import TimeSeries
from ta.volatility import BollingerBands
from ta.momentum import rsi as rsi
import yfinance as yf


# Class for creating MLP Regression model objects for selected ticker
class Confidence_Rating:

    # Initialise object with selected security, trading interval, and RSI lookback period
    def __init__(self, ticker, interval, window):
        self.ticker = ticker
        self.interval = interval
        self.window = window
        self.close = ''
        
        # Initialise an MLP Regressor model using input parameters
        self.model = self.create_model(ticker, interval, window)


    
    # Build the model from the object initialisation 
    def create_model(self, ticker, interval, window):

        # Set data parameters for model
        api_key = '0T2WORTYUXU5Z4XA'    # My alpha vantage API key
        ticker_symbol = ticker
        interval = interval
        window = window
        close = self.close

        try:
            # Fetch the data from the Alpha Vantage API
            # Alpha Vantage API requires interval written as min not m
            model_interval = interval + "in"
            ts = TimeSeries(key=api_key, output_format='pandas')
            data, _ = ts.get_intraday(symbol=ticker_symbol, interval=model_interval, outputsize='full')
            data = data.iloc[::-1]
            data = data.between_time('09:30', '16:00')  # Only recieve data from normal US stock exchange trading hours
            close = '4. close'
        except:
            # 1m interval max 7d period on API call
            if (interval == '1m'):
                data_period = '7d'
            else:
                data_period = '30d'
            # Fetch data from yf API as not available through Alpha Vantage API
            data = yf.download(ticker_symbol, interval=interval, period=data_period, progress=False)
            close = 'Close'

        # Depending on API used, set panadaframe string for close column
        self.close = close

        # Calculate the input features usinig built in python library ta
        # Create the input features and labels
        bb = BollingerBands(data[close])
        data['bb_upper'], data['bb_middle'], data['bb_lower'] = bb.bollinger_mavg(), bb.bollinger_hband(), bb.bollinger_lband()
        data['rsi'] = rsi(data[close], window)

        # Drop rows with missing values
        data = data.dropna()

        # Create the input features
        required_data = data[[close, 'rsi', 'bb_upper', 'bb_lower']]
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        required_data.dropna(inplace=True)

        # Initiate lists for input data to be used in the model
        y = []
        X = []

        # select look ahead time interval (Price at n amount of time after to determine price at that point)
        look_ahead = 2

        # Find relevant data for model from historical data
        # Only interested in data where technical indicators hit requirements to reduce imbalancing
        for i in range(len(required_data)-look_ahead):

            # check if RSI is below 30 and price is below boiler band lower band
            if data.iloc[i]['rsi'] < 30 and data.iloc[i]['bb_lower'] > data.iloc[i][close]:
                # check if the price has gone up at look ahead point
                if data.iloc[i+look_ahead][close] > data.iloc[i][close]:
                    y.append(1) # price trend changed from downward to upward
                else:
                    y.append(0) # price trend did not change or continued
                X.append(required_data.iloc[i])

            # check if RSI is above 70 and price is above boiler band upper band
            elif data.iloc[i]['rsi'] > 70 and data.iloc[i]['bb_upper'] < data.iloc[i][close]:
                # check if the price has gone down at look ahead point
                if data.iloc[i+look_ahead][close] < data.iloc[i][close]:
                    y.append(1) # price trend changed from upward to downward
                else:
                    y.append(0) # price trend did not change or continued
                X.append(required_data.iloc[i])

                
        # Train the neural network
        nn = MLPRegressor(hidden_layer_sizes=(40,40), max_iter=1000, random_state=42)
        nn.fit(X, y)

        # return the created model
        return nn
    

    # Using the model, create confidence rating predictions
    # takes curent stock price, the RSI reading, Boiler Band lower and upper bands as inputs
    def confidence_rating(self, price, rsi, bb_upper, bb_lower):
        close = self.close
        # Create test data frame using input parameters for prediction
        test_data_point = pd.DataFrame({close: [price], 'rsi': [rsi], 'bb_upper': [bb_upper], 'bb_lower': [bb_lower]})
        rating = self.model.predict(test_data_point)

        # return prediction value
        return round(rating[0], 2)



"""AAPL = Confidence_Rating('AAPL', '1min', 14)
rating = AAPL.confidence_rating(161.85, 71, 161.78, 160.12)
rating2 = AAPL.confidence_rating(160.55, 28.37, 161.04, 160.60)
rating3 = AAPL.confidence_rating(162.01, 80, 161.78, 160.12)
rating4 = AAPL.confidence_rating(160.35, 23.37, 161.04, 160.60)
rating5 = AAPL.confidence_rating(160.45, 28.45, 162.14, 160.58)

print("Rating1: ", rating)
print("Rating2: ", rating2)
print("Rating3: ", rating3)
print("Rating4: ", rating4)
print("Rating5: ", rating5)"""

#AAPL = Confidence_Rating('GBPUSD=X', '1min', 14)
#AAPL = Confidence_Rating('AAPL', '1min', 14)
#rating = AAPL.confidence_rating(1.2521, 50, 1.2523, 1.2520)
#print(rating)