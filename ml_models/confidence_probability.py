""" Nerual Network for producing confidence rating on security signals """

# TAKE ALPHA VANTAGE MINUTE DATA, WORK OUT BB AND RSI FOR EACH MINUTE, 
# THEN PRODUCE PERCENTAGE ACCURACY IT WILL CHANGE DIRECTION

# remove pandas warnings
import warnings
warnings.simplefilter(action='ignore')

import pandas as pd
from sklearn.neural_network import MLPRegressor
from ta_indicators.dataset_preparation import *


# Class for creating MLP Regression model objects for selected ticker
class Confidence_Probability:

    # Initialise object with selected security, trading interval, and RSI lookback period
    def __init__(self, ticker, interval, training_range):
        self.ticker = ticker
        self.interval = interval
        self.training_range = training_range
        
        # Initialise an MLP Regressor model using input parameters
        self.model = self.create_model(ticker, interval)

    
    # Build the model from the object initialisation 
    def create_model(self, ticker, interval):

        # Fetch data from yf API as not available through Alpha Vantage API
        training_data = prepare_training_dataset(self.ticker, self.interval, self.training_range)

        # Initiate lists for input data to be used in the model
        y = []
        X = []

        # select look ahead time interval (Price at n amount of time after to determine price at that point)
        look_ahead = 2

        # Find relevant data for model from historical data
        # Only interested in data where technical indicators hit requirements to reduce imbalancing
        for i in range(len(training_data)-look_ahead):

            """ WRITE STRATEGY FOR DATA POINTS,
                SAME AS SIGNAL GENERATION STRATEGY PARAMS"""

                
        # Train the neural network
        nn = MLPRegressor(hidden_layer_sizes=(40,40), max_iter=1000, random_state=42)
        nn.fit(X, y)

        # return the created model
        return nn
    

    # Using the model, create confidence rating predictions
    # takes curent stock price, the RSI reading, Boiler Band lower and upper bands as inputs
    """The Columns: .index, Close, bb_upper, bb_middle, bb_lower, rsi, ADX, DI+, DI-"""
    def confidence_rating(self, price, rsi, bb_upper, bb_lower):
        close = self.close
        # Create test data frame using input parameters for prediction
        test_data_point = pd.DataFrame({close: [price], 'rsi': [rsi], 'bb_upper': [bb_upper], 'bb_lower': [bb_lower]})
        rating = self.model.predict(test_data_point)

        # return prediction value
        return round(rating[0], 2)


#GBP = Confidence_Rating('GBPUSD=X', '1min', 14)
#rating = GBP.confidence_rating(1.2521, 50, 1.2523, 1.2520)
#print(rating)