""" Neural Network for producing confidence rating on security signals """

# remove pandas warnings
import warnings
warnings.simplefilter(action='ignore')

import pandas as pd
from sklearn.neural_network import MLPRegressor
from ta_indicators.dataset_preparation import *

""" INSERT OTHER MODELS HERE TOO AS PARAMETER """


# Class for creating MLP Regression model objects for selected ticker
class Confidence_Probability:

    # Initialise object with selected security, trading interval, and RSI lookback period
    def __init__(self, ticker, interval, training_range, desired_model):
        self.ticker = ticker
        self.interval = interval
        self.training_range = training_range
        self.desired_model = desired_model
        
        # Initialise an MLP Regressor model using input parameters
        self.model = self.create_model()

    
    # Build the model from the object initialisation 
    def create_model(self):

        # Fetch data from yf API as not available through Alpha Vantage API
        training_data = prepare_training_dataset(self.ticker, self.interval, self.training_range)

        # Initiate lists for input data to be used in the model
        y = []
        X = []

        # select look ahead time interval (Price at n amount of time after to determine price at that point)
        look_ahead = [2,5,10,15,30]

        # Find relevant data for model from historical data
        # Only interested in data where technical indicators hit requirements to reduce imbalancing
        for i in range(len(training_data)-max(look_ahead)):

            """ WRITE STRATEGY FOR DATA POINTS,
                SAME AS SIGNAL GENERATION STRATEGY PARAMS"""
            
            """ NEED TO THINK OF SELECTION FOR TRAINING OUTCOMES. I.e, how many rows and how to determine up to what point price changed
                how to label? how many data points to include? """
            


        """if self.model == 1:     
            # Train the neural network
            trained_model = MLPRegressor(hidden_layer_sizes=(40,40), max_iter=1000, random_state=42)
            trained_model.fit(X, y)
        if self.model == 2:
            trained_model = 0 
        if self.model == 3:
            return 0"""

        # return the created model
        return 0 #trained_model
    

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


#GBP = Confidence_Probability('GBPUSD=X', '1m', 2, 1)
#GBP.create_model()
#rating = GBP.confidence_rating(1.2521, 50, 1.2523, 1.2520)
#print(rating)