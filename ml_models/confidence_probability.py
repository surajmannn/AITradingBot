""" Neural Network for producing confidence rating on security signals """

# remove pandas warnings
import warnings
warnings.simplefilter(action='ignore')

import pandas as pd
from sklearn.neural_network import MLPRegressor
from ta_indicators.dataset_preparation import *
import numpy as np


# This class creates machine learning model objects which provide probabilities for price moving in direction of signals
class Confidence_Probability:

    # Initialise object with selected security, trading interval, 
    #... range of historic training data by weeks, and desired machine learning model as integers
    def __init__(self, ticker, interval, training_range, desired_model):
        self.ticker = ticker                        # Security name
        self.interval = interval                    # Desired chart interval (i.e, 1min, 5min)
        self.training_range = training_range        # Select amount of historic market weeks for training dataset
        self.desired_model = desired_model          # 1 for MLPRegressor, 2 for SVM, 3 for LSTM
        self.training_data_set = None
        
        # Initialise an MLP Regressor model using input parameters
        #self.model = self.create_model()

    
    # Build the model from the object initialisation 
    def create_training_data(self):

        # Fetch data from yf API as not available through Alpha Vantage API
        training_data = prepare_training_dataset(self.ticker, self.interval, self.training_range)

        # Initialise class label column for training dataset
        training_data['Label'] =  0

        # select look ahead time interval (Price at n amount of time after to determine price at that point)
        look_ahead = [2,5,10,15,30]

        # Compute classifications for each row in the dataset using pandas vectorisation calculations
        for period in look_ahead:

            # Use shift to get future prices at look_ahead periods
            future_prices = training_data['Close'].shift(-period)
            
            # Perform vectorized comparisons and update the 'Label' column. 
            #... Using the 5 look ahead periods take majority price change direction
            #... 1 if majority of future prices are higher than current price, and -1 if majority of future prices are below current price
            training_data['Label'] += np.where(training_data['Close'] < future_prices, 1, 0)
            training_data['Label'] -= np.where(training_data['Close'] > future_prices, 1, 0)

        # Finalize the 'Label' column: set to 1 if positive, -1 if negative
        training_data['Label'] = np.where(training_data['Label'] > 0, 1, -1)

        # Remove last 30 rows as labels cannot be generated based on look ahead prices not exisiting
        training_data = training_data[:-30]
        
        """# Count the number of 1's and -1's
        label_counts = training_data['Label'].value_counts()
        print("Number of 1's:", label_counts.get(1, 0))
        print("Number of -1's:", label_counts.get(-1, 0))"""

        # Add dataset to class variable
        self.training_data_set = training_data

        # return the created model
        return training_data #trained_model
    

    # This function creates and trains the machine learning models based on the desired model
    def create_model(self):

            # Obtain the training data for model training
            training_data = self.create_training_data()

            # Define training data as all columns except labels columns
            X = training_data.drop(columns=['Label'])

            # y output to be labels column
            y = training_data['Label']

            if self.desired_model == 1:     
                # Train the neural network
                #trained_model = MLPRegressor(hidden_layer_sizes=(40,40), max_iter=1000, random_state=42)
                #trained_model.fit(X, y)
                trained_model = 0
            
            if self.desired_model == 2:
                trained_model = 0 
            
            if self.desired_model == 3:
                trained_model = 0

            return trained_model
    

    # Test models on a train/test split to get model performance metrics
    #... Different from live trading as the test set would be the actual running of the bot
    #... Whereas the model here is tested based on class labels
    def test_model(self):

        training_data = self.training_data_set

        return 0
    

    # Using the model, create confidence rating predictions
    # takes curent stock price, the RSI reading, Boiler Band lower and upper bands as inputs
    """The Columns: .index, Close, BB_upper, BB_middle, BB_lower, RSI, ADX, DI+, DI-, Volatility, Label"""
    def confidence_rating(self, price, BB_upper, BB_middle, BB_lower, RSI, ADX, DI_pos, DI_neg, Volatility):

        # Create test data frame using input parameters for prediction
        test_data_point = pd.DataFrame({'Close': [price], 'BB_upper': [BB_upper], 'BB_middle': [BB_middle], 'BB_lower': [BB_lower],
                                        'RSI': [RSI], 'ADX': [ADX], 'DI+': [DI_pos], 'DI-': [DI_neg], 'Volatility': [Volatility]})
        rating = self.model.predict(test_data_point)

        # return prediction value
        return round(rating[0], 2)