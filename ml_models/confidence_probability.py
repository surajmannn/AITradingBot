""" Neural Network for producing confidence rating on security signals """

# remove pandas warnings
import warnings
warnings.simplefilter(action='ignore')

# Suppress TensorFlow logs
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from ta_indicators.dataset_preparation import *

import numpy as np


# This class creates machine learning model objects which provide probabilities for price moving in direction of signals
class Confidence_Probability:

    # Initialise object with selected security, trading interval, 
    #... range of historic training data by weeks, and desired machine learning model as integers
    def __init__(self, ticker, interval, training_range, desired_model, look_ahead_values):
        self.ticker = ticker                        # Security name
        self.interval = interval                    # Desired chart interval (i.e, 1min, 5min)
        self.training_range = training_range        # Select amount of historic market weeks for training dataset
        self.desired_model = desired_model          # 1 for MLPRegressor, 2 for SVM, 3 for Random Forest, 4 for LSTM
        self.training_data_set = None
        self.look_ahead = look_ahead_values         # Array of +n price intervals for determining labelling
        
        # Initialise and train the model on class construction
        self.model = self.create_model()

    
    # Build the model from the object initialisation 
    def create_training_data(self):

        # Fetch data from yf API as not available through Alpha Vantage API
        training_data = prepare_training_dataset(self.ticker, self.interval, self.training_range)

        # Initialise class label column for training dataset
        training_data['Label'] =  0

        # Compute classifications for each row in the dataset using pandas vectorisation calculations
        for period in self.look_ahead:

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

        # Determine model
        if self.desired_model == 1:
            trained_model = MLPClassifier(hidden_layer_sizes=(100,100), max_iter=500, random_state=42)
        
        if self.desired_model == 2:
            trained_model = SVC(probability=True, random_state=42)

        if self.desired_model == 3:
            trained_model = RandomForestClassifier(n_estimators=100, random_state=42)

        if self.desired_model == 4:
            trained_model = self.LSTM('simulation')
            return trained_model

        # Fit the model on the training data
        trained_model.fit(X, y)

        return trained_model
    

    # Test models on a train/test split to get model performance metrics
    #... Different from live trading as the test set would be the actual running of the bot
    #... Whereas the model here is tested based on class labels
    def test_model(self):

        training_data = self.training_data_set

        # Define training data as all columns except labels columns
        X = training_data.drop(columns=['Label'])

        # y output to be labels column
        y = training_data['Label']

        # Determine the split index (80% training, 20% testing)
        split_index = int(len(training_data) * 0.8)

        # As the security data is timeseries, split based on past and future pricing
        X_train = X.iloc[:split_index]
        y_train = y.iloc[:split_index]
        X_test = X.iloc[split_index:]
        y_test = y.iloc[split_index:]

        # Determine model
        if self.desired_model == 1:
            trained_model = MLPClassifier(hidden_layer_sizes=(100,100), max_iter=500, random_state=42)
        
        if self.desired_model == 2:
            trained_model = SVC(probability=True, random_state=42)

        if self.desired_model == 3:
            trained_model = RandomForestClassifier(n_estimators=100, random_state=42)

        if self.desired_model == 4:
            self.LSTM('test')
            return 0

        # Fit the model on the training data
        trained_model.fit(X_train, y_train)

        # Predict probabilities on the test set
        y_proba = trained_model.predict_proba(X_test)

        # Extract probabilities for each class
        prob_up = y_proba[:, 1]  # Probability of price going up (class +1)
        prob_down = y_proba[:, 0]  # Probability of price going down (class -1)

        # Predict class labels based on the higher probability
        y_pred = np.where(prob_up > prob_down, 1, -1)

        # Convert y_test to binary (0 and 1)
        y_test_binary = (y_test + 1) // 2  # Converts -1 to 0 and 1 to 1

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy}")

        # Print classification report
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        # Compute ROC AUC score
        roc_auc = roc_auc_score(y_test_binary, prob_up)  # Use prob_up for AUC calculation
        print(f"ROC AUC Score: {roc_auc}")

        return 0
    

    # LSTM Model as requires different architecture
    def LSTM(self, training_type):

        training_data = self.training_data_set

        # Define training data as all columns except labels columns
        X = training_data.drop(columns=['Label']).values

        # y output to be labels column
        y = training_data['Label'].values

        # Convert target to categorical (for Keras)
        y = to_categorical((y + 1) // 2)  # Converts -1 to 0 and 1 to 1

        # Reshape input to [samples, time steps, features]
        X = X.reshape((X.shape[0], 1, X.shape[1]))

        if training_type == 'test':
            # Split the data into training and test sets (80-20 split)
            split_index = int(len(training_data) * 0.8)
            X_train, X_test = X[:split_index], X[split_index:]
            y_train, y_test = y[:split_index], y[split_index:]

        # Define LSTM model
        model = Sequential()
        model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dense(2, activation='softmax'))
        model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

        # Train the model
        model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

        if training_type == 'simulation':
            return model

        # Predict probabilities on the test set
        y_proba = model.predict(X_test)

        # Extract probabilities for each class
        prob_up = y_proba[:, 1]  # Probability of class +1
        prob_down = y_proba[:, 0]  # Probability of class -1

        # Predict class labels based on the higher probability
        y_pred = np.where(prob_up > prob_down, 1, -1)

        # Convert y_test back from categorical
        y_test = np.argmax(y_test, axis=1) * 2 - 1  # Converts 0 to -1 and 1 to 1

        # Evaluate the model
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        # Compute ROC AUC score
        roc_auc = roc_auc_score((y_test + 1) // 2, prob_up)  # Use prob_up for AUC calculation, adjust y_test
        print(f"ROC AUC Score: {roc_auc}")

        return 0
    

    # Using the model, create confidence rating predictions
    # takes curent stock price, the RSI reading, Boiler Band lower and upper bands as inputs
    """The Columns: .index, Close, BB_upper, BB_middle, BB_lower, RSI, ADX, DI+, DI-, Volatility, Label"""
    def confidence_rating(self, price, BB_upper, BB_middle, BB_lower, RSI, ADX, DI_pos, DI_neg, Volatility):

        # Create test data frame using input parameters for prediction
        test_data_point = pd.DataFrame({'Close': [price], 'BB_upper': [BB_upper], 'BB_middle': [BB_middle], 'BB_lower': [BB_lower],
                                        'RSI': [RSI], 'ADX': [ADX], 'DI+': [DI_pos], 'DI-': [DI_neg], 'Volatility': [Volatility]})
        
        rating = self.model.predict_proba(test_data_point)

        # Extract probabilities for each class
        prob_up = rating[:, 1]  # Probability of price going up (class +1)
        prob_down = rating[:, 0]  # Probability of price going down (class -1)

        # return prediction value
        return prob_up, prob_down