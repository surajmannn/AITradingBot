""" This file produces probabilities of prices moving in the direction of the generated signals.
    ... Taking the desired model choice, this file creates and trains the machine learning model as well as testing it,
    ... which is used as the final decision basis on trade execution. """

# remove pandas warnings
import warnings
warnings.simplefilter(action='ignore')

# Suppress TensorFlow logs
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import libraries for datasets
import pandas as pd
import numpy as np

# Import machine learning models and metrics from sklearn and tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV

# Import dataset preparation logic from indicators folder
from ta_indicators.dataset_preparation import *


# This class creates machine learning model objects which provide probabilities for price moving in direction of signals
class Confidence_Probability:

    # Initialise object with selected security, trading interval, 
    #... range of historic training data by weeks, and desired machine learning model as integers
    def __init__(self, ticker, interval='1m', training_range=1, desired_model=1, look_ahead_values=[5,15,30,45,60], dataset=None):
        self.ticker = ticker                        # (str) Security name
        self.interval = interval                    # (str) Desired chart interval (i.e, 1min, 5min)
        self.training_range = training_range        # (int) Select amount of historic market weeks for training dataset (e.g. 1, 2, 3)
        self.desired_model = desired_model          # (int) The specific model to use for simulation (1=MLP, 2=SVM, 3=RF, 4=LSTM)
        self.dataset = dataset                      # Dataset to be used for training
        self.look_ahead = look_ahead_values         # (int array) Array of +n price intervals for determining labelling

        self.model = None                                       # initialise empty model
        self.current_training_set = self.create_training_data() # Initialise first training dataset on instantiation

    
    # Updates the current training data set
    def update_training_data(self, new_training_data):
        self.dataset = new_training_data
        self.create_training_data()

    
    # Build the model from the object initialisation 
    def create_training_data(self):

        training_data = self.dataset

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

        # Remove max look_ahead value of rows as labels cannot be generated based on look ahead prices not exisiting (end of dataset)
        training_data = training_data[:-max(self.look_ahead)]
        
        """# Count the number of 1's and -1's
        label_counts = training_data['Label'].value_counts()
        print("Number of 1's:", label_counts.get(1, 0))
        print("Number of -1's:", label_counts.get(-1, 0))"""

        # Update training data set
        self.current_training_set = training_data

        # return the created model
        return training_data #trained_model
    

    # This function creates and trains the machine learning models based on the desired model
    def create_model(self):

        # Use latest training dataset
        training_data = self.current_training_set

        # Define training data as all columns except labels columns
        X = training_data.drop(columns=['Label'])

        # y output to be labels column
        y = training_data['Label']

        # Determine and train desired model
        if self.desired_model == 1:
            trained_model = MLPClassifier(activation='relu', alpha=0.001, solver='adam', hidden_layer_sizes=(100,100), max_iter=1000, random_state=42)
        
        if self.desired_model == 2:
            trained_model = SVC(kernel='rbf', C=0.1, gamma='scale', probability=True, random_state=42)

        if self.desired_model == 3:
            trained_model = RandomForestClassifier(max_depth=None, min_samples_leaf=1, min_samples_split=10, n_estimators=100, random_state=42)

        if self.desired_model == 4:
            trained_model = self.LSTM('simulation')
            self.model = trained_model
            return trained_model

        # Fit the model on the training data
        trained_model.fit(X, y)

        # Update class variable
        self.model = trained_model

        return trained_model
    

    # Test models on a train/test split to get model performance metrics
    #... Different from live trading as the test set would be the actual running of the bot
    #... Whereas the model here is tested based on class labels on training dataset
    def test_model(self):

        # Use latest training data
        training_data = self.current_training_set

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
            trained_model = MLPClassifier(activation='relu', alpha=0.001, solver='adam', hidden_layer_sizes=(100,100), max_iter=1000, random_state=42)
        
        if self.desired_model == 2:
            trained_model = SVC(kernel='rbf', C=0.1, gamma='scale', probability=True, random_state=42)

        if self.desired_model == 3:
            trained_model = RandomForestClassifier(max_depth=None, min_samples_leaf=1, min_samples_split=10, n_estimators=100, random_state=42)

        if self.desired_model == 4:
            accuracy, roc_auc = self.LSTM('test')
            return accuracy, roc_auc            # End function as LSTM function returns the model as the architecture is different

        # Fit the model on the training data
        trained_model.fit(X_train, y_train)

        # Filter the test set for significant signal conditions (RSI > 65 or RSI < 35) for more realistic performance results compared with simulation
        X_test_signal = X_test[(X_test['RSI'] > 65) | (X_test['RSI'] < 35)]
        y_test_signal = y_test[(X_test['RSI'] > 65) | (X_test['RSI'] < 35)]

        if len(X_test_signal) == 0:
            print("No significant signal conditions found in the test set.")
            return None

        # Predict probabilities on the test set
        y_proba = trained_model.predict_proba(X_test)
        y_proba_signal = trained_model.predict_proba(X_test_signal)

        # Extract probabilities for each class
        prob_up = y_proba[:, 1]  # Probability of price going up (class +1)
        prob_down = y_proba[:, 0]  # Probability of price going down (class -1)

        # Extract probabilities for each class
        prob_up_sig = y_proba_signal[:, 1]  # Probability of price going up (class +1)
        prob_down_sig = y_proba_signal[:, 0]  # Probability of price going down (class -1)

        # Predict class labels based on the higher probability
        y_pred = np.where(prob_up > prob_down, 1, -1)

        # Predict class labels based on the higher probability
        y_pred_sig = np.where(prob_up_sig > prob_down_sig, 1, -1)

        # Convert y_test to binary (0 and 1)
        y_test_binary = (y_test + 1) // 2  # Converts -1 to 0 and 1 to 1
        y_test_binary_signal = (y_test_signal + 1) // 2

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy}")

        # Evaluate the model on signal only
        sig_accuracy = accuracy_score(y_test_signal, y_pred_sig)
        print(f"Signal Accuracy: {sig_accuracy}")

        # Print classification report
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        # Print classification report for sig
        print("Signal Classification Report:")
        print(classification_report(y_test_signal, y_pred_sig))

        # Compute ROC AUC score
        roc_auc = roc_auc_score(y_test_binary, prob_up)  # Use prob_up for AUC calculation
        print(f"ROC AUC Score: {roc_auc}")

        # Compute ROC AUC score for sig
        sig_roc_auc = roc_auc_score(y_test_binary_signal, prob_up_sig)  # Use prob_up for AUC calculation
        print(f"ROC AUC Signal Score: {sig_roc_auc}")

        return accuracy, roc_auc, sig_accuracy, sig_roc_auc
    

    # LSTM Model as requires different architecture
    def LSTM(self, training_type):

        # Use latest training data
        training_data = self.current_training_set

        # Define training data as all columns except labels columns
        X = training_data.drop(columns=['Label']).values

        # y output to be labels column
        y = training_data['Label'].values

        # Convert target to categorical (for Keras)
        y = to_categorical((y + 1) // 2)  # Converts -1 to 0 and 1 to 1

        # Reshape input to [samples, time steps, features]
        X = X.reshape((X.shape[0], 1, X.shape[1]))

        # If the model is for test, create train test split for performance metrics
        if training_type == 'test':
            # Split the data into training and test sets (80-20 split)
            split_index = int(len(training_data) * 0.8)
            X_train, X_test = X[:split_index], X[split_index:]
            y_train, y_test = y[:split_index], y[split_index:]

        # Otherwise, just use the whole dataset as training data, as trading data simulation is the test set
        else:
            X_train = X
            y_train = y

        # Define LSTM model
        model = Sequential()
        model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dense(2, activation='softmax'))
        model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

        # Train the model
        model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

        # If the model is for simulation, there is not train test split so return the model
        if training_type == 'simulation':
            return model

        # Predict probabilities on the test set
        y_proba = model.predict(X_test)

        # Extract probabilities for each class
        prob_up = y_proba[:, 1]  # Probability of price going up (class +1)
        prob_down = y_proba[:, 0]  # Probability of price going down (class -1)

        # Predict class labels based on the higher probability
        y_pred = np.where(prob_up > prob_down, 1, -1)

        # Convert y_test back from categorical
        y_test = np.argmax(y_test, axis=1) * 2 - 1  # Converts 0 to -1 and 1 to 1

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy}")

        # Print classification report
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        # Compute ROC AUC score
        roc_auc = roc_auc_score(y_test, prob_up)  # Use prob_up for AUC calculation
        print(f"ROC AUC Score: {roc_auc}")

        return accuracy, roc_auc
    

    # Perform optimisation through hyper parameter grid search to determine best tuning
    def hyperparameter_tuning(self):

        # Use latest training data
        training_data = self.current_training_set

        # Define training data as all columns except labels columns
        X = training_data.drop(columns=['Label'])

        # y output to be labels column
        y = training_data['Label']

        # Define the parameter grids for each model
        param_grid_mlp = {
            'hidden_layer_sizes': [(50,50), (100,100), (200,), (100,50,25)],
            'activation': ['relu', 'tanh'],
            'solver': ['adam', 'sgd'],
            'alpha': [0.001, 0.01, 0.1],
            'max_iter': [1000, 2000]
        }

        param_grid_svc = {
            'C': [0.05, 0.1, 1],
            'kernel': ['rbf', 'poly'],
            'gamma': ['scale', 0.01, 0.1, 1],
        }

        param_grid_rf = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        # Choose the model and parameter grid based on the desired model
        if self.desired_model == 1:
            model = MLPClassifier(random_state=42)
            param_grid = param_grid_mlp

        if self.desired_model == 2:
            model = SVC(probability=True, random_state=42)
            param_grid = param_grid_svc

        if self.desired_model == 3:
            model = RandomForestClassifier(random_state=42)
            param_grid = param_grid_rf

        print("BEGINNING GRID SEARCH")
        # Create the GridSearchCV object
        grid_search = GridSearchCV(model, param_grid, scoring='accuracy', n_jobs=-1)

        print("\nFITTING")
        # Fit the model on the training data
        grid_search.fit(X, y)

        print("\nPREPARING RESULTS\n")
        # Extract the results
        results = grid_search.cv_results_

        # Get the indices of the top 3 best hyperparameter sets
        top_3_idx = results['rank_test_score'].argsort()[:3]

        # Display the top 3 hyperparameter sets
        print("Top 3 hyperparameter sets:")
        for idx in top_3_idx:
            print(f"Rank: {results['rank_test_score'][idx]}")
            print(f"Mean Validation Score: {results['mean_test_score'][idx]}")
            print(f"Hyperparameters: {results['params'][idx]}\n")

        # Print the best parameters and score
        print("Best parameters found: ", grid_search.best_params_)
        print("Best cross-validation score: ", grid_search.best_score_)

        return grid_search.best_params_
    

    # Using the model, create confidence rating predictions
    # takes curent stock price, the RSI reading, Boiler Band lower and upper bands as inputs
    """The Columns: .index, Close, BB_upper, BB_middle, BB_lower, RSI, ADX, DI+, DI-, Volatility, Label"""
    def confidence_rating(self, price, BB_upper, BB_middle, BB_lower, RSI, ADX, DI_pos, DI_neg, Volatility):

        # Create test data frame using input parameters for prediction
        test_data_point = pd.DataFrame({'Close': [price], 'BB_upper': [BB_upper], 'BB_middle': [BB_middle], 'BB_lower': [BB_lower],
                                        'RSI': [RSI], 'ADX': [ADX], 'DI+': [DI_pos], 'DI-': [DI_neg], 'Volatility': [Volatility]})
        
        # If LSTM model, requires different formatting than the other 3 models for getting prediction probability
        #... as requires reshape for test input
        if self.desired_model == 4:
            test_data_point = test_data_point.values
            test_data_point = test_data_point.reshape((test_data_point.shape[0], 1, test_data_point.shape[1]))
            rating = self.model.predict(test_data_point)
        else:
            rating = self.model.predict_proba(test_data_point)

        # Extract probabilities for each class
        prob_up = rating[:, 1]  # Probability of price going up (class +1)
        prob_down = rating[:, 0]  # Probability of price going down (class -1)

        # return prediction value
        return prob_up, prob_down