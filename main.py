""" Main file for the project """

from ml_models.confidence_probability import *
from ta_indicators.dataset_preparation import *
from trading_system.run_trading import *
import pandas as pd

def testing():
    """ Testing functions"""

    training_data = prepare_entire_dataset('GBPUSD=X', '1m')
    # Get best hyperparameters for the models
    mlp_tuning = Confidence_Probability(
            ticker="GBPUSD=X",
            interval='1m',
            training_range=1,
            desired_model=1,
            look_ahead_values=[5,15,30,45,60],
            dataset=training_data
        )
    mlp_params = mlp_tuning.hyperparameter_tuning()
    #Best parameters found:  {'activation': 'relu', 'alpha': 0.001, 'hidden_layer_sizes': (100, 100), 'max_iter': 1000, 'solver': 'adam'}
    
    svm_tuning = Confidence_Probability(
            ticker="GBPUSD=X",
            interval='1m',
            training_range=1,
            desired_model=2,
            look_ahead_values=[5,15,30,45,60],
            dataset=training_data
        )
    svm_params = svm_tuning.hyperparameter_tuning()
    #Best parameters found:  {'C': 0.1, 'kernel': rbf, 'gamma': 'scale'}
    
    rf_tuning = Confidence_Probability(
            ticker="GBPUSD=X",
            interval='1m',
            training_range=1,
            desired_model=3,
            look_ahead_values=[5,15,30,45,60],
            dataset=training_data
        )
    rf_params = rf_tuning.hyperparameter_tuning()
    #Best parameters found:  {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 10, 'n_estimators': 100}

    return 0

# Write dataset which contains all interval data to a csv
def write_all_data():

    ticker = 'AUDUSD=X'
    interval = '1m'
    csv_file_path = f"{ticker}.csv"  # Create a file name using the ticker symbol

    # Prepare the dataset and write it to the specified CSV file
    data = prepare_dataset(ticker=ticker, start_date='2024-08-26', end_date='2024-08-31', data_period='5d', interval=interval)
    data = data.round(5)
    data.to_csv(csv_file_path)
    

def main():
    
    """ FX """

    """ Input Parameters: 
        (str) ticker:               The ticker symbol for the currency pair
        (str) data_period:          The period of data to be used (5days by default)
        (str) interval:             The trading interval (1minute required for now)
        (float) confidence_level:   Confidence level for the machine learning probability (trade execution)
        (int) desired_model:        The specific model to use for simulation (1=MLP, 2=SVM, 3=RF, 4=LSTM)
        (int) simulation_range:     The range of the simulation in days (maximum 17)
        (int) rsi_oversold:         Min for RSI oversold condition
        (int) rsi_overbought        Min for RSI overbought condition
        (int) adx_extreme_val       Max value for adx extreme value (Threshold) 
        (int) DI_extreme_val        Max value for DI+/DI- extreme value (Threshold)
        (int) volatility_range      Minimum value of range between DI+ and DI-
        (int) stop_loss             Percentage stop loss value
    """

    GBPUSD = Run_Trading(ticker='GBPUSD=X', start_balance=1000, data_period='5d', interval='1m', confidence_level=0.5, desired_model=1, simulation_range=7,
                             rsi_oversold=30, rsi_overbought=70, adx_extreme_val=30, DI_extreme_val=75, volatility_range=15, min_di_level=10, stop_loss=0.15)
    balance = GBPUSD.run_trading_simulation()
    
    # Models: 1: 1664.8, 2: 1450.81, 3: 1271, 4: 1431.05, noml: 1459.74
    print("\nFinal Balance: ", balance)

    USDJPY = Run_Trading(ticker='USDJPY=X', start_balance=1000, data_period='5d', interval='1m', confidence_level=0.5, desired_model=1, simulation_range=7,
                             rsi_oversold=30, rsi_overbought=70, adx_extreme_val=30, DI_extreme_val=75, volatility_range=15, min_di_level=12, stop_loss=0.1)
    balance = USDJPY.run_trading_simulation()

    # Models: 1: 1145.74, 2: 1134.87, 3: 1528.24, 4: 977.468, noml: 1008.13 
    print("\nFinal Balance: ", balance)

    AUDUSD = Run_Trading(ticker='AUDUSD=X', start_balance=1000, data_period='5d', interval='1m', confidence_level=0.5, desired_model=1, simulation_range=7,
                             rsi_oversold=30, rsi_overbought=70, adx_extreme_val=30, DI_extreme_val=75, volatility_range=15, min_di_level=10, stop_loss=0.15)
    balance = AUDUSD.run_trading_simulation()

    # Models: 1: 1665.58, 2: 1664.49, 3: 1798.53, 4: 1657.2, noml: 1382.65
    print("\nFinal Balance: ", balance)


if __name__ == "__main__":
    main()
    #testing()
    #write_all_data()