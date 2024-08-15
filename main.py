""" Main file for the project """

from ml_models.confidence_probability import *
from ta_indicators.dataset_preparation import *
from trading_system.run_trading import *
import pandas as pd

def testing():
    """ Testing functions"""
    

def main():
    
    """ FX """

    """ Input Parameters: 
        (str) ticker:               The ticker symbol for the currency pair
        (str) data_period:          The period of data to be used (5days by default)
        (str) interval:             The trading interval (1minute required for now)
        (float) confidence_level:   Confidence level for the machine learning probability (trade execution)
        (int) desired_model:        The specific model to use for simulation (1=MLP, 2=SVM, 3=RF, 4=LSTM)
        (int) simulation_range:     The range of the simulation in days (maximum 17) 
    """

    GBPUSD = Run_Trading(ticker='GBPUSD=X', data_period='5d', interval='1m', confidence_level=0.5, desired_model=1, simulation_range=5)

    #EURUSD = Run_Trading(ticker='EURUSDY=X', data_period='5d', interval='1m', confidence_level=0.5, desired_model=1, simulation_range=5)

    #USDJPY = Run_Trading(ticker='USDJPY=X', data_period='5d', interval='1m', confidence_level=0.5, desired_model=1, simulation_range=5)

    #AUDUSD = Run_Trading(ticker='AUDUSD=X', data_period='5d', interval='1m', confidence_level=0.5, desired_model=1, simulation_range=5)
    
    #EURJPY = Run_Trading(ticker='EURJPY=X', data_period='5d', interval='1m', confidence_level=0.5, desired_model=1, simulation_range=5)

    #EURGBP = Run_Trading(ticker='EURGBPY=X', data_period='5d', interval='1m', confidence_level=0.5, desired_model=1, simulation_range=5)

    balance = GBPUSD.run_trading_simulation()
    print("\nFinal Balance: ", balance)


if __name__ == "__main__":
    main()