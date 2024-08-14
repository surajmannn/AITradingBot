""" Main file for the project """

from ml_models.confidence_probability import *
from ta_indicators.dataset_preparation import *
from trading_system.run_trading import *

from alpha_vantage.foreignexchange import ForeignExchange
import pandas as pd

def main():

    """ CREATE OBJECTS FROM CLASS
    # Input parameters - (ticker), (interval), (window), (rsi_buy_boundary), (rsi_short_boundary), (confidence boundary), (stoploss range percentage)
    # Signalling input requires (close_short_rsi), (close_buy_rsi), (decimal precision of security price) """

    """ FX """


def testing():

    # Call to run the simulation
    GBPUSD = Run_Trading(
            ticker='GBPUSD=X',            # The ticker symbol for the currency pair
            data_period='5d',             # The period of data to be used (5days by default)
            interval='1m',                # The trading interval (1minute required for now)
            confidence_level=0.5,         # Confidence level for the machine learning probability (trade execution)
            desired_model=1,              # The specific model to use for simulation (1=MLP, 2=SVM, 3=RF, 4=LSTM)
            start_date='2024-07-01',      # The start date for the simulation
            end_date='2024-08-12',        # The end date for the simulation
            training_range=1,             # The range of training data in days
            simulation_range=5            # The range of the simulation in days (maximum 17)
    )
    balance = GBPUSD.run_trading_simulation()
    print("\nFinal Balance: ", balance)


if __name__ == "__main__":
    #main()
    testing()