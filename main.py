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

    # Check dataset in prepare data
    #print(prepare_training_dataset('GBPUSD=X', '1m', 2))

    # Check model training
    #GBP = Confidence_Probability('GBPUSD=X', '1m', 1, 4, [2,5,10,15,30])
    #data = GBP.create_training_data()
    #print(data.loc[data.index.month == 8])
    #GBP.test_model()

    balance = run_trading_simulation(ticker='GBPUSD=X', start_date='2024-07-01', end_date='2024-08-12', training_range=1, data_period='5d', interval='1m')
    print("\nFinal Balance: ", balance)


if __name__ == "__main__":
    #main()
    testing()