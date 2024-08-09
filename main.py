""" Main file for the project """

from signals_handler import Signal_Handler
from ml_models.confidence_probability import *
from ta_indicators.dataset_preparation import *
from trading_system.run_trading import *

def main():

    """ CREATE OBJECTS FROM CLASS
    # Input parameters - (ticker), (interval), (window), (rsi_buy_boundary), (rsi_short_boundary), (confidence boundary), (stoploss range percentage)
    # Signalling input requires (close_short_rsi), (close_buy_rsi), (decimal precision of security price) """

    """ FX """
    GBPUSD = Signal_Handler('GBPUSD=X', '1m', 14, 27, 73, 0.4, 0.25)
    GBPUSD.signalling(35.1, 64.9, 6)

    GBPEUR = Signal_Handler('GBPEUR=X', '1m', 14, 27, 73, 0.4, 0.25)
    GBPEUR.signalling(35.1, 64.9, 6)

    AUDJPY = Signal_Handler('AUDJPY=X', '1m', 14, 27, 73, 0.4, 0.25)
    AUDJPY.signalling(35.1, 64.9, 4)


def testing():

    # Check dataset in prepare data
    #print(prepare_training_dataset('GBPUSD=X', '1m', 2))

    # Check model training
    #GBP = Confidence_Probability('GBPUSD=X', '1m', 1, 4, [2,5,10,15,30])
    #data = GBP.create_training_data()
    #print(data.loc[data.index.month == 8])
    #GBP.test_model()

    balance = run_trading_simulation('GBPUSD=X', trading_range=1, data_period='5d', interval='1m')
    print("\nFinal Balance: ", balance)



if __name__ == "__main__":
    #main()
    testing()
