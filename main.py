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
        (int) rsi_oversold:         Min for RSI oversold condition
        (int) rsi_overbought        Min for RSI overbought condition
        (int) adx_extreme_val       Max value for adx extreme value (Threshold) 
        (int) DI_extreme_val        Max value for DI+/DI- extreme value (Threshold)
        (int) volatility_range      Minimum value of range between DI+ and DI-
        (int) stop_loss             Percentage stop loss value
    """

    GBPUSD = Run_Trading(ticker='GBPUSD=X', data_period='5d', interval='1m', confidence_level=0.5, desired_model=1, simulation_range=17,
                             rsi_oversold=30, rsi_overbought=70, adx_extreme_val=40, DI_extreme_val=85, volatility_range=10, min_di_level=10, stop_loss=1)
    balance = GBPUSD.run_trading_simulation()

    """EURUSD = Run_Trading(ticker='EURUSD=X', data_period='5d', interval='1m', confidence_level=0.5, desired_model=1, simulation_range=17, 
                         rsi_oversold=27, rsi_overbought=73, adx_extreme_val=35, DI_extreme_val=75, volatility_range=10, min_di_level=10, stop_loss=1)
    balance = EURUSD.run_trading_simulation()"""

    """USDJPY = Run_Trading(ticker='USDJPY=X', data_period='5d', interval='1m', confidence_level=0.5, desired_model=1, simulation_range=17,
                             rsi_oversold=27, rsi_overbought=73, adx_extreme_val=35, DI_extreme_val=75, volatility_range=10, min_di_level=10, stop_loss=1)
    balance = USDJPY.run_trading_simulation()"""

    """AUDUSD = Run_Trading(ticker='AUDUSD=X', data_period='5d', interval='1m', confidence_level=0.5, desired_model=1, simulation_range=5,
                             rsi_oversold=27, rsi_overbought=73, adx_extreme_val=35, DI_extreme_val=75, volatility_range=10, min_di_level=10, stop_loss=1)
    balance = AUDUSD.run_trading_simulation()"""
    
    """EURJPY = Run_Trading(ticker='EURJPY=X', data_period='5d', interval='1m', confidence_level=0.5, desired_model=1, simulation_range=5,
                             rsi_oversold=27, rsi_overbought=73, adx_extreme_val=35, DI_extreme_val=75, volatility_range=10, min_di_level=10, stop_loss=1)
    balance = EURJPY.run_trading_simulation()"""

    """EURGBP = Run_Trading(ticker='EURGBP=X', data_period='5d', interval='1m', confidence_level=0.5, desired_model=1, simulation_range=5,
                             rsi_oversold=27, rsi_overbought=73, adx_extreme_val=35, DI_extreme_val=75, volatility_range=10, min_di_level=10, stop_loss=1)
    balance = EURGBP.run_trading_simulation()"""

    print("\nFinal Balance: ", balance)


if __name__ == "__main__":
    main()