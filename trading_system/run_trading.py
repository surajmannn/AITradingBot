""" This file contains the logic that runs the simulated trading environment and trading executions. """

from trading_system.signal_generation import *
from ta_indicators.dataset_preparation import *
from trading_system.buy_and_sell import *
from ml_models.confidence_probability import *
import time
import yahoo_fin.stock_info as si


def run_trading_simulation(ticker, trading_range, data_period, interval):

    # Call to dataset preparation function using ticker, data period (e.g. 1d, 5d), trading interval (e.g. 1m, 5m)   
    trading_dataset = prepare_dataset(ticker=ticker, data_period=data_period, interval=interval)

    # Create desired model for training on historic ticker data
    ml_model = Confidence_Probability(ticker=ticker, interval=interval, training_range=1, desired_model=1, look_ahead_values=[2,5,10,15,30])

    # Test the model for performance metrics
    ml_model.test_model()

    # Create signalling object for trading signalling
    signaller = Signal_Generation(rsi_oversold_level=30, rsi_overbought_level=70, adx_extreme_value=40, volatility_range=10, stoploss_range=2)

    position = 0
    balance = 1000
    entry_price = 0
    lot_size = 0

    for row in trading_dataset.itertuples(index=True, name='Pandas'):

        current_price = row.Close
        BB_upper = row.BB_upper
        BB_middle = row.BB_middle
        BB_lower = row.BB_lower
        rsi = row.RSI
        adx = row.ADX
        DI_pos = row._7
        DI_neg = row._8
        volatility = row.Volatility

        #print("\nCurrent Time: ", row.Index)
        #print(row)

        if position != 0:
            if signaller.close_position(security_data=row, position_type=position):
                if position == 1:
                    balance += (lot_size*current_price - 500)*30
                if position == -1:
                    balance += (500 - current_price*lot_size)*30
                position = 0
                entry_price = 0
                lot_size = 0
                print("Position Closed at: ", current_price)

        else:
            if position == 0:
                # Check current minute data for signal
                signal = signaller.signal_generation(security_data=row)
                if signal != 0:
                    prob_up, prob_down = ml_model.confidence_rating(current_price, BB_upper, BB_middle, BB_lower, rsi, adx, DI_pos, DI_neg, volatility)
                    if signal == 1:
                        print("Confidence Probability: ", prob_up)
                    if prob_up > 0.5:
                        print("Bought at: ", current_price)  
                        position = signal
                        entry_price = current_price
                        lot_size = 500/entry_price
            
                    if signal == -1:
                        print("Confidence Probability: ", prob_down)
                        if prob_down > 0.5:
                            print("Shorted at: ", current_price)
                            position = signal
                            entry_price = current_price
                            lot_size = 500/entry_price

    """ Create dataset for desired range
        Create trained machine learning model
        Create signal object 
        Create trading simulation time frame loop
        Run on each interval and check for signals: if signal then check ML probability """

    """ Second Logic Challenge: Work out how to run day by day. At end of each day,
        append newest day to machine learning training set and remove oldest day. Retrain the model and run again for the next day. 
        Append newest day to reduce computational load """

    """ After this do hyperparameter tuning. """

    return balance