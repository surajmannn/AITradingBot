""" This file contains the logic that runs the simulated trading environment and trading executions. """

from trading_system.signal_generation import *
from ta_indicators.dataset_preparation import *
from trading_system.buy_and_sell import *
from ml_models.confidence_probability import *
import time
import yahoo_fin.stock_info as si
from datetime import datetime, timedelta

# Retrieves the first datetime for the training dataset which should be 7*training range behind the inputted start_date
def get_training_start_date(start_date, training_range):
    # Convert input start date to datetime
    training_start_date = (datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=7 * training_range)).date()

    return training_start_date


# Obtains the initial starting index for the simulation
#... Max simulation is 17days (inidicated by 3 weeks), otherwise start index current day subtracted by 5d trading week range
def starting_index(simulation_range, trading_days):
    if simulation_range == 3:
        start_index = 6
    else:
        start_index = trading_days - ((simulation_range*5))
    return start_index


# Prepare initial training dataset based on desired simulation range
def get_initial_training_data(ticker, simulation_range, data_period, interval):
    initial_training_data = 0
    return initial_training_data


# Runs the trading simulation environment
def run_trading_simulation(ticker, desired_model, start_date, end_date, training_range, simulation_range, data_period, interval):

    # Get list of the valid open forex trading days 
    trading_days = get_dates_list(ticker)
    print(trading_days)

    # Obtain starting index for simulation
    start_index = starting_index(simulation_range, len(trading_days))

    if simulation_range < 3:
        print("\nCURRENT TRAINING RANGE: ", trading_days[-5-(simulation_range*5)], " TO ", trading_days[-(simulation_range*5)], "\n")
        initial_training_data = prepare_dataset(ticker=ticker, start_date=trading_days[-5-(simulation_range*5)], end_date=trading_days[-(simulation_range*5)], 
                                                data_period=data_period, interval=interval)
    elif simulation_range == 3:
        print("\nCURRENT TRAINING RANGE: ", trading_days[-7-(simulation_range*5)], " TO ", trading_days[-2-(simulation_range*5)], "\n")
        # Get initial training data which should be 5 trading days of data prior to the starting day of the simulation
        initial_training_data = prepare_dataset(ticker=ticker, start_date=trading_days[-7-(simulation_range*5)], end_date=trading_days[-2-(simulation_range*5)], 
                                                data_period=data_period, interval=interval)
    else:
        print("Desired simulation range out of API range!")
        return 0
    

    # Create desired model for training on historic ticker data
    ml_model = Confidence_Probability(ticker=ticker, interval=interval, training_range=1, desired_model=desired_model, look_ahead_values=[2,5,10,15,30], dataset=initial_training_data)

    # Test the model for performance metrics
    ml_model.test_model()

    ml_model.create_model()

    # Create signalling object for trading signalling
    signaller = Signal_Generation(rsi_oversold_level=30, rsi_overbought_level=70, adx_extreme_value=40, volatility_range=10, stoploss_range=2)

    position = 0
    balance = 1000
    entry_price = 0
    lot_size = 0
    retrain = True

    for x in range(start_index, len(trading_days)):

        print("\nCURRENT TRADING DAY: ", trading_days[x], "\n")
        if x == len(trading_days)-1:
            current_trading_day_data = get_current_days_data(ticker, interval)
            retrain = False
        else:
            current_trading_day_data = prepare_dataset(ticker=ticker, start_date=trading_days[x], end_date=trading_days[x+1], data_period='1d', interval=interval)

        for row in current_trading_day_data.itertuples(index=True, name='Pandas'):

            current_price = row.Close
            BB_upper = row.BB_upper
            BB_middle = row.BB_middle
            BB_lower = row.BB_lower
            rsi = row.RSI
            adx = row.ADX
            DI_pos = row._7
            DI_neg = row._8
            volatility = row.Volatility

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

        # Update model
        print("\n END OF TRADING DAY!")
        print("CURRENT BALANCE: ", balance)

        if retrain:
            print("\nNEW TRAINING RANGE: ", trading_days[x-4], " TO ", trading_days[x+1], "\n")
            new_training_data = prepare_dataset(ticker=ticker, start_date=trading_days[x-5], end_date=trading_days[x], data_period='5d', interval=interval)
            ml_model.update_training_data(new_training_data=new_training_data)
            ml_model.test_model()
            ml_model.create_model()
            print("\n")

    return balance