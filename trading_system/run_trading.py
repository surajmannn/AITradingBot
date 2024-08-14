""" This file contains the logic that runs the simulated trading environment and trading executions. """

from trading_system.signal_generation import *
from ta_indicators.dataset_preparation import *
from trading_system.buy_and_sell import *
from ml_models.confidence_probability import *
import time
import yahoo_fin.stock_info as si
from datetime import datetime, timedelta

# This class initialises a trading simulation instance with required parameters
class Run_Trading():

    def __init__(self, ticker, data_period, interval, confidence_level, desired_model, start_date, end_date, training_range, simulation_range):
        self.ticker = ticker                                # Security Ticker
        self.data_period = data_period                      # The period of data to be used (5days by default)
        self.interval = interval                            # The trading interval (1minute required for now)
        self.confidence_level = confidence_level            # Confidence level for the machine learning probability (trade execution)
        self.desired_model = desired_model                  # The specific model to use for simulation (1=MLP, 2=SVM, 3=RF, 4=LSTM)
        self.start_date = start_date                        # The start date for the simulation
        self.end_date = end_date                            # The end date for the simulation
        self.training_range = training_range                # The range of training data in days
        self.simulation_range = simulation_range            # The range of the simulation in days (maximum 17)

        self.trading_days = get_dates_list(self.ticker)     # Gets list of all trading days within simulation period
        self.number_trading_days = len(self.trading_days)   # Number of trading days in the simulation

        # Prepares the initial training dataset which is the 5days prior to the simulation start date
        initial_training_data = prepare_dataset(
            ticker=self.ticker, 
            start_date=self.trading_days[self.number_trading_days-(self.simulation_range+5)], 
            end_date=self.trading_days[self.number_trading_days-self.simulation_range], 
            data_period=self.data_period, interval=self.interval
        )

        # Initialises the machine learning model object to be used
        self.ml_model = Confidence_Probability(
            ticker=self.ticker, 
            interval=self.interval, 
            training_range=1, 
            desired_model=self.desired_model, 
            look_ahead_values=[2,5,10,15,30],   # Values to be used as look ahead time intervals
            dataset=initial_training_data
        )

        # Initialises a signalling object which handles signal generation during simulation
        self.signaller = Signal_Generation(
            rsi_oversold_level=30, 
            rsi_overbought_level=70, 
            adx_extreme_value=40, 
            volatility_range=10, 
            stoploss_range=2)

    
    # Obtains the initial starting index for the simulation
    #... Max simulation is 17days, so obtains relevant index reflecting desired simulation range
    def starting_index(self):
        start_index = self.number_trading_days-self.simulation_range
        return start_index
    

    # Runs the trading simulation environment
    def run_trading_simulation(self):

        if self.simulation_range > 17:
            print("\nERROR! Desired trading day range out of bounds!")
            return 0

        print(self.trading_days)

        # Obtain starting index for simulation
        start_index = self.starting_index()

        # Prepare the initial training dataset which is 5days prior to simulation starting range
        print("\nCURRENT TRAINING RANGE: ", self.trading_days[self.number_trading_days-(self.simulation_range+5)], " TO ", self.trading_days[self.number_trading_days-self.simulation_range], "\n")

        # Test the model for performance metrics
        self.ml_model.test_model()
        # Create the initial desired model
        self.ml_model.create_model()

        # Security value variables for simulation
        position = 0
        balance = 1000
        entry_price = 0
        lot_size = 0
        retrain = True

        # Loop through each market trading day within desired trading range
        for x in range(start_index, self.number_trading_days):

            print("\nCURRENT TRADING DAY: ", self.trading_days[x], "\n")

            # Check if end of loop is reached which indicates the data should be for the current trading day
            if x == len(self.trading_days)-1:
                current_trading_day_data = get_current_days_data(self.ticker, self.interval)    # Get data for latest trading day  
                retrain = False                                                                 # As on current day, do not retrain the model as this is the final simulated trading day    
            
            # Otherwise run on current simulation day and obtain that days trading data
            else:
                current_trading_day_data = prepare_dataset(ticker=self.ticker, start_date=self.trading_days[x], end_date=self.trading_days[x+1], 
                                                           data_period='1d', interval=self.interval)

            # Loop through each row of interval data on current trading day
            for row in current_trading_day_data.itertuples(index=True, name='Pandas'):

                # Current trading interval values from the dataset
                current_price = row.Close
                BB_upper = row.BB_upper
                BB_middle = row.BB_middle
                BB_lower = row.BB_lower
                rsi = row.RSI
                adx = row.ADX
                DI_pos = row._7
                DI_neg = row._8
                volatility = row.Volatility

                # If position is currently open, only look for closing conditions
                if position != 0:
                    if self.signaller.close_position(security_data=row, position_type=position):
                        if position == 1:
                            balance += (lot_size*current_price - 500)*30
                        if position == -1:
                            balance += (500 - current_price*lot_size)*30
                        position = 0
                        entry_price = 0
                        lot_size = 0
                        print("Position Closed at: ", current_price)

                # If no position is open, then look to generate signals
                else:
                    if position == 0:
                        # Check current minute data for signal
                        signal = self.signaller.signal_generation(security_data=row)
                        if signal != 0:
                            prob_up, prob_down = self.ml_model.confidence_rating(current_price, BB_upper, BB_middle, BB_lower, rsi, adx, DI_pos, DI_neg, volatility)
                            if signal == 1:
                                print("Confidence Probability: ", prob_up)
                            if prob_up > self.confidence_level:
                                print("Bought at: ", current_price)  
                                position = signal
                                entry_price = current_price
                                lot_size = 500/entry_price
                    
                            if signal == -1:
                                print("Confidence Probability: ", prob_down)
                                if prob_down > self.confidence_level:
                                    print("Shorted at: ", current_price)
                                    position = signal
                                    entry_price = current_price
                                    lot_size = 500/entry_price

            # End of current trading day
            print("\n END OF TRADING DAY!")
            print("CURRENT BALANCE: ", balance)

            # Retrain the model on the most recent 5 days of data
            if retrain:
                self.retrain(start_day=x-4, end_day=x+1)

        return balance
    

    # This function retrains the current model for trading simulation
    def retrain(self, start_day, end_day):
        print("\nNEW TRAINING RANGE: ", self.trading_days[start_day], " TO ", self.trading_days[end_day], "\n")
        new_training_data = prepare_dataset(ticker=self.ticker, start_date=self.trading_days[start_day], end_date=self.trading_days[end_day], 
                                            data_period='5d', interval=self.interval)
        self.ml_model.update_training_data(new_training_data=new_training_data)
        self.ml_model.test_model()
        self.ml_model.create_model()
        print("\n")
        

    # Retrieves the first datetime for the training dataset which should be 7*training range behind the inputted start_date
    def get_training_start_date(start_date, training_range):
        # Convert input start date to datetime
        training_start_date = (datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=7 * training_range)).date()

        return training_start_date