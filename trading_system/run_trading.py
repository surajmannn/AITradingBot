""" This file contains the logic that runs the simulated trading environment and trading executions. """

from trading_system.signal_generation import *
from ta_indicators.dataset_preparation import *
from trading_system.buy_and_sell import *
from trading_system.ml_performance import *
from ml_models.confidence_probability import *
import time
import yahoo_fin.stock_info as si
from datetime import datetime, timedelta

# This class initialises a trading simulation instance with required parameters
class Run_Trading():

    # Class constructor
    def __init__(self, ticker, data_period='5d', interval='1m', confidence_level=0.5, desired_model=1, 
                 start_date=None, end_date=None, training_range=1, simulation_range=17, 
                 rsi_oversold=30, rsi_overbought=70, adx_extreme_val=35, DI_extreme_val=75, 
                 volatility_range=10, min_di_level=10, stop_loss=1):
        """ Simulation Params """
        self.ticker = ticker                                # (str) Security Ticker                                
        self.data_period = data_period                      # (str) The period of data to be used (5days by default)
        self.interval = interval                            # (str) The trading interval (1minute required for now)
        self.confidence_level = confidence_level            # (float) Confidence level for the machine learning probability (trade execution)
        self.desired_model = desired_model                  # (int) The specific model to use for simulation (1=MLP, 2=SVM, 3=RF, 4=LSTM)
        self.start_date = start_date                        # (str) The start date for the simulation
        self.end_date = end_date                            # (str) The end date for the simulation
        self.training_range = training_range                # (int) The range of training data in days
        self.simulation_range = simulation_range            # (int) The range of the simulation in days (maximum 17)

        """ Signaller Params """
        self.rsi_oversold = rsi_oversold                    # RSI Oversold value
        self.rsi_overbought = rsi_overbought                # RSI Overbought value
        self.adx_extreme_val = adx_extreme_val              # ADX Trend line extreme value (Threshold)
        self.DI_extreme_value = DI_extreme_val              # DI+ or DI_ extreme value (Threshold)
        self.volatility_range = volatility_range            # Minimum range between DI+ and DI- values
        self.min_di_level = min_di_level                    # Minimum DI+/DI- value needed
        self.stop_loss = stop_loss                          # Stop loss percentage

        self.trading_days = get_dates_list(self.ticker)     # Gets list of all trading days within simulation period
        self.number_trading_days = len(self.trading_days)   # Number of trading days in the simulation

        # Prepares the initial training dataset which is the 5days prior to the simulation start date
        self.initial_training_data = self.prepare_initial_training_data()

        # Initialises the machine learning model object to be used
        self.ml_model = self.initialise_ml_model()

        # Initialises a signalling object which handles signal generation during simulation
        self.signaller = self.initialise_signaller()
        
    """ Initialisation Functions """

    # Prepares the initial dataset
    def prepare_initial_training_data(self):
        start_date = self.trading_days[self.number_trading_days - (self.simulation_range + 5)]
        end_date = self.trading_days[self.number_trading_days - self.simulation_range]
        return prepare_dataset(
            ticker=self.ticker,
            start_date=start_date,
            end_date=end_date,
            data_period=self.data_period,
            interval=self.interval
        )
    
    # Intialises the machine learning model
    def initialise_ml_model(self):
        return Confidence_Probability(
            ticker=self.ticker,
            interval=self.interval,
            training_range=1,
            desired_model=self.desired_model,
            look_ahead_values=[5,15,30,45,60],
            dataset=self.initial_training_data
        )
    
    # Intialises the signal generator 
    def initialise_signaller(self):
        return Signal_Generation(
            rsi_oversold_level=self.rsi_oversold,
            rsi_overbought_level=self.rsi_overbought,
            adx_extreme_value=self.adx_extreme_val,
            DI_extreme_val=self.DI_extreme_value,
            volatility_range=self.volatility_range,
            min_di_level=self.min_di_level,
            stoploss_range=self.stop_loss
        )

    
    """ Simulation Functions"""

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

        # Test the model for performance metrics and initial metrics to database (first entry)
        if self.desired_model != 4:
            accuracy, roc_auc, sig_accuracy, sig_roc_auc  = self.ml_model.test_model()
        else:
            accuracy, roc_auc = self.ml_model.test_model()
            sig_accuracy = accuracy
            sig_roc_auc = roc_auc
        add_metrics(ticker=self.ticker, ml_model=self.desired_model, 
                    training_start_date=str(self.trading_days[self.number_trading_days-(self.simulation_range+5)]), 
                    training_end_date=str(self.trading_days[self.number_trading_days-self.simulation_range]), 
                    accuracy=float(accuracy), roc_auc=float(roc_auc), sig_accuracy=float(sig_accuracy), sig_roc_auc=float(sig_roc_auc)
        )

        # Create the initial desired model
        self.ml_model.create_model()

        # Security value variables for trade simulation
        position = 0        # Default 0 = no position (1: Buy, -1: Short)
        leverage = 30       # The leverage value for the trade
        trade_size = 500    # The value amount per trade
        balance = 1000      # Starting account balance
        entry_price = 0     # Security price on position opening
        lot_size = 0        # Size of position (pips)
        retrain = True      # Should the model retrain

        # Loop through each market trading day within desired trading range
        for x in range(start_index, self.number_trading_days):

            print("\nCURRENT TRADING DAY: ", self.trading_days[x], "\n")

            # Check if end of loop is reached which indicates the data should be for the current trading day
            if x == len(self.trading_days)-1:
                """if position != 0:
                    # Force close at end of trading day
                    if position == 1:   # Buy position
                        value = (lot_size*current_price) - (trade_size*leverage)
                    if position == -1:  # Short position
                        value = (trade_size*leverage) - (current_price*lot_size)
                    balance += value    # Adjust balance

                    # Add close to database
                    close(ticker=self.ticker, mla=self.desired_model, quantity=lot_size, security_price=current_price, 
                            total_price=(value+trade_size), profit=value, balance=balance, purchase_date=current_date, BB_upper=BB_upper, BB_lower=BB_lower, 
                            rsi=rsi, adx=adx, di_pos=DI_pos, di_neg=DI_neg, volatility=volatility
                    )
                    # Reset open position values
                    position = 0
                    entry_price = 0
                    lot_size = 0
                    print("Position Closed at: ", current_price)
                return balance"""
                current_trading_day_data = get_current_days_data(self.ticker, self.interval)    # Get data for latest trading day  
                retrain = False                                                                 # As on current day, do not retrain the model as this is the final simulated trading day
            
            # Otherwise run on current simulation day and obtain that days trading data
            else:
                current_trading_day_data = prepare_dataset(ticker=self.ticker, start_date=self.trading_days[x], end_date=self.trading_days[x+1], 
                                                           data_period='1d', interval=self.interval)

            # Loop through each row of interval data on current trading day
            for row in current_trading_day_data.itertuples(index=True, name='Pandas'):

                # Current trading interval values from the dataset
                current_date = str(row.Index)
                current_price = float(row.Close)
                BB_upper = float(row.BB_upper)
                BB_middle = float(row.BB_middle)
                BB_lower = float(row.BB_lower)
                rsi = float(row.RSI)
                adx = float(row.ADX)
                DI_pos = float(row._7)
                DI_neg = float(row._8)
                volatility = float(row.Volatility)

                # If position is currently open, only look for closing conditions
                if position != 0:
                    # Check probability of price direction
                    prob_up, prob_down = self.ml_model.confidence_rating(current_price, BB_upper, BB_middle, BB_lower, rsi, adx, DI_pos, DI_neg, volatility)

                    # Check open position price for stoploss condition (auto position close to mitigate losses)
                    if self.signaller.stoploss(entry_price, current_price, position):
                        if position == 1:   # Buy position
                            value = (lot_size*current_price) - (trade_size*leverage)
                        if position == -1:  # Short position
                            value = (trade_size*leverage) - (current_price*lot_size)
                        balance += value    # Adjust balance
                        # Add stop loss close to database
                        close_stoploss(ticker=self.ticker, mla=self.desired_model, quantity=lot_size, security_price=current_price, 
                              total_price=(value+trade_size), profit=value, balance=balance, purchase_date=current_date, BB_upper=BB_upper, BB_lower=BB_lower, 
                              rsi=rsi, adx=adx, di_pos=DI_pos, di_neg=DI_neg, volatility=volatility
                        )
                        # Reset open position values
                        position = 0
                        entry_price = 0
                        lot_size = 0
                        print("Position Stoploss at: ", current_price)

                    # Otherwise check values for closing condition
                    elif self.signaller.close_position(security_data=row, position_type=position, prob_up=prob_up, prob_down=prob_down):
                        if position == 1:   # Buy position
                            value = (lot_size*current_price) - (trade_size*leverage)
                        if position == -1:  # Short position
                            value = (trade_size*leverage) - (current_price*lot_size)
                        balance += value    # Adjust balance
                        
                        # Add close to database
                        close(ticker=self.ticker, mla=self.desired_model, quantity=lot_size, security_price=current_price, 
                              total_price=(value+trade_size), profit=value, balance=balance, purchase_date=current_date, BB_upper=BB_upper, BB_lower=BB_lower, 
                              rsi=rsi, adx=adx, di_pos=DI_pos, di_neg=DI_neg, volatility=volatility
                        )
                        # Reset open position values
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
                                    lot_size = (trade_size*leverage)/entry_price

                                    # Add buy to database
                                    buy(ticker=self.ticker, mla=self.desired_model, quantity=lot_size, security_price=entry_price, total_price=trade_size, 
                                        balance=balance, purchase_date=current_date, BB_upper=BB_upper, BB_lower=BB_lower, rsi=rsi, adx=adx, 
                                        di_pos=DI_pos, di_neg=DI_neg, volatility=volatility, confidence_probability=float(prob_up)
                                    )
                    
                            if signal == -1:
                                print("Confidence Probability: ", prob_down)
                                if prob_down > self.confidence_level:
                                    print("Shorted at: ", current_price)
                                    position = signal
                                    entry_price = current_price
                                    lot_size = (trade_size*leverage)/entry_price

                                    # Add sell to database
                                    sell(ticker=self.ticker, mla=self.desired_model, quantity=lot_size, security_price=entry_price, total_price=trade_size, 
                                        balance=balance, purchase_date=current_date, BB_upper=BB_upper, BB_lower=BB_lower, rsi=rsi, adx=adx, 
                                        di_pos=DI_pos, di_neg=DI_neg, volatility=volatility, confidence_probability=float(prob_down)
                                    )
            
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
        if self.desired_model != 4:
            accuracy, roc_auc, sig_accuracy, sig_roc_auc  = self.ml_model.test_model()
        else:
            accuracy, roc_auc = self.ml_model.test_model()
            sig_accuracy = accuracy
            sig_roc_auc = roc_auc

        # Add model performance metrics to database
        add_metrics(ticker=self.ticker, ml_model=self.desired_model, 
                    training_start_date=str(self.trading_days[start_day]), 
                    training_end_date=str(self.trading_days[end_day]), 
                    accuracy=float(accuracy), roc_auc=float(roc_auc), 
                    sig_accuracy=float(sig_accuracy), sig_roc_auc=float(sig_roc_auc)
        )
        self.ml_model.create_model()
        print("\n")


    # Runs the trading simulation environment without machine learning models (benchmarking)
    def run_noml_trading_simulation(self):

        if self.simulation_range > 17:
            print("\nERROR! Desired trading day range out of bounds!")
            return 0

        print(self.trading_days)

        # Obtain starting index for simulation
        start_index = self.starting_index()

        # Security value variables for trade simulation
        position = 0        # Default 0 = no position (1: Buy, -1: Short)
        leverage = 30       # The leverage value for the trade
        trade_size = 500    # The value amount per trade
        balance = 1000      # Starting account balance
        entry_price = 0     # Security price on position opening
        lot_size = 0        # Size of position (pips)

        # Loop through each market trading day within desired trading range
        for x in range(start_index, self.number_trading_days):

            print("\nCURRENT TRADING DAY: ", self.trading_days[x], "\n")

            # Check if end of loop is reached which indicates the data should be for the current trading day
            if x == len(self.trading_days)-1:
                current_trading_day_data = get_current_days_data(self.ticker, self.interval)    # Get data for latest trading day
                """if position == 1:   # Buy position
                    value = (lot_size*current_price) - (trade_size*leverage)
                if position == -1:  # Short position
                    value = (trade_size*leverage) - (current_price*lot_size)
                balance += value    # Adjust balance

                # Add close to database
                close(ticker=self.ticker, mla=self.desired_model, quantity=lot_size, security_price=current_price, 
                        total_price=(value+trade_size), profit=value, balance=balance, purchase_date=current_date, BB_upper=BB_upper, BB_lower=BB_lower, 
                        rsi=rsi, adx=adx, di_pos=DI_pos, di_neg=DI_neg, volatility=volatility
                )
                # Reset open position values
                position = 0
                entry_price = 0
                lot_size = 0
                print("Position Closed at: ", current_price)
                return balance""" 
            
            # Otherwise run on current simulation day and obtain that days trading data
            else:
                current_trading_day_data = prepare_dataset(ticker=self.ticker, start_date=self.trading_days[x], end_date=self.trading_days[x+1], 
                                                           data_period='1d', interval=self.interval)

            # Loop through each row of interval data on current trading day
            for row in current_trading_day_data.itertuples(index=True, name='Pandas'):

                # Current trading interval values from the dataset
                current_date = str(row.Index)
                current_price = float(row.Close)
                BB_upper = float(row.BB_upper)
                BB_middle = float(row.BB_middle)
                BB_lower = float(row.BB_lower)
                rsi = float(row.RSI)
                adx = float(row.ADX)
                DI_pos = float(row._7)
                DI_neg = float(row._8)
                volatility = float(row.Volatility)

                # If position is currently open, only look for closing conditions
                if position != 0:

                    # Check open position price for stoploss condition (auto position close to mitigate losses)
                    if self.signaller.stoploss(entry_price, current_price, position):
                        if position == 1:   # Buy position
                            value = (lot_size*current_price) - (trade_size*leverage)
                        if position == -1:  # Short position
                            value = (trade_size*leverage) - (current_price*lot_size)
                        balance += value    # Adjust balance
                        # Add stop loss close to database
                        close_stoploss(ticker=self.ticker, mla=self.desired_model, quantity=lot_size, security_price=current_price, 
                              total_price=(value+trade_size), profit=value, balance=balance, purchase_date=current_date, BB_upper=BB_upper, BB_lower=BB_lower, 
                              rsi=rsi, adx=adx, di_pos=DI_pos, di_neg=DI_neg, volatility=volatility
                        )
                        # Reset open position values
                        position = 0
                        entry_price = 0
                        lot_size = 0
                        print("Position Stoploss at: ", current_price)

                    # Otherwise check values for closing condition
                    elif self.signaller.close_position(security_data=row, position_type=position, prob_up=0, prob_down=0):
                        if position == 1:   # Buy position
                            value = (lot_size*current_price) - (trade_size*leverage)
                        if position == -1:  # Short position
                            value = (trade_size*leverage) - (current_price*lot_size)
                        balance += value    # Adjust balance
                        
                        # Add close to database
                        close(ticker=self.ticker, mla=self.desired_model, quantity=lot_size, security_price=current_price, 
                              total_price=(value+trade_size), profit=value, balance=balance, purchase_date=current_date, BB_upper=BB_upper, BB_lower=BB_lower, 
                              rsi=rsi, adx=adx, di_pos=DI_pos, di_neg=DI_neg, volatility=volatility
                        )
                        # Reset open position values
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
                            if signal == 1:
                                print("Bought at: ", current_price)  
                                position = signal
                                entry_price = current_price
                                lot_size = (trade_size*leverage)/entry_price

                                # Add buy to database
                                buy(ticker=self.ticker, mla=self.desired_model, quantity=lot_size, security_price=entry_price, total_price=trade_size, 
                                    balance=balance, purchase_date=current_date, BB_upper=BB_upper, BB_lower=BB_lower, rsi=rsi, adx=adx, 
                                    di_pos=DI_pos, di_neg=DI_neg, volatility=volatility, confidence_probability=0
                                )
                    
                            if signal == -1:
                                print("Shorted at: ", current_price)
                                position = signal
                                entry_price = current_price
                                lot_size = (trade_size*leverage)/entry_price

                                # Add sell to database
                                sell(ticker=self.ticker, mla=self.desired_model, quantity=lot_size, security_price=entry_price, total_price=trade_size, 
                                    balance=balance, purchase_date=current_date, BB_upper=BB_upper, BB_lower=BB_lower, rsi=rsi, adx=adx, 
                                    di_pos=DI_pos, di_neg=DI_neg, volatility=volatility, confidence_probability=0
                                )

            # End of current trading day
            print("\n END OF TRADING DAY!")
            print("CURRENT BALANCE: ", balance)

        return balance
        

    # Retrieves the first datetime for the training dataset which should be 7*training range behind the inputted start_date
    def get_training_start_date(start_date, training_range):
        # Convert input start date to datetime
        training_start_date = (datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=7 * training_range)).date()

        return training_start_date