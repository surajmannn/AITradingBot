""" Trading Logic File """

""" Runs on live market, creates buy and short signals on selected securities and 
    runs MLP Regression model for confidence of price changing direction """

from bollinger_bands import *
from RSI import *
from buy_and_sell import *
from confidence_rating import *
import time
import yahoo_fin.stock_info as si

class Signal_Handler():

    # Class constructor with trading paramater inputs
    def __init__(self, ticker, interval, window, rsi_oversold_level, rsi_overbought_level, confidence_boundary, stoploss_range):
        self.ticker = ticker                                # Stock ticker
        self.interval = interval                            # Trading interval (e.g. 1min, 5min, 10min, 1hour, 4hour etc...)
        self.window = window                                # Look back period
        self.rsi_oversold_level = rsi_oversold_level        # RSI oversold level (Usually 30)
        self.rsi_overbought_level = rsi_overbought_level    # RSI overbought level (Usually 70)
        self.confidence_boundary = confidence_boundary      # Confidence rating boundary required for trade
        self.stoploss_range = stoploss_range                # Percentage stop loss for positions

        # CALL Confidence rating to initialise new model object
        self.confidence_model = Confidence_Rating(ticker, interval,  window)


    # This function runs live trading for selected security, trading interval and look back periods
    def signalling(self, close_rsi_short, close_rsi_buy, precision):

        # Set object parameters
        interval = self.interval
        ticker = self.ticker
        window = self.window
        rsi_oversold_level = self.rsi_oversold_level
        rsi_overbought_level = self.rsi_overbought_level
        confidence_boundary = self.confidence_boundary

        # Set model to object model instance
        confidence_model = self.confidence_model

        # add timing contraints
        #start_date = pd.Timestamp.today().strftime('%Y-%m-%d')
        # Wait until the next minute starts
        current_time = pd.Timestamp.now()
        next_minute = (current_time + pd.Timedelta(minutes=1)).replace(second=0, microsecond=0)
        print("Waiting for the next minute to start....")
        time.sleep((next_minute - current_time).seconds)

        # Is there an open position? Boolean
        position_status = False

        # Initialise position variables
        entry_price = 0
        position_type = 0

        # Begin trading
        while True:

            # Call RSI and Bollinger Band Functions to recieve readings at current price
            rsi, last_time = RSI(ticker, interval, window)
            bbu, bbl = bollinger_bands(ticker, interval)

            # Get the current market price using yahoo_fin
            current_price = si.get_live_price(ticker)

            # Set decimal precisions
            current_price = round(current_price, precision)
            rsi = round(rsi, 2)
            bbu = round(bbu, precision)
            bbl = round(bbl, precision)

            # Print all values at each time interval
            print("\n", ticker)
            print(last_time)
            print(f"Current price: ${current_price}")
            print(f"RSI: {rsi}")
            print(f"BB Upper: {bbu}")
            print(f"BB Lower: {bbl}")

            # Check RSI/Bollinger lower band
            if (rsi < rsi_oversold_level and bbl > current_price and position_status == False):
                # if rsi and bbl within buy boundary, check confidence rating with MLP
                buy_confidence = confidence_model.confidence_rating(current_price, rsi, bbu, bbl)
                print("Confidence Rating: ", buy_confidence)
                if (buy_confidence > confidence_boundary):
                    # if mlp confidence boundary met, create buy order
                    buy_signal = buy(ticker, 10, float(current_price), float(rsi), float(bbl), float(bbu), float(buy_confidence))
                    print("\n", ticker, " Buy order filled.")
                    print(buy_signal)
                    position_status = True      # Open position True
                    entry_price = current_price # entry price = current price
                    position_type = 1           # Buy order
                else:
                    print("Confidence too low.")

            # Check RSI Sell
            elif (rsi > rsi_overbought_level and bbu < current_price and position_status == False):
                # if rsi and bbl within sell boundary, check confidence rating with MLP
                short_confidence = confidence_model.confidence_rating(current_price, rsi, bbu, bbl)
                print("Confidence Rating: ", short_confidence)
                if (short_confidence > confidence_boundary):
                    # if mlp confidence boundary met, create short order
                    sell_signal = sell(ticker, 10, float(current_price), float(rsi), float(bbl), float(bbu), float(short_confidence))
                    print("\n", ticker, " Sell order filled.")
                    print(sell_signal)
                    position_status = True      # Open position True
                    entry_price = current_price # entry price = current price
                    position_type = 2           # Sell order
                else:
                    print("Confidence too low.")

            # Compare current position with current market price. Trigger Stop Loss at inputted threshold 
            elif (position_status == True and self.stoploss(entry_price, current_price, position_type) == True):
                close_position = close_stoploss(ticker, 10, float(current_price), float(rsi), float(bbl), float(bbu), 0)
                print("\n", ticker, " position closed due to stop loss.")
                print(close_position)
                position_status = False     # set position to False
                entry_price = 0             # reset
                position_type = 0           # reset

            # Close buy position
            elif (position_type == 1 and rsi > close_rsi_buy):
                # if RSI condition met and currently position is open then close position
                close_position = close(ticker, 10, float(current_price), float(rsi), float(bbl), float(bbu), 0)
                print("\n", ticker, " position closed.")
                print(close_position)
                position_status = False     # set position to False
                entry_price = 0             # reset
                position_type = 0           # reset

            
            # Close sell position
            elif (position_type == 2 and rsi < close_rsi_short):
                # if RSI condition met and currently position is open then close position
                close_position = close(ticker, 10, float(current_price), float(rsi), float(bbl), float(bbu), 0)
                print("\n", ticker, " position closed.")
                print(close_position)
                position_status = False     # set position to False
                entry_price = 0             # reset
                position_type = 0           # reset

            # Set loop interval period
            time.sleep(60)


    # Check current price against entry price for stoploss margin forcing a closing of a position
    def stoploss(self, entry_price, current_price, position_type):
        # convert to percentage decimal
        stoploss_range = self.stoploss_range / 100
        # stoploss percentage below the buy entry price
        if (position_type == 1 and entry_price*(1-stoploss_range) > current_price):
            return True
        # stoploss percentage below the short entry price
        elif (position_type == 2 and entry_price*(1+stoploss_range) < current_price):
            return True
        else:
            return False