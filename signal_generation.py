""" Trading Logic File """

""" Creates trading signals based on the technical indicator strategy to determine if a position should be taken on forex pair """

from technical_indicators.bollinger_bands import *
from ta_indicators.bollinger_bands import *
from technical_indicators.RSI import *
from ta_indicators.rsi import *
from technical_indicators.dmi_adx import *
from ta_indicators.dmi_adx import *
from ta_indicators.rsi import *
from buy_and_sell import *
from ml_models.confidence_rating import *
import time
import yahoo_fin.stock_info as si

class Signal_Generation():

    # Class constructor with trading paramater inputs
    def __init__(self, ticker, interval, window, rsi_oversold_level, rsi_overbought_level, stoploss_range):
        self.ticker = ticker                                # Stock ticker
        self.interval = interval                            # Trading interval (e.g. 1min, 5min, 10min, 1hour, 4hour etc...)
        self.window = window                                # Look back period
        self.rsi_oversold_level = rsi_oversold_level        # RSI oversold level (Usually 30)
        self.rsi_overbought_level = rsi_overbought_level    # RSI overbought level (Usually 70)
        self.stoploss_range = stoploss_range                # Percentage stop loss for positions


    def signalling():

        """ 
        FOR BUY/SHORT:
        BB Above or below band with supporting RSI reading
        Above average gap between DI+ and DI- and No extreme ADX value, i.e. 35+ ---> Then produce signal
        """

        """
        FOR CLOSE:
        RSI above/below 50 and ADX below 25, otherwise continue holding as trend present.
        """

        # use loop, or return?
        return 0
        

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