""" Creates trading signals based on the technical indicator strategy to determine if a position should be taken on the forex pair """

# This class generates signals depending on current values of the security data, 
#...which implements the day trading strategy using technical indicators and price
class Signal_Generation():

    # Class constructor with trading paramater input conditions
    def __init__(self, rsi_oversold_level=30, rsi_overbought_level=70, adx_extreme_value=40, DI_extreme_val=80, 
                 volatility_range=10, min_di_level=10, stoploss_range=1):
        self.rsi_oversold_level = rsi_oversold_level        # RSI oversold level (Usually 30)
        self.rsi_overbought_level = rsi_overbought_level    # RSI overbought level (Usually 70)
        self.adx_extreme_value = adx_extreme_value          # Extreme adx values indicate strong trend, set threshold
        self.DI_extreme_val = DI_extreme_val                # Extreme DI value indicates extreme trend in DI+ or DI-
        self.volatility_range = volatility_range            # Desired volatility for DMI(ADX)
        self.min_di_level = min_di_level                    # Minimum DI+/DI- value needed for signal (below may indicate strong opposite trend)
        self.stoploss_range = stoploss_range                # Percentage stop loss for positions (absolute value)


    # This function generates buy or short signals given current data interval
    def signal_generation(self, security_data):
        # Create variables using the data from the dataset
        """ CAN REMOVE THIS FORMAT AND TAKE VALUES AS ALRADY DONE IN RUN_TRADING ON FUNCTION CALL """
        current_price = security_data.Close
        BB_upper = security_data.BB_upper
        BB_middle = security_data.BB_middle
        BB_lower = security_data.BB_lower
        rsi = security_data.RSI
        adx = security_data.ADX
        DI_pos = security_data._7
        DI_neg = security_data._8
        volatility = security_data.Volatility

        # Check data for buy or short condition (function calls)
        buy_signal = self.buy_signal(current_price, BB_lower, rsi, adx, DI_pos, DI_neg, volatility)
        short_signal = self.short_signal(current_price, BB_upper, rsi, adx, DI_pos, DI_neg, volatility)

        # If buy signal generated return 1 (indicating buy)
        if buy_signal:
            print("Buy Signal Generated!")
            return 1
            
        # If short signal generated return -1 (indicating short)
        if short_signal:
            print("Short Signal Generated!")
            return -1
        
        # Otherwise return 0 as no signal generated
        return 0
    

    # Buy signal generation
    def buy_signal(self, current_price, BB_lower, rsi, adx, DI_pos, DI_neg, volatility):
        # Check current values for buy signal generation, return True if satisfied
        if (BB_lower < current_price and rsi < self.rsi_oversold_level and volatility > self.volatility_range 
            and adx < self.adx_extreme_value and DI_neg < self.DI_extreme_val and DI_pos > self.min_di_level):

            return True     # Buy signal generated
        return False        # Return False as no signal present
    

    # Short signal generation
    def short_signal(self, current_price, BB_upper, rsi, adx, DI_pos, DI_neg, volatility):
        # Check current values for short signal generation
        if (BB_upper > current_price and rsi > self.rsi_overbought_level and volatility > self.volatility_range and 
            adx < self.adx_extreme_value and DI_pos < self.DI_extreme_val and DI_neg > self.min_di_level):

            return True     # Short signal generated
        return False        # Return False as no signal present
    

    # If position is open, checks criteria for closing or continuing to hold due to trend strength
    #...type is 1 if a buy position and -1 for a short position
    def close_position(self, security_data, position_type, prob_up, prob_down):
        # Create variables using the data from the dataset
        current_price = security_data.Close
        BB_upper = security_data.BB_upper
        BB_middle = security_data.BB_middle
        BB_lower = security_data.BB_lower
        rsi = security_data.RSI
        adx = security_data.ADX
        DI_pos = security_data._7
        DI_neg = security_data._8
        volatility = security_data.Volatility

        # Buy position
        if position_type == 1:
            if prob_up < 0.7:                          # If no strong probability of price continuing in trade direction 
                if (rsi > 50 and DI_neg>DI_pos):       # adx > 25 indicates a strong trend which means it may be better to hold
                    return True
        
        # Short position
        if position_type == -1:
            if prob_down < 0.7:                        # If no strong probability of price continuing in trade direction                                                
                if (rsi < 50 and DI_pos>DI_neg):       # adx > 25 indicates a strong trend which means it may be better to hold
                    # or DI_neg < DI_pos and adx < 20 ? Try this
                    return True

        return False
        

    # Check current price against entry price for stoploss margin forcing a closing of a position
    def stoploss(self, entry_price, current_price, position_type):
        # convert to percentage decimal
        stoploss_range = self.stoploss_range / 100
        # stoploss percentage below the buy entry price
        if (position_type == 1 and entry_price*(1-stoploss_range) > current_price):
            return True
        # stoploss percentage above the short entry price
        if (position_type == -1 and entry_price*(1+stoploss_range) < current_price):
            return True
        
        return False