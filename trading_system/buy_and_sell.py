""" Execution logic to execute a buy or sell of stock through the database """

from database_models.simple_trades import *

# This function sends a buy execution into the database
def buy(ticker, ml_type, position_date, security_price, BB_upper, BB_lower, rsi, adx, di_pos, di_neg, volatility, confidence_probability):
    userID = 1          # irrelevant for test scenario
    position = 1        # 1 indicates Buy position
    quantity = 10
    total_price = round(security_price * quantity, 2)
    values = (userID, ticker, ml_type, position_date, position, quantity, security_price, total_price, BB_upper, BB_lower, rsi, adx, di_pos, di_neg, volatility, confidence_probability)
    data = add_position(values)     # database handler call
    return data


# This function sends a sell execution into the database
def sell(ticker, ml_type, position_date, security_price, BB_upper, BB_lower, rsi, adx, di_pos, di_neg, volatility, confidence_probability):
    userID = 1          # irrelevant for testing scenario
    position = -1        # 2 indicates Sell position
    quantity = 10
    total_price = round(security_price * quantity, 2)
    values = (userID, ticker, ml_type, position_date, position, quantity, security_price, total_price, BB_upper, BB_lower, rsi, adx, di_pos, di_neg, volatility, confidence_probability)
    data = add_position(values)     # database handler call
    return data

# This function sends close signal to database
def close(ticker, ml_type, position_date, security_price, BB_upper, BB_lower, rsi, adx, di_pos, di_neg, volatility, confidence_probability):
    userID = 1          # irrelevant for testing scenario
    position = 0        # 0 indicates close position
    quantity = 10
    total_price = round(security_price * quantity, 2)
    values = (userID, ticker, ml_type, position_date, position, quantity, security_price, total_price, BB_upper, BB_lower, rsi, adx, di_pos, di_neg, volatility, confidence_probability)
    data = add_position(values)     # database handler call
    return data

# This function sends close signal to database due to stoploss
def close_stoploss(ticker, ml_type, position_date, security_price, BB_upper, BB_lower, rsi, adx, di_pos, di_neg, volatility, confidence_probability):
    userID = 1          # irrelevant for testing scenario
    position = -2        # -2 indicates close position due to stop loss
    quantity = 10
    total_price = round(security_price * quantity, 2)
    values = (userID, ticker, ml_type, position_date, position, quantity, security_price, total_price, BB_upper, BB_lower, rsi, adx, di_pos, di_neg, volatility, confidence_probability)
    data = add_position(values)     # database handler call
    return data