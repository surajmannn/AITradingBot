""" Execution logic to execute a buy or sell of stock through the database """

from database_models.simple_trades import *

# This function sends a buy execution into the database
def buy(ticker, mla, quantity, security_price, total_price, balance, purchase_date, BB_upper, BB_lower, rsi, adx, di_pos, di_neg, volatility, confidence_probability):
    userID = 1          # irrelevant for test scenario
    position = 1        # 1 indicates Buy position
    values = (userID, ticker, mla, position, quantity, security_price, total_price, balance, purchase_date, BB_upper, BB_lower, rsi, adx, di_pos, di_neg, volatility, confidence_probability)
    data = add_position(values)     # database handler call
    return data

# This function sends a sell execution into the database
def sell(ticker, mla, quantity, security_price, total_price, balance, purchase_date, BB_upper, BB_lower, rsi, adx, di_pos, di_neg, volatility, confidence_probability):
    userID = 1          # irrelevant for testing scenario
    position = -1        # 2 indicates Sell position
    values = (userID, ticker, mla, position, quantity, security_price, total_price, balance, purchase_date, BB_upper, BB_lower, rsi, adx, di_pos, di_neg, volatility, confidence_probability)
    data = add_position(values)     # database handler call
    return data

# This function sends close signal to database
def close(ticker, mla, quantity, security_price, total_price, balance, purchase_date, BB_upper, BB_lower, rsi, adx, di_pos, di_neg, volatility):
    userID = 1                      # irrelevant for testing scenario
    position = 0                    # 0 indicates close position
    confidence_probability = 0      # No confidence probaility on close position
    values = (userID, ticker, mla, position, quantity, security_price, total_price, balance, purchase_date, BB_upper, BB_lower, rsi, adx, di_pos, di_neg, volatility, confidence_probability)
    data = add_position(values)     # database handler call
    return data

# This function sends close signal to database due to stoploss
def close_stoploss(ticker, mla, quantity, security_price, total_price, balance, purchase_date, BB_upper, BB_lower, rsi, adx, di_pos, di_neg, volatility):
    userID = 1                      # irrelevant for testing scenario
    position = -2                   # -2 indicates close position due to stop loss
    confidence_probability = 0      # No confidence probability on close
    values = (userID, ticker, mla, position, quantity, security_price, total_price, balance, purchase_date, BB_upper, BB_lower, rsi, adx, di_pos, di_neg, volatility, confidence_probability)
    data = add_position(values)     # database handler call
    return data