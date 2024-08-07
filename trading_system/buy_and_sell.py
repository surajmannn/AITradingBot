""" Execution logic to execute a buy or sell of stock through the database """

from database_models.simple_trades import *

# This function sends a buy execution into the database
def buy(ticker, quantity, stock_price, RSI, BBLB, BBUB, confidence_raiting):
    userID = 1          # irrelevant for test scenario
    position = 1        # 1 indicates Buy position
    total_price = round(stock_price * quantity, 2)
    values = (userID, ticker, position, quantity, stock_price, total_price, RSI, BBLB, BBUB, confidence_raiting)
    data = add_position(values)     # database handler call
    return data


# This function sends a sell execution into the database
def sell(ticker, quantity, stock_price, RSI, BBLB, BBUB, confidence_raiting):
    userID = 1          # irrelevant for testing scenario
    position = 2        # 2 indicates Sell position
    total_price = round(stock_price * quantity, 2)
    values = (userID, ticker, position, quantity, stock_price, total_price, RSI, BBLB, BBUB, confidence_raiting)
    data = add_position(values)     # database handler call
    return data

# This function sends close signal to database
def close(ticker, quantity, stock_price, RSI, BBLB, BBUB, confidence_raiting):
    userID = 1          # irrelevant for testing scenario
    position = 0        # 0 indicates close position
    total_price = round(stock_price * quantity, 2)
    values = (userID, ticker, position, quantity, stock_price, total_price, RSI, BBLB, BBUB, confidence_raiting)
    data = add_position(values)     # database handler call
    return data

# This function sends close signal to database due to stoploss
def close_stoploss(ticker, quantity, stock_price, RSI, BBLB, BBUB, confidence_raiting):
    userID = 1          # irrelevant for testing scenario
    position = -1        # 0 indicates close position
    total_price = round(stock_price * quantity, 2)
    values = (userID, ticker, position, quantity, stock_price, total_price, RSI, BBLB, BBUB, confidence_raiting)
    data = add_position(values)     # database handler call
    return data