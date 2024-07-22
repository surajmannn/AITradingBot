""" simple_trades helper functions for database interaction """

import sys
sys.path.append('../AITradingBot')    # path to parent directory
from database import *


def get_all_trades():
    query = "SELECT * FROM simple_trades;"
    data = run_all_query(query)
    return data


def get_users_trades(id):
    query = "SELECT * FROM simple_trades WHERE userID = %s"
    value = id
    data = run_value_query(query, value)
    return data


def add_position(values):
    # position = 1 if Buy and 2 if Sell
    query = "INSERT INTO simple_trades (userID, ticker, position, quantity, stock_price, total_price, RSI, BB_upper_band, BB_lower_band, confidence_rating) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
    data = run_alter_query(query, values)
    return data


def get_trade(ID):
    query = "SELECT * FROM simple_trades WHERE ID = %s"
    value = ID
    data = run_value_query(query, value)
    return data


def check_position(id, ticker):
    query = "SELECT EXISTS(SELECT * FROM simple_trades WHERE userID = %s and ticker = %s)"
    values = (id, ticker)
    data = run_multiple_value_query(query, values)
    data = int("{}".format(*data[0]))
    if(data == 1):
        # check if position type
        new_query = "SELECT position FROM simple_trades where userID = %s and ticker = %s"
        new_values = (id, ticker)
        result = run_multiple_value_query(new_query, new_values)
        return result
    return False