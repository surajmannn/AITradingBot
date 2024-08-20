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

# Pull specific trades related to specific machine learning aglorithms
def get_mla_trades(mla):
    query = "SELECT * FROM simple_trades WHERE mla = %s"
    value = mla
    data = run_value_query(query, value)
    return data

def add_position(values):
    # position = 1 if Buy, -1 if Sell, 0 if close, and -2 if stoploss close
    # INSERT INTO trade_results
    # INSERT INTO noml_trades
    # INSERT INTO simple_trades
    query = (
        "INSERT INTO trade_results ("
        "userID, ticker, mla, position_type, quantity, security_price, total_price, profit, balance, purchase_date,"
        "BB_upper_band, BB_lower_band, RSI, ADX, DI_pos, DI_neg, volatility, confidence_probability"
        ") VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
    )
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