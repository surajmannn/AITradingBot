""" trades functions for database interaction """

import sys
sys.path.append('../AITradingBot')    # path to parent directory
from database import *
from stock_data import get_live_stock_price


def get_all_trades():
    query = "SELECT * FROM trades;"
    data = run_all_query(query)
    return data


def get_users_trades(id):
    query = "SELECT * FROM trades WHERE userID = %s"
    value = id
    data = run_value_query(query, value)
    return data


def add_position(values):
    query = "INSERT INTO trades (userID, ticker, position, quantity, entry_stock_price, entry_total_price, status) VALUES (%s, %s, %s, %s, %s, %s, %s)"
    data = run_alter_query(query, values)
    return data


def get_trade(ID):
    query = "SELECT * FROM trades WHERE ID = %s"
    value = ID
    data = run_value_query(query, value)
    return data


def check_position(id, ticker, position_type):
    query = "SELECT EXISTS(SELECT * FROM trades WHERE userID = %s and ticker = %s and position = %s)"
    values = (id, ticker, position_type)
    data = run_multiple_value_query(query, values)
    data = int("{}".format(*data[0]))
    if(data == 1):
        # check if position is open
        new_query = "SELECT status, ID FROM trades where userID = %s and ticker = %s and position = %s"
        new_values = (id, ticker, position_type)
        result = run_multiple_value_query(new_query, new_values)
        status = result[0][0]
        ID = result[0][1]
        if(status ==  1):
            close_position(ID, ticker)
            return True
    return False


def close_position(trade_id, ticker):
    quantity = int("{}".format((get_trade(trade_id)[0][5])))
    exit_price = round(get_live_stock_price(ticker), 2)
    exit_total_price = round(quantity * exit_price, 2)
    query = "UPDATE trades SET status = %s, close_date = %s, exit_stock_price = %s, exit_total_stock_price = %s WHERE ID = %s"
    values = (2, "CURRENT_TIMESTAMP", exit_price, exit_total_price, trade_id)
    data = run_alter_query(query, values)
    return data


#print(get_all_trades())
#print(get_users_trades(1))
#data = check_position(1, 'MSFT', 1)
#print(data)

"""values = (1, 'MSFT', 1, 10, 200, 2000, 1)
add_position(values)"""