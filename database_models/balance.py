""" balance functions for database interaction """

import sys
sys.path.append('../AITradingBot')    # path to parent directory
from database import *


# get all rows of the balance table
def get_all_balances():
    query = "SELECT * FROM balance;"
    data = run_all_query(query)
    return data


# Pulls specific users current balance from the database
def get_users_balance(id):
    query = "SELECT current_balance FROM balance WHERE userID = %s"
    value = id
    data = run_value_query(query, value)
    data = float("{}".format(*data[0]))     # converts tuple from sql query to float
    return data


# Pulls specific users current balance from the database
def get_users_stock_balance(id):
    query = "SELECT stock_balance FROM balance WHERE userID = %s"
    value = id
    data = run_value_query(query, value)
    data = float("{}".format(*data[0]))     # converts tuple from sql query to float
    return data


# update a users current balance to reflect result of most recent closed trade
def update_users_current_balance(id, balance):
    query = "UPDATE balance SET current_balance = %s WHERE userID = %s"
    new_balance = get_users_balance(id)
    new_balance = new_balance + balance
    values = (new_balance, id)
    data = run_alter_query(query, values)
    return data


# update a users stock balance to reflect how much is owned in stock at given time
def edit_stock_balance(id, balance):
    query = "UPDATE balance SET stock_balance = %s WHERE userID = %s"
    new_balance = get_users_stock_balance(id)
    new_balance = new_balance + balance
    values = (new_balance, id)
    data = run_alter_query(query, values)
    return data