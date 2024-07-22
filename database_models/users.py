""" user functions for database interaction """

import sys
sys.path.append('../AITradingBot')    # path to parent directory
from database import *

def get_all_users():
    query = "SELECT * FROM users;"
    data = run_all_query(query)
    return data

def get_user(id):
    query = "select * from users where id = %s"
    value = id
    data = run_value_query(query, value)
    return data

print(get_all_users())
print(get_user(1))