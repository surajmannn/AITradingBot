""" ml_metrics helper functions for database interaction """

import sys
sys.path.append('../AITradingBot')    # path to parent directory
from database import *

def get_all_metrics():
    query = "SELECT * FROM ml_metrics;"
    data = run_all_query(query)
    return data

def get_model_data(id):
    query = "SELECT * FROM ml_metrics WHERE ml_model = %s"
    value = id
    data = run_value_query(query, value)
    return data

def add_performance_metrics(values):
    query = (
        "INSERT INTO ml_metrics ("
        "userID, ticker, ml_model, training_start_date, training_end_date, accuracy, ROC_AUC"
        ") VALUES (%s, %s, %s, %s, %s, %s, %s)"
    )
    data = run_alter_query(query, values)
    return data