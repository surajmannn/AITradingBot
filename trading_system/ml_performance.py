""" Execution logic to enter machine learning algorithm metrics into the database """

from database_models.ml_metrics import *

# This function sends a buy execution into the database
def add_metrics(ticker, ml_model, training_start_date, training_end_date, accuracy, roc_auc, sig_accuracy, sig_roc_auc):
    userID = 1          # irrelevant for test scenario
    values = (userID, ticker, ml_model, training_start_date, training_end_date, accuracy, roc_auc, sig_accuracy, sig_roc_auc)
    data = add_performance_metrics(values)     # database handler call
    return data