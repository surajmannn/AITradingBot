""" Functionality for RSI Technical Indicator using ta library """

import ta

# Function returns the appended RSI values to the dataset
def rsi(security_data):

    # Get RSI values from RSI function in ta library
    security_data['RSI'] = ta.momentum.rsi(security_data['Close'])

    # Round RSI values to 2 decimal places
    security_data['RSI'] = security_data['RSI'].round(2)

    return security_data