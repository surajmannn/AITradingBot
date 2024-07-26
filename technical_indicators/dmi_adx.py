""" Functionality for DMI/ADX Technical Indicator """

import yfinance as yf

# Function returns ADX, DI+, DI- values. Takes security, trading interval, and period as parameters
def dmi_adx(ticker, interval, period_length):

    adx = 0
    DI_pos = 0
    DI_neg = 0

    return adx, DI_pos, DI_neg