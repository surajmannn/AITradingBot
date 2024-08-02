""" Functionality for DMI/ADX Technical Indicator using ta library"""

import ta

# Function returns ADX, DI+, DI- values appended to the dataset
def dmi_adx(security_data):

    # Calculate DMI/ADX using the ta library and append to dataset
    security_data['ADX'] = ta.trend.adx(security_data['High'], security_data['Low'], security_data['Close'])
    security_data['DI+'] = ta.trend.adx_pos(security_data['High'], security_data['Low'], security_data['Close'])
    security_data['DI-'] = ta.trend.adx_neg(security_data['High'], security_data['Low'], security_data['Close'])

    return security_data
