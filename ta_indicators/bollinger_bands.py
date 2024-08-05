""" Functionality for Bollinger Bands Technical Indicator using ta library"""

import ta

# Function returns appended bollinger band values to the dataset
def bollinger_bands(security_data):

    # Use Bollinger Band technical indicator from ta library to get upper and lower bands
    bb = ta.volatility.BollingerBands(security_data['Close'])
    security_data['BB_upper'], security_data['BB_middle'], security_data['BB_lower'] = bb.bollinger_hband(), bb.bollinger_mavg(), bb.bollinger_lband()

    return security_data