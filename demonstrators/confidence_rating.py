""" Nerual Network for producing confidence rating on security signals """

# TAKE ALPHA VANTAGE MINUTE DATA, WORK OUT BB AND RSI FOR EACH MINUTE, 
# THEN PRODUCE PERCENTAGE ACCURACY IT WILL CHANGE DIRECTION

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from alpha_vantage.timeseries import TimeSeries
from ta.volatility import BollingerBands
from ta.momentum import rsi as rsi
import datetime as dt

# set API key and ticker symbol
api_key = '0T2WORTYUXU5Z4XA'
ticker_symbol = 'AAPL'
interval = '1min'
window = 14

# Fetch the data from the Alpha Vantage API
ts = TimeSeries(key=api_key, output_format='pandas')
data, _ = ts.get_intraday(symbol=ticker_symbol, interval=interval, outputsize='full')
data = data.iloc[::-1]
data = data.between_time('09:30', '16:00')

# Calculate the input features
# Create the input features and labels
bb = BollingerBands(data['4. close'])
data['bb_upper'], data['bb_middle'], data['bb_lower'] = bb.bollinger_mavg(), bb.bollinger_hband(), bb.bollinger_lband()
data['rsi'] = rsi(data['4. close'], window)

# Drop rows with missing values
data = data.dropna()

# Create the input features
required_data = data[['4. close', 'rsi', 'bb_upper', 'bb_lower']]
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
#print(X)
required_data.dropna(inplace=True)

# Calculate the labels
y = []
X = []
look_ahead = 2

for i in range(len(required_data)-look_ahead):
    # check if RSI is below 30 and price is below boiler band lower band
    if data.iloc[i]['rsi'] < 30 and data.iloc[i]['bb_lower'] > data.iloc[i]['4. close']:
        # check if the price goes up after look ahead point
        if data.iloc[i+look_ahead]['4. close'] > data.iloc[i]['4. close']:
            y.append(1) # price trend changed from downward to upward
        else:
            y.append(0) # price trend did not change or continued
        X.append(required_data.iloc[i])

    # check if RSI is above 70 and price is above boiler band upper band
    elif data.iloc[i]['rsi'] > 70 and data.iloc[i]['bb_upper'] < data.iloc[i]['4. close']:
        # check if the price goes down after look ahead point
        if data.iloc[i+look_ahead]['4. close'] < data.iloc[i]['4. close']:
            y.append(1) # price trend changed from upward to downward
        else:
            y.append(0) # price trend did not change or continued
        X.append(required_data.iloc[i])


# Train the neural network, (40,40) best performing parameters for hidden layers
nn = MLPRegressor(hidden_layer_sizes=(40,40), max_iter=1000, random_state=42)
nn.fit(X, y)

# Test predictions
new_data_point = pd.DataFrame({'4. close': [161.85], 'rsi': [71], 'bb_upper': [161.78], 'bb_lower': [160.12]})
new_data_point_low = pd.DataFrame({'4. close': [160.55], 'rsi': [28.37], 'bb_upper': [161.04], 'bb_lower': [160.60]})
new_data_point2 = pd.DataFrame({'4. close': [162.01], 'rsi': [80], 'bb_upper': [161.78], 'bb_lower': [160.12]})
new_data_point_low2 = pd.DataFrame({'4. close': [160.35], 'rsi': [23.37], 'bb_upper': [161.04], 'bb_lower': [160.60]})
new_data_point3 = pd.DataFrame({'4. close': [160.45], 'rsi': [28.45], 'bb_upper': [162.14], 'bb_lower': [160.58]})
prediction = nn.predict(new_data_point)
prediction2 = nn.predict(new_data_point_low)
prediction3 = nn.predict(new_data_point2)
prediction4 = nn.predict(new_data_point_low2)
prediction5 = nn.predict(new_data_point3)

# Print the prediction
print(f"Probability of price trend changing direction at ${new_data_point.iloc[0]['4. close']} : {prediction[0]:.2f}")

# Print the prediction low
print(f"Probability of price trend changing direction at ${new_data_point_low.iloc[0]['4. close']} : {prediction2[0]:.2f}")

print(f"Probability of price trend changing direction at ${new_data_point2.iloc[0]['4. close']} : {prediction3[0]:.2f}")

print(f"Probability of price trend changing direction at ${new_data_point_low2.iloc[0]['4. close']} : {prediction4[0]:.2f}")

print(f"Probability of price trend changing direction at ${new_data_point3.iloc[0]['4. close']} : {prediction5[0]:.2f}")
