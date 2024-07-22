import yfinance as yf
import yahoo_fin.stock_info as si
import time

# Define the stock symbol and the time interval for which you want to retrieve data
symbol = 'GBPUSD=X'
interval = '1m'

def boiler_bands(ticker, interval):

    symbol = ticker
    interval = interval

    while True:
        # Use yfinance to retrieve the stock data and load it into a pandas DataFrame
        df = yf.download(symbol, interval=interval, period='7d', progress=False)

        # Calculate the 20-day moving average and standard deviation using the rolling() method
        rolling_mean = df['Close'].rolling(window=20).mean()
        rolling_std = df['Close'].rolling(window=20).std()

        # Calculate the upper and lower bands using the extracted data
        upper_band = rolling_mean + (2 * rolling_std)
        lower_band = rolling_mean - (2 * rolling_std)

        # Get the current market price using yahoo_fin
        current_price = si.get_live_price(symbol).round(8)

        # Check if the current price crosses above the upper band or below the lower band
        # generate signal if either hits
        if current_price > upper_band.iloc[-1]:
            print("Time: ", upper_band.index[-1])
            print(f"SELL signal generated at {current_price}")
            print("Upper Band: ", upper_band.iloc[-1].round(5))
            print("Lower Band: ", lower_band.iloc[-1].round(5))
            print("\n")
        elif current_price < lower_band.iloc[-1]:
            print("Time: ", upper_band.index[-1])
            print(f"BUY signal generated at {current_price}")
            print("Upper Band: ", upper_band.iloc[-1].round(5))
            print("Lower Band: ", lower_band.iloc[-1].round(5))
            print("\n")
        else:
            print("Time: ", upper_band.index[-1])
            print(f"No signal generated at {current_price}")
            print("Upper Band: ", upper_band.iloc[-1].round(5))
            print("Lower Band: ", lower_band.iloc[-1].round(5))
            print("\n")

        # Wait for 1 second before updating the signals
        time.sleep(2)

boiler_bands('GBPUSD=X', '1m')
#boiler_bands('AAPL', '1m')