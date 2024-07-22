""" Create stock ticker object (for selected stock) """

# Use this to run all functions from other files for created stock object
class Stock:

    def __init__(self, ticker):
        self._ticker = ticker
        

    def get_ticker(self):
        return self._ticker