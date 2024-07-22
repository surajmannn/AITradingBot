""" Main file for the project """

from signals_handler import Signal_Handler

def main():

    """ CREATE OBJECTS FROM CLASS
    # Input parameters - (ticker), (interval), (window), (rsi_buy_boundary), (rsi_short_boundary), (confidence boundary), (stoploss range percentage)
    # Signalling input requires (close_short_rsi), (close_buy_rsi), (decimal precision of security price) """

    """ FX """
    
    GBPUSD = Signal_Handler('GBPUSD=X', '1m', 14, 27, 73, 0.4, 0.25)
    GBPUSD.signalling(35.1, 64.9, 6)

    GBPEUR = Signal_Handler('GBPEUR=X', '1m', 14, 27, 73, 0.4, 0.25)
    GBPEUR.signalling(35.1, 64.9, 6)

    AUDJPY = Signal_Handler('AUDJPY=X', '1m', 14, 27, 73, 0.4, 0.25)
    AUDJPY.signalling(35.1, 64.9, 4)

    """ Stocks """
    NVDA = Signal_Handler('NVDA', '1m', 14, 27, 73, 0.4, 0.5)
    NVDA.signalling(38, 62, 2)

    AAPL = Signal_Handler('AAPL', '1m', 14, 28, 72, 0.4, 0.4)
    AAPL.signalling(38, 62, 2)

    WAYFAIR = Signal_Handler('W', '1m', 14, 27, 73, 0.4, 0.4)
    WAYFAIR.signalling(36, 64, 2)

    CPE = Signal_Handler('CPE', '1m', 14, 27, 73, 0.4, 0.4)
    CPE.signalling(36, 64, 2)

    TSLA = Signal_Handler('TSLA', '1m', 14, 27, 73, 0.4, 0.4)
    TSLA.signalling(38, 62, 2)

    OCADO = Signal_Handler('OCDO.L', '1m', 14, 27, 73, 0.4, 0.3)
    OCADO.signalling(38, 62, 2)

    LLOYDS = Signal_Handler('LLOY.L', '1m', 14, 27, 73, 0.4, 0.3)
    LLOYDS.signalling(38, 62, 2)

    """ Indices """
    KOSPI = Signal_Handler('^KS11', '1m', 14, 27, 73, 0.4, 0.3)
    KOSPI.signalling(40, 60, 2)

    FTSE250 = Signal_Handler('^FTMC', '1m', 14, 27, 73, 0.4, 0.3)
    FTSE250.signalling(38, 62, 2)

    NASDAQ = Signal_Handler('^IXIC', '1m', 14, 29, 71, 0.4, 0.4)
    NASDAQ.signalling(40, 60, 2)

    NQF = Signal_Handler('NQ=F', '1m', 14, 27, 73, 0.4, 0.3)
    NQF.signalling(40, 60, 2)


    """ Cryptos """
    ETHUSD = Signal_Handler('ETH-USD', '5m', 14, 27, 73, 0.4, 0.2)
    ETHUSD.signalling(39, 61, 2)

    BTCUSD = Signal_Handler('BTC-USD', '2m', 14, 27, 73, 0.45, 0.2)
    BTCUSD.signalling(39, 61, 2)

    DOTGBP = Signal_Handler('DOT-GBP', '1m', 14, 27, 73, 0.35, 0.2)
    DOTGBP.signalling(40,60,4)

    SOLGBP = Signal_Handler('SOL-GBP', '1m', 14, 23, 77, 0.5, 0.2)
    SOLGBP.signalling(40,60, 2)

    BNBGBP = Signal_Handler('BNB-GBP', '5m', 14, 1, 99, 0.45, 0.2)
    BNBGBP.signalling(40, 60, 2)

    AVAXGBP = Signal_Handler('XRP-GBP', '1m', 14, 20, 80, 0.5, 0.2)
    AVAXGBP.signalling(40, 60)


if __name__ == "__main__":
    main()