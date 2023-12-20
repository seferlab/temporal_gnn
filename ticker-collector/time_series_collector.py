import os
import sys
import time

import yfinance as yf
from tqdm import tqdm

from tickers import crypto_tickers


def main(tickers, output_path):
    os.makedirs(output_path, exist_ok=True)

    # Downloader part is taken from @derekbanas, see: https://github.com/derekbanas/Python4Finance/blob/3064e244048930631d0a6c174709f4b6f561c4d0/Download%20Every%20Stock.ipynb
    stocks_not_downloaded = []

    def save_to_csv_from_yahoo(folder, ticker):
        stock = yf.Ticker(ticker)

        # noinspection PyBroadException
        try:
            df = stock.history(period="max")
            time.sleep(2)

            if df.empty:
                stocks_not_downloaded.append(ticker)
            else:
                the_file = folder + '/' + ticker.replace(".", "_") + '.csv'
                df.to_csv(the_file)
        except:
            stocks_not_downloaded.append(ticker)
            print("Couldn't Get Data for:", ticker)

    [save_to_csv_from_yahoo(output_path, ticker) for ticker in tqdm(tickers)]
    print("Save completed.")

    if stocks_not_downloaded:
        print("Stocks not downloaded:")
        print(stocks_not_downloaded)


if __name__ == '__main__':
    main(crypto_tickers, sys.argv[1])
