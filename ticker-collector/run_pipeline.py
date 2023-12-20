import time

import time_series_collector
import time_series_merger
import week_marker
from tickers import crypto_tickers

start_time = time.time()
print('Collecting time series...')
time_series_collector.main(crypto_tickers, 'data/crypto-daily')
print(f'Time series are collected in {time.time() - start_time} seconds.')

start_time = time.time()
print('Merging time series...')
time_series_merger.main('data/crypto', 'out/crypto', 'daily', 20, 2190)
print(f'Time series are merged in {time.time() - start_time} seconds.')

start_time = time.time()
print('Marking weeks...')
week_marker.main('out/crypto/daily_20_2190.csv')
print(f'Weeks are marked in {time.time() - start_time} seconds.')
