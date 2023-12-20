from dataclasses import dataclass

import seaborn as sns

from models import DataPath

initial_capital = 100
connecting_the_dots_base = '../MTGNN/old_simulations/crypto/20221121'
lstm_base = '../lstm-predictor/prediction'


# Before running analysis, adjust paths
class FXPaths:
    actual = DataPath('../ticker-collector/out/fx/daily_10_4506_marked.csv', 'Actual')
    raw_prediction_paths = [
        DataPath(f'../experiment-results/fx/mtgnn-results/202312090808', 'MTGNN'),
        DataPath(f'../experiment-results/fx/stemgnn-results/202312080300/__fx_daily_simple_returns', 'StemGNN'),
        DataPath(f'../experiment-results/fx/lstm-results/202312062100/weeks', 'LSTM')
    ]


class CryptoPaths:
    actual = DataPath('../ticker-collector/out/crypto/daily_20_2189_marked.csv', 'Actual')
    raw_prediction_paths = [
        DataPath(f'../experiment-results/crypto/stemgnn-results/__crypto_daily_simple_returns', 'StemGNN'),
        DataPath(f'{connecting_the_dots_base}/prediction/7/crypto/daily_20_2189_marked/weeks', 'MTGNN:7'),
        DataPath(f'{connecting_the_dots_base}/prediction/21/crypto/daily_20_2189_marked/weeks', 'MTGNN:21'),
        DataPath(f'../experiment-results/crypto/mtgnn-results', 'MTGNN:35'),
        DataPath(f'{connecting_the_dots_base}/prediction/42/crypto/daily_20_2189_marked/weeks', 'MTGNN:42'),
        DataPath(f'{connecting_the_dots_base}/prediction/63/crypto/daily_20_2189_marked/weeks', 'MTGNN:63'),
        DataPath(f'{connecting_the_dots_base}/prediction/84/crypto/daily_20_2189_marked/weeks', 'MTGNN:84'),
        DataPath(f'{lstm_base}/202302221651/weeks', 'LSTM:202302221651'),
        DataPath(f'{lstm_base}/202303221739/weeks', 'LSTM:202303221739'),
        DataPath(f'{lstm_base}/202309210830/weeks', 'LSTM:202309210830', 'log-return')
    ]


num_weeks = 104
random_trials = 1000
hold_threshold = 0.01
transaction_cost = 0.005
quick = False
sns.set_theme(style="darkgrid")


@dataclass
class OutputFlags:
    plot_standardized_prices: bool = True
    plot_pairwise_returns: bool = False
    plot_predicted_and_actual_portfolio_values: bool = False
