from dataclasses import dataclass

import seaborn as sns

from models import DataPath

initial_capital = 100
connecting_the_dots_base = "../MTGNN/old_simulations/crypto/20221121"
lstm_base = "../lstm-predictor/prediction"


# Before running analysis, adjust paths
class FXPaths:
    actual = DataPath("../ticker-collector/out/fx/daily_10_4506_marked.csv", "Actual")
    raw_prediction_paths = [
        DataPath(f"../experiment-results/fx/mtgnn-results/202312090808", "MTGNN"),
        DataPath(
            f"../experiment-results/fx/stemgnn-results/202312080300/__fx_daily_simple_returns",
            "StemGNN",
        ),
        DataPath(f"../experiment-results/fx/lstm-results/202312062100/weeks", "LSTM"),
        DataPath(f"../experiment-results/fx/fouriergnn-results/20250319", "FourierGNN"),
        DataPath(f"../experiment-results/fx/arima-results/20250321", "ARIMA"),
        DataPath(f"../experiment-results/fx/var-results/202503221825", "VAR"),
    ]


class CryptoPaths:
    actual = DataPath("../ticker-collector/out/crypto/daily_20_2189_marked.csv", "Actual")
    raw_prediction_paths = [
        DataPath(f"../experiment-results/crypto/stemgnn-results/202312090602", "StemGNN"),
        DataPath(f"../experiment-results/crypto/mtgnn-results/7", "MTGNN:7"),
        DataPath(f"../experiment-results/crypto/mtgnn-results/21", "MTGNN:21"),
        DataPath(f"../experiment-results/crypto/mtgnn-results/35", "MTGNN:35"),
        DataPath(f"../experiment-results/crypto/mtgnn-results/42", "MTGNN:42"),
        DataPath(f"../experiment-results/crypto/mtgnn-results/63", "MTGNN:63"),
        DataPath(f"../experiment-results/crypto/mtgnn-results/84", "MTGNN:84"),
        DataPath(
            f"../experiment-results/crypto/lstm-results/202302221651",
            "LSTM:202302221651",
        ),
        DataPath(
            f"../experiment-results/crypto/lstm-results/202303221739",
            "LSTM:202303221739",
        ),
        DataPath(
            f"../experiment-results/crypto/lstm-results/202309210830",
            "LSTM:202309210830",
            "log-return",
        ),
        DataPath(f"../experiment-results/crypto/fouriergnn-results/20250318", "FourierGNN"),
        DataPath(f"../experiment-results/crypto/arima-results/20250321", "ARIMA"),
        DataPath(f"../experiment-results/crypto/var-results/202503221852", "VAR"),
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
