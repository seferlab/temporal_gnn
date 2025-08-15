import datetime
import os

import numpy as np
import pandas as pd
from permetrics import RegressionMetric
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from statsmodels.tsa.api import VAR
from statsmodels.tsa.arima.model import ARIMA
from tqdm import tqdm, trange


# noinspection PyPep8Naming
def format_time_as_YYYYMMddHHmm(time):
    return (time.isoformat(sep="/", timespec="minutes").replace("-", "").replace(":", "").replace("/", ""))


class Constants:
    PREDICTION_TIME = format_time_as_YYYYMMddHHmm(datetime.datetime.now())


# noinspection DuplicatedCode
class Config:
    early_stopping_patience = 100
    horizon = 7

    training_ratio = 0.6
    validation_ratio = 0.2

    log_active = False
    prices_path = "../ticker-collector/out/crypto/daily_20_2189_marked.csv"
    predictions_base_path = "var_prediction-crypto"
    num_weeks_to_train = 104
    total_weeks = 104


# Config.num_weeks_to_train = 1
# Config.total_weeks = 1
# Config.num_epochs_to_run = 500
# Config.hpo_max_evals = 20
# Config.prices_path = 'lstm_predictor/tests/resources/daily_5_2189_marked.csv'
# Config.predictions_base_path = 'test_prediction'
# Config.log_active = True


def log(week, message, log_anyway=False):
    if log_anyway or Config.log_active:
        print(f"Week: {week} | {message}")


# noinspection DuplicatedCode
def read_return_ratios(
    data_path,
    week,
    num_weeks,
    training_ratio,
    fill_zero_for_the_first_horizon_samples=False,
):
    """
    :return: Log returns, the last line is the last split point in the data when week == num_weeks.
             If week == num_weeks -1, the last line is the second last split point in the data, and so on.
             If fill_zero_for_the_first_horizon_samples is False, then the first output np array will have `horizon`
             fewer rows.
    """
    assert (0 <= week <= num_weeks), f"week must be between 0 and num_weeks (inclusive): 0 <= {week} <= {num_weeks}"
    assert num_weeks >= 0, f"num_weeks must be >= 0, got {num_weeks}"

    raw_data = pd.read_csv(data_path)

    truncate_index = len(raw_data)
    stopping_point = -1
    truncation_mark = num_weeks - week

    for point in reversed(raw_data["split_point"]):
        if stopping_point == truncation_mark:
            break

        if point:
            stopping_point += 1

        truncate_index -= 1

    return _convert_to_simple_returns(
        raw_data.loc[:truncate_index].drop(["split_point"], axis=1),
        fill_zero_for_the_first_horizon_samples,
    ).to_numpy()


# noinspection DuplicatedCode
def _convert_to_simple_returns(df, fill_zero_for_the_first_horizon_samples):
    np_form = df.drop(["Date"], axis=1).to_numpy()
    simple_returns = np_form[Config.horizon:] / np_form[:-Config.horizon]
    if fill_zero_for_the_first_horizon_samples:
        simple_returns = np.vstack([np.full((Config.horizon, np_form.shape[1]), 0.01), simple_returns])
    return pd.DataFrame(simple_returns, columns=df.columns[1:])


# noinspection DuplicatedCode
class WindowedSplitDataLoader:

    def __init__(self, data_np, seq_len, horizon, training_ratio, validation_ratio):
        self._seq_length = seq_len
        self._horizon = horizon
        self._dat = data_np
        self._num_rows, self._num_cols = self._dat.shape
        self._split(
            int(training_ratio * self._num_rows),
            int((training_ratio + validation_ratio) * self._num_rows),
        )
        self._last_day = (self._dat[-seq_len:].reshape((1, seq_len, -1)).astype(np.float32))

    @property
    def training(self):
        return self._training

    @property
    def validation(self):
        return self._validation

    @property
    def training_and_validation(self):
        return self._training_and_validation

    @property
    def test(self):
        return self._test

    @property
    def all(self):
        return self._all

    @property
    def last_day(self):
        return self._last_day

    def _split(self, train_end_index, valid_end_index):
        training_set_indices = range(self._seq_length + self._horizon - 1, train_end_index)
        validation_set_indices = range(train_end_index, valid_end_index)
        test_set = range(valid_end_index, self._num_rows)
        self._training = self._batchify(training_set_indices)
        self._validation = self._batchify(validation_set_indices)
        self._test = self._batchify(test_set)
        self._training_and_validation = {
            "X": np.concatenate((self.training["X"], self.validation["X"])),
            "y": np.concatenate((self.training["y"], self.validation["y"])),
        }
        self._all = {
            "X": np.concatenate((self.training["X"], self.validation["X"], self.test["X"])),
            "y": np.concatenate((self.training["y"], self.validation["y"], self.test["y"])),
        }

    # noinspection PyPep8Naming
    def _batchify(self, idx_set):
        n = len(idx_set)
        X = np.zeros((n, self._seq_length, self._num_cols))
        y = np.zeros((n, self._num_cols))
        for i in range(n):
            end = idx_set[i] - self._horizon + 1
            start = end - self._seq_length
            X[i, :, :] = self._dat[start:end, :]
            y[i, :] = self._dat[idx_set[i], :]
        return {"X": X.astype(np.float32), "y": y.astype(np.float32)}


class Runner:

    def __init__(self, week):
        self._week = week

    @staticmethod
    def rmse(predictions, targets):
        return np.sqrt(np.mean((predictions - targets)**2))

    def evaluate(self):
        data_np = read_return_ratios(Config.prices_path, self._week, Config.total_weeks, Config.training_ratio)
        dl = WindowedSplitDataLoader(
            data_np,
            Config.horizon,
            Config.horizon,
            Config.training_ratio,
            Config.validation_ratio,
        )

        training_and_validation_size = dl.training_and_validation["X"].shape[0]
        all_size = dl.all["X"].shape[0]

        forecasts = []
        optimal_lag = None
        for i in trange(training_and_validation_size - Config.horizon, all_size - Config.horizon):
            data = dl.all["y"][:i, :]
            model = VAR(data)
            if not optimal_lag:
                lag_order_results = model.select_order(maxlags=20)
                optimal_lag = lag_order_results.aic
                print("Optimal Lag Found:", optimal_lag)
            var_fit = model.fit(optimal_lag)
            forecast = var_fit.forecast(data[-optimal_lag:], steps=Config.horizon)[-1, :]
            forecasts.append(forecast)
        forecasts = np.asarray(forecasts)
        actuals = dl.test["y"]

        def a20_index(v, v_):
            evaluator = RegressionMetric(v.reshape(v.shape[0], -1), v_.reshape(v_.shape[0], -1))
            a20 = evaluator.a20_index()
            return np.mean(a20)

        mape = mean_absolute_percentage_error(actuals, forecasts)
        mae = mean_absolute_error(actuals, forecasts)
        rmse = (mean_squared_error(actuals, forecasts))**0.5
        a20 = a20_index(actuals, forecasts)
        print(f"VAR Model Evaluation: MAPE/MAE/RMSE/A20: {mape:4.5f}/{mae:4.5f}/{rmse:4.5f}/{a20:4.5f}")

    def make_predictions(self):
        data_np = read_return_ratios(Config.prices_path, self._week, Config.total_weeks, Config.training_ratio)
        dl = WindowedSplitDataLoader(
            data_np,
            Config.horizon,
            Config.horizon,
            Config.training_ratio,
            Config.validation_ratio,
        )

        model = VAR(dl.all["y"])
        lag_order_results = model.select_order(maxlags=20)
        optimal_lag = lag_order_results.aic
        var_fit = model.fit(optimal_lag)
        forecast = var_fit.forecast(dl.all["y"][-optimal_lag:], steps=Config.horizon)[-1, :].reshape(1, -1)
        return forecast

    def save_predictions(self, predictions):
        dir_path = f"{Config.predictions_base_path}/{Constants.PREDICTION_TIME}/weeks"
        os.makedirs(dir_path, exist_ok=True)
        np.save(f"{dir_path}/{self._week + 1}", predictions)


def main_quick():
    if os.path.exists("figs"):
        os.system("rm -rf figs")
    if os.path.exists("runs"):
        os.system("rm -rf runs")
    if os.path.exists("models"):
        os.system("rm -rf models")

    for week in tqdm(range(Config.num_weeks_to_train)):
        loop = True
        while loop:
            try:
                runner = Runner(week=week)
                if week == 0:
                    runner.evaluate()
                predictions = runner.make_predictions()
                runner.save_predictions(predictions)
                loop = False
            except Exception as e:
                print(e)


if __name__ == "__main__":
    main_quick()
