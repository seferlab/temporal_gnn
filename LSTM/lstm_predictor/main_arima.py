import datetime
import os

import numpy as np
import pandas as pd
from hyperopt import fmin, hp, space_eval, tpe
from permetrics import RegressionMetric
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
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
    horizon = 5

    hpo_max_evals = 30
    hpo_space = {
        "seq_len": hp.choice("seq_len", [horizon]),
        "p": hp.choice("p", range(1, 6)),
        "d": hp.choice("d", range(0, 3)),
        "q": hp.choice("q", range(1, 6)),
    }

    training_ratio = 0.6
    validation_ratio = 0.2

    log_active = False
    prices_path = "../ticker-collector/out/fx/daily_10_4506_marked.csv"
    predictions_base_path = "arima_prediction-crypto"
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
        self.space = Config.hpo_space

    def get_best_hparams(self):
        """
        Trains the model multiple times, finds the best hparams and returns.
        """
        best_hparams_for_assets = []
        num_assets = len(pd.read_csv(Config.prices_path).columns) - 2
        for asset_index in tqdm(range(num_assets), desc="get_best_hparams"):
            best = fmin(
                self.asset_objective(asset_index),
                self.space,
                algo=tpe.suggest,
                max_evals=Config.hpo_max_evals,
            )
            best_hparams = space_eval(self.space, best)
            print(f"Best hparams for asset: {asset_index}:", best_hparams)
            best_hparams_for_assets.append(best_hparams)
        return best_hparams_for_assets

    def asset_objective(self, asset_index):

        def objective(hparams):
            """
            :return: Validation loss
            """
            log(self._week, f"Trying hyperparams: {hparams}")

            data_np = read_return_ratios(
                Config.prices_path,
                self._week,
                Config.total_weeks,
                Config.training_ratio,
            )

            dl = WindowedSplitDataLoader(
                data_np,
                hparams["seq_len"],
                Config.horizon,
                Config.training_ratio,
                Config.validation_ratio,
            )

            training_validation_X = dl.training_and_validation["X"][:, -1, :]
            training_validation_y = dl.training_and_validation["y"]
            prediction_y = np.empty_like(training_validation_y)
            prediction_y[:, :] = 0

            i = dl.training["X"].shape[0]
            model = ARIMA(
                training_validation_X[:i, asset_index],
                order=(hparams["p"], hparams["d"], hparams["q"]),
            )
            model_fit = model.fit()
            if not model_fit.mle_retvals["converged"]:
                print("not converged")
                return 1e7
            else:
                print("Converged")
            forecasts = model_fit.forecast(steps=Config.horizon * 2)
            for k in range(forecasts.shape[0]):
                prediction_y[i + k, asset_index] = forecasts[k]

            pred = prediction_y[
                dl.training["X"].shape[0]:dl.training["X"].shape[0] + Config.horizon * 2,
                asset_index,
            ]
            expected = training_validation_y[
                dl.training["X"].shape[0]:dl.training["X"].shape[0] + Config.horizon * 2,
                asset_index,
            ]

            print(self.rmse(pred, expected))

            return model_fit.aic

        return objective

    def individual_asset_objective(self, asset_index, row_index):

        def objective(hparams):
            """
            :return: Validation loss
            """
            log(self._week, f"Trying hyperparams: {hparams}")

            data_np = read_return_ratios(
                Config.prices_path,
                self._week,
                Config.total_weeks,
                Config.training_ratio,
            )

            dl = WindowedSplitDataLoader(
                data_np,
                hparams["seq_len"],
                Config.horizon,
                Config.training_ratio,
                Config.validation_ratio,
            )

            model = ARIMA(
                dl.all["y"][:row_index, asset_index],
                order=(hparams["p"], hparams["d"], hparams["q"]),
            )
            model_fit = model.fit()
            if not model_fit.mle_retvals["converged"]:
                print("not converged")
                return 1e7
            else:
                print("Converged")

            return model_fit.aic

        return objective

    @staticmethod
    def rmse(predictions, targets):
        return np.sqrt(np.mean((predictions - targets)**2))

    def evaluate_the_best_model(self):
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
        best_hparams_for_assets = [None] * dl.training_and_validation["y"].shape[1]
        forecasts = []
        for asset_index in trange(dl.training_and_validation["y"].shape[1]):
            asset_forecast = []
            for i in trange(
                    training_and_validation_size - Config.horizon,
                    all_size - Config.horizon,
                    desc=f"Evaluating for asset: {asset_index}",
            ):
                loop = True
                while loop:
                    try:
                        if not best_hparams_for_assets[asset_index]:
                            best = fmin(
                                self.individual_asset_objective(asset_index, i),
                                self.space,
                                algo=tpe.suggest,
                                max_evals=Config.hpo_max_evals,
                            )
                            best_hparams = space_eval(self.space, best)
                            print(f"Best hparams for asset: {asset_index}:", best_hparams)
                            best_hparams_for_assets[asset_index] = best_hparams
                        hparams = best_hparams_for_assets[asset_index]

                        model = ARIMA(
                            dl.all["y"][:i, asset_index],
                            order=(hparams["p"], hparams["d"], hparams["q"]),
                        ).fit()
                        forecast = model.forecast(Config.horizon)[-1]
                        asset_forecast.append(forecast)
                        loop = False
                    except Exception as e:
                        best_hparams_for_assets[asset_index] = None
                        print(
                            f"An error occurred during fitting an ARIMA model for asset: {asset_index} at sample: {i}. Error: {e}"
                        )

            forecasts.append(asset_forecast)
        forecasts = np.asarray(forecasts).T
        actuals = dl.test["y"]

        def a20_index(v, v_):
            evaluator = RegressionMetric(v.reshape(v.shape[0], -1), v_.reshape(v_.shape[0], -1))
            a20 = evaluator.a20_index()
            return np.mean(a20)

        mape = mean_absolute_percentage_error(actuals, forecasts)
        mae = mean_absolute_error(actuals, forecasts)
        rmse = (mean_squared_error(actuals, forecasts))**0.5
        a20 = a20_index(actuals, forecasts)
        print(f"Best Model Evaluation: MAPE/MAE/RMSE/A20: {mape:4.5f}/{mae:4.5f}/{rmse:4.5f}/{a20:4.5f}")

    def make_predictions(self, best_hparams_for_assets):
        data_np = read_return_ratios(Config.prices_path, self._week, Config.total_weeks, Config.training_ratio)
        dl = WindowedSplitDataLoader(
            data_np,
            Config.horizon,
            Config.horizon,
            Config.training_ratio,
            Config.validation_ratio,
        )

        forecasts = []
        for j in range(len(best_hparams_for_assets)):
            hparams = best_hparams_for_assets[j]
            model = ARIMA(dl.all["y"][:, j], order=(hparams["p"], hparams["d"], hparams["q"])).fit()
            forecast = model.forecast(Config.horizon)[-1]
            forecasts.append(forecast)
        forecasts = np.asarray(forecasts).reshape(1, -1)
        return forecasts

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

    best_hparams_for_assets = None
    for week in tqdm(range(Config.num_weeks_to_train)):
        loop = True
        while loop:
            runner = Runner(week=week)
            try:
                # if not best_hparams_for_assets:
                #     best_hparams_for_assets = runner.get_best_hparams()
                if week == 0:
                    runner.evaluate_the_best_model()
                # predictions = runner.make_predictions(best_hparams_for_assets)
                # runner.save_predictions(predictions)
                loop = False
            except Exception as e:
                print("An error occurred during evaluation", e)
                best_hparams_for_assets = runner.get_best_hparams()


if __name__ == "__main__":
    main_quick()
