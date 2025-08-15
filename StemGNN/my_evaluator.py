import numpy as np
import pandas as pd


def MAPE(v, v_):
    """
    Mean absolute percentage error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :param axis: axis to do calculation.
    :return: int, MAPE averages on all elements of input.
    """
    mape = (np.abs(v_ - v) / (np.abs(v) + 1e-5)).astype(np.float64)
    return np.mean(mape)


def print_results(dataset):
    print(f"Printing results for {dataset}")

    prediction = pd.read_csv(f"output/{dataset}/test/predict.csv", header=None).to_numpy()
    target = pd.read_csv(f"output/{dataset}/test/target.csv", header=None).to_numpy()

    print(f"MAPE of prices: {MAPE(target, prediction)}")

    prediction_returns = prediction[1:] / prediction[:-1] - 1
    target_returns = target[1:] / target[:-1] - 1
    print(f"MAPE of returns: {MAPE(target_returns, prediction_returns)}")

    norm_stats = pd.read_json(f"output/{dataset}/train/norm_stat.json")
    means = norm_stats["mean"].to_numpy()
    stds = norm_stats["std"].to_numpy()

    standardized_prediction = (prediction - means) / stds
    standardized_target = (target - means) / stds
    print(f"MAPE of standardized prices: {MAPE(standardized_target, standardized_prediction)}")

    print("=========================================")


print_results("fx_daily")
print_results("fx_daily_without_try")
print_results("crypto_daily")
