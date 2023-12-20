import numpy as np


class Stats:

    @staticmethod
    def mape(actual, prediction):
        return np.mean(np.abs((actual - prediction) / actual)) * 100

    @staticmethod
    def std_mean(mat):
        mean = np.mean(mat, axis=0)
        std = np.std(mat, axis=0)

        std_adj = np.where(std < 1e-6, 1, std)

        standardized_mat = (mat - mean) / std_adj
        row_means = np.mean(standardized_mat, axis=1)
        return row_means

    @staticmethod
    def mbd(actual, prediction):
        percentage_error = 100 * (prediction - actual) / actual
        return np.mean(percentage_error)

    @staticmethod
    def cfe(actual, prediction):
        forecast_error = prediction - actual
        cfe_per_column = np.sum(forecast_error, axis=0)
        total_cfe = np.sum(cfe_per_column)
        return total_cfe
