from unittest import TestCase

import numpy as np
import pandas as pd

from lstm_predictor_tmp.main import Config, read_log_returns, WindowedSplitDataLoader


class Test(TestCase):

    def test_read_log_returns(self):
        path = 'resources/daily_1_200_marked.csv'
        raw_data = pd.read_csv(path)[['BTC-USD']].to_numpy()
        log_returns = read_log_returns(path, 0, 1, True)
        for i in range(179, -1, -1):
            self.assertEqual(log_returns[i + Config.horizon], np.log(raw_data[i + Config.horizon] / raw_data[i]))
        for i in range(Config.horizon):
            self.assertEqual(log_returns[i], np.full((log_returns.shape[1]), np.log(.01)))

    def test_read_log_returns_without_prepending_zeros(self):
        path = 'resources/daily_1_200_marked.csv'
        raw_data = pd.read_csv(path)[['BTC-USD']].to_numpy()
        log_returns = read_log_returns(path, 0, 1)
        for i in range(172, -1, -1):
            self.assertEqual(log_returns[i + Config.horizon],
                             np.log(raw_data[i + 2 * Config.horizon] / raw_data[i + Config.horizon]))

    # noinspection PyPep8Naming
    def test_SlidingWindowDataSplitter(self):
        log_returns = np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]).reshape(
            (-1, 1))
        dl = WindowedSplitDataLoader(log_returns, seq_len=2, horizon=3, training_ratio=.6, validation_ratio=.2)

        # shapes: (num_samples, seq_len, num_assets) as input X, (num_samples, num_assets) as target y
        expected_training_X = np.asarray([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9]]).reshape(
            (-1, 2, 1))
        expected_training_y = np.asarray([5, 6, 7, 8, 9, 10, 11, 12]).reshape((-1, 1))
        np.testing.assert_equal(dl.training['X'], expected_training_X)
        np.testing.assert_equal(dl.training['y'], expected_training_y)

        expected_validation_X = np.asarray([[9, 10], [10, 11], [11, 12], [12, 13]]).reshape((-1, 2, 1))
        expected_validation_y = np.asarray([13, 14, 15, 16]).reshape((-1, 1))
        np.testing.assert_equal(dl.validation['X'], expected_validation_X)
        np.testing.assert_equal(dl.validation['y'], expected_validation_y)

        expected_test_X = np.asarray([[13, 14], [14, 15], [15, 16], [16, 17]]).reshape((-1, 2, 1))
        expected_test_y = np.asarray([17, 18, 19, 20]).reshape((-1, 1))
        np.testing.assert_equal(dl.test['X'], expected_test_X)
        np.testing.assert_equal(dl.test['y'], expected_test_y)

        expected_last_day = np.asarray([19, 20]).reshape((1, -1, 1))
        np.testing.assert_equal(dl.last_day, expected_last_day)
