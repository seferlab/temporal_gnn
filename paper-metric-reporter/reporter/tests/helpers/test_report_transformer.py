from unittest import TestCase

import numpy as np
import pandas as pd

from helpers import ReportTransformer


class TestReportTransformer(TestCase):

    def test_make_trading_decisions(self):
        actual = np.asarray([[1, 2], [3, 4], [15, 6], [7, 5.9999]])
        prediction = np.asarray([[7, 3], [4, 7], [2, 7], [1, 12]])

        expected_predicted_decisions = np.asarray([['Buy', 'Buy'], ['Sell', 'Buy'], ['Sell', 'Buy']])
        np.testing.assert_array_equal(ReportTransformer.get_trading_decisions(actual, prediction, 0.01),
                                      expected_predicted_decisions)

        expected_ground_truth_decisions = np.asarray([['Buy', 'Buy'], ['Buy', 'Buy'], ['Sell', 'Hold']])
        np.testing.assert_array_equal(ReportTransformer.get_trading_decisions(actual, actual, 0.01),
                                      expected_ground_truth_decisions)

    def test_get_returns(self):
        actual = np.asarray([[1, 2], [3, 4], [15, 6], [7.5, 8]])
        expected = np.asarray([[3, 2], [5, 1.5], [.5, 8 / 6]])

        # It should work with pandas DFs
        actual_df = pd.DataFrame(actual)
        returns = ReportTransformer.get_returns(actual_df, normalize=False)
        np.testing.assert_array_equal(returns.to_numpy(), expected)

        # It should work with numpy arrays
        returns = ReportTransformer.get_returns(actual, normalize=False)
        np.testing.assert_array_equal(returns, expected)
