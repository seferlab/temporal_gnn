from unittest import TestCase

import numpy as np

from helpers import Portfolio


class TestPortfolio(TestCase):

    def test__calculate_portfolio_values(self):
        values = Portfolio._calculate_portfolio_values(asset_indices=np.asarray([[0, 1], [1, 2], [0, 2]]),
                                                       actual_prices=np.asarray([[10, 20, 30], [5, 15, 25],
                                                                                 [1, 30, 100]]),
                                                       initial_budget=100,
                                                       is_long=True,
                                                       predicted_prices=np.asarray([[1, 2, 3], [20, 40, 35],
                                                                                    [10, 50, 75]]))
        expected_predicted_portfolio_values = np.asarray(
            [100, 200, values.actual[1] / 2 * 50 / 15 + values.actual[1] / 2 * 75 / 25])
        np.testing.assert_array_equal(values.prediction, expected_predicted_portfolio_values)

    def test__get_asset_selection_accuracy(self):
        y_true = np.asarray([[0, 1, 3], [4, 2, 5], [3, 6, 6], [6, 6, 6], [6, 6, 2]])
        y_pred = np.asarray([[3, 1, 6], [2, 4, 5], [4, 2, 1], [6, 6, 2], [6, 6, 6]])
        accuracy = Portfolio._get_asset_selection_accuracy(y_true, y_pred)

        print(accuracy)
        self.assertAlmostEqual(accuracy, (2 + 3 + 0 + 2 + 2) / 5 / 3, delta=0.0001)
