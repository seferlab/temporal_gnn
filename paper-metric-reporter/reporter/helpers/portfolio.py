import random
from collections import Counter

import numpy as np

import config
from models import PortfolioResult, PortfolioValues
from .report_transformer import ReportTransformer


class Portfolio:
    _initial_capital = config.initial_capital

    @classmethod
    def get_equally_weighted_portfolio_results(cls, actual_prices, predicted_returns, long_assets, short_assets):

        def get_results(is_long):
            assets = long_assets if is_long else short_assets
            return cls._get_results(actual_prices.data, assets, predicted_returns.data, is_long,
                                    cls._calculate_initial_budget(long_assets, short_assets, is_long))

        (long_results, long_selection_accuracy) = get_results(True)
        (short_results, short_selection_accuracy) = get_results(False)

        return PortfolioResult(data=long_results.actual + short_results.actual,
                               label=predicted_returns.label,
                               long_short_combination=(long_assets, short_assets),
                               predicted_result_data=long_results.prediction + short_results.prediction,
                               asset_selection_accuracy=np.average([long_selection_accuracy, short_selection_accuracy],
                                                                   weights=[long_assets, short_assets]))

    @classmethod
    def get_all_results(cls, actual_prices):
        return cls._get_all_results(actual_prices.data, cls._initial_capital)

    @classmethod
    def get_random_results(cls, actual_prices, long_assets, short_assets):

        def get_results(is_long):
            assets = long_assets if is_long else short_assets
            return cls._get_random_results(actual_prices.data,
                                           cls._calculate_initial_budget(long_assets, short_assets, is_long), is_long,
                                           assets)

        long_results, long_selection_accuracy = get_results(True)
        short_results, short_selection_accuracy = get_results(False)

        asset_selection_accuracy = np.average([long_selection_accuracy, short_selection_accuracy],
                                              weights=[long_assets, short_assets])
        return long_results + short_results, asset_selection_accuracy

    @classmethod
    def get_initial_capital(cls):
        return cls._initial_capital

    @classmethod
    def _get_results(cls, actual_prices, num_assets, predicted_returns, is_long, initial_budget):
        portfolio_asset_indices = np.apply_along_axis(Portfolio._find_assets_to_buy(num_assets, is_long),
                                                      axis=1,
                                                      arr=predicted_returns)

        actual_returns = ReportTransformer.get_returns(actual_prices, normalize=False, append_ones=True)
        best_asset_indices = np.apply_along_axis(Portfolio._find_assets_to_buy(num_assets, is_long),
                                                 axis=1,
                                                 arr=actual_returns)

        asset_selection_accuracy = cls._get_asset_selection_accuracy(best_asset_indices, portfolio_asset_indices)
        return cls._calculate_portfolio_values(portfolio_asset_indices, actual_prices, initial_budget, is_long,
                                               predicted_returns), asset_selection_accuracy

    @classmethod
    def _get_all_results(cls, actual_prices, initial_capital):
        portfolio_asset_indices = np.apply_along_axis(cls._all_assets(), axis=1, arr=actual_prices)
        return cls._calculate_portfolio_values(portfolio_asset_indices,
                                               actual_prices,
                                               initial_capital,
                                               True,
                                               no_transaction=True).actual

    @classmethod
    def _get_random_results(cls, actual_prices, initial_budget, is_long, num_assets):
        portfolio_asset_indices = np.apply_along_axis(cls._random_assets(num_assets), axis=1, arr=actual_prices)
        actual_returns = ReportTransformer.get_returns(actual_prices, normalize=False, append_ones=True)
        best_asset_indices = np.apply_along_axis(Portfolio._find_assets_to_buy(num_assets, is_long),
                                                 axis=1,
                                                 arr=actual_returns)
        asset_selection_accuracy = cls._get_asset_selection_accuracy(best_asset_indices, portfolio_asset_indices)
        return cls._calculate_portfolio_values(portfolio_asset_indices, actual_prices, initial_budget,
                                               is_long).actual, asset_selection_accuracy

    @classmethod
    def _calculate_initial_budget(cls, long_assets, short_assets, is_long):
        if is_long:
            return cls._initial_capital * long_assets / (long_assets + short_assets)
        else:
            return cls._initial_capital * short_assets / (long_assets + short_assets)

    @staticmethod
    def _find_assets_to_buy(n_assets, is_long=True):

        def find(row):
            n = min(n_assets, row.shape[0])
            if not is_long:
                row = -row
            asset_indices = np.argpartition(row, -n)[-n:]
            for i, asset_index in enumerate(asset_indices):
                if (row[asset_index] <= 1 and is_long) or (row[asset_index] >= 1 and not is_long):
                    asset_indices[i] = row.shape[0]
            return asset_indices

        return find

    @staticmethod
    def _random_assets(n_assets):

        def find(row):
            n = min(n_assets, row.shape[0])
            return np.asarray(random.sample(range(row.shape[0] + 1), n))

        return find

    @staticmethod
    def _all_assets():

        def find(row):
            return np.asarray(range(row.shape[0] + 1))

        return find

    @staticmethod
    def _calculate_portfolio_values(asset_indices,
                                    actual_prices,
                                    initial_budget,
                                    is_long,
                                    predicted_prices=None,
                                    no_transaction=False):
        portfolio_values = [initial_budget]
        predicted_portfolio_values = None if predicted_prices is None else [initial_budget]
        if asset_indices.shape[1] == 0:
            return PortfolioValues(portfolio_values, portfolio_values)

        def add_constant_prices(prices):
            return np.hstack((prices, np.ones((prices.shape[0], 1))))

        actual_prices_with_constant = add_constant_prices(actual_prices)
        predicted_prices_with_constant = add_constant_prices(predicted_prices) if predicted_prices is not None else None

        # noinspection PyShadowingNames
        def calculate_actual_and_predicted_value(budget_per_asset, i):

            def calculate_change_rate(asset_index, prices):
                current_prices = prices[i, asset_index]
                previous_prices = actual_prices_with_constant[i - 1, asset_index]

                if is_long:
                    return current_prices / previous_prices
                else:
                    return 1 - (current_prices - previous_prices) / previous_prices

            total_change = sum(
                calculate_change_rate(asset_index, actual_prices_with_constant)
                for asset_index in asset_indices[i - 1, :])
            predicted_total_change = None if predicted_prices_with_constant is None else sum(
                calculate_change_rate(asset_index, predicted_prices_with_constant)
                for asset_index in asset_indices[i - 1, :])
            return budget_per_asset * total_change, None if predicted_prices is None else budget_per_asset * predicted_total_change

        transaction_cost = 0 if no_transaction else config.transaction_cost
        for i in range(1, asset_indices.shape[0]):
            budget_per_asset = (portfolio_values[i - 1] / asset_indices.shape[1]) * (1 - transaction_cost)
            actual_value, predicted_value = calculate_actual_and_predicted_value(budget_per_asset, i)
            portfolio_values.append(actual_value)
            predicted_portfolio_values.append(predicted_value) if predicted_prices is not None else ()

        return PortfolioValues(np.asarray(portfolio_values),
                               None if predicted_prices is None else np.asarray(predicted_portfolio_values))

    @staticmethod
    def _get_asset_selection_accuracy(y_true, y_pred):
        num_assets = y_true.shape[1]

        def count_common_assets(row):
            left_assets = Counter(row[:num_assets])
            right_assets = Counter(row[num_assets:])
            return sum(min(left_assets[asset], right_assets[asset]) for asset in left_assets)

        y_true_and_pred = np.hstack((y_true, y_pred))
        common_asset_counts = np.apply_along_axis(count_common_assets, axis=1, arr=y_true_and_pred)
        return np.mean(common_asset_counts) / num_assets
