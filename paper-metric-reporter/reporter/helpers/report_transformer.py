from itertools import groupby

import numpy as np
import pandas as pd

import config
from models import Data


class ReportTransformer:

    @staticmethod
    def get_predicted_prices(actual_prices, predicted_returns):
        num_assets = actual_prices.data.shape[1]
        predicted_prices = []
        for j in range(len(predicted_returns)):
            predicted_asset_prices = []
            for ind in range(num_assets):
                a = actual_prices.data[:, ind]
                r = predicted_returns[j].data[:, ind]
                p = [a[0]]
                for i in range(1, len(r)):
                    p.append(a[i - 1] * r[i])
                predicted_asset_prices.append(p)

            predicted_asset_prices = np.asarray(predicted_asset_prices).T
            predicted_prices.append(Data(predicted_asset_prices, predicted_returns[j].label))
        return predicted_prices

    @staticmethod
    def get_best_models(all_model_results):
        sorted_models = sorted(all_model_results, key=ReportTransformer.get_model_prefix)
        return [
            max(g, key=lambda model: model.final_portfolio_value())
            for k, g in groupby(sorted_models, ReportTransformer.get_model_prefix)
        ]

    @staticmethod
    def remove_model_prefixes(models):
        for model_result in models:
            model_result.label = ReportTransformer.get_model_prefix(model_result)
        return models

    @staticmethod
    def remove_model_prefix(model):
        model.label = ReportTransformer.get_model_prefix(model)
        return model

    @staticmethod
    def get_model_prefix(model):
        return model.label.split(":")[0]

    @staticmethod
    def create_portfolio_values_df(random_results, all_results, best_model_results):
        df = pd.DataFrame()
        df[random_results.label] = random_results.data
        df[all_results.label] = all_results.data

        for model_result in best_model_results:
            df[model_result.label] = model_result.data
        return df

    @staticmethod
    def get_returns(prices, normalize=True, append_ones=False):
        next_values = prices[1:]
        current_values = prices[:-1]

        if isinstance(prices, pd.DataFrame):
            next_values = next_values.reset_index(drop=True)

        returns = next_values / current_values
        if normalize:
            returns -= 1

        if append_ones:
            if isinstance(prices, pd.DataFrame):
                returns = pd.concat([returns, pd.DataFrame(np.ones(returns.shape[1])).T])
            else:
                returns = np.concatenate((returns, np.ones((1, returns.shape[1]))))
        return returns

    @staticmethod
    def get_trading_decisions(actual, prediction, hold_threshold=config.hold_threshold):
        actual = actual[:-1]
        prediction = prediction[1:]

        diff = np.abs((prediction - actual) / actual)
        decisions = np.full(actual.shape, "Hold", dtype=object)
        hold = diff <= hold_threshold
        buy = prediction > actual
        sell = prediction < actual

        decisions[~hold & buy] = "Buy"
        decisions[~hold & sell] = "Sell"
        return decisions
