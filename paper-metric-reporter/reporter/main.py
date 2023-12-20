import random
import sys

import pandas as pd

import config
from helpers import PortfolioGroup, Reader, Reporter, ReportTransformer
from models import *

random.seed(0)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)


def main(dataset):
    if dataset == 'fx':
        paths = config.FXPaths
    elif dataset == 'crypto':
        paths = config.CryptoPaths
    else:
        sys.exit("Error: Unknown dataset. Please use 'crypto' or 'fx' as the dataset argument.")

    actual_prices = Reader.get_actual_prices(paths.actual.path, config.num_weeks)
    num_assets = actual_prices.data.shape[1]
    predicted_returns = [
        Data(Reader.get_predicted_returns(path.path, config.num_weeks, num_assets, path.data_type), path.label)
        for path in paths.raw_prediction_paths
    ]
    predicted_prices = ReportTransformer.get_predicted_prices(actual_prices, predicted_returns)

    long_short_combinations = [(long, short) for long in range(num_assets) for short in range(num_assets)
                               if not (long == 0 and short == 0)]

    if config.quick:
        config.random_trials = 3
        long_short_combinations = long_short_combinations[:3]

    Reporter.plot_correlation_matrix_of_actual_prices(actual_prices)
    Reporter.plot_standardized_prices(actual_prices, predicted_prices)

    random_results = PortfolioGroup.calculate_random_portfolio_for_predictions(config.random_trials, actual_prices,
                                                                               (5, 0))
    all_results = PortfolioGroup.calculate_all_portfolio_for_predictions(actual_prices)
    all_model_results = PortfolioGroup.calculate_best_portfolio_for_all_predictions(actual_prices, predicted_returns,
                                                                                    long_short_combinations)

    best_model_results = ReportTransformer.get_best_models(all_model_results)
    best_model_names = [m.label for m in best_model_results]

    best_model_results = ReportTransformer.remove_model_prefixes(best_model_results)
    portfolio_values_df = ReportTransformer.create_portfolio_values_df(random_results, all_results, best_model_results)

    Reporter.plot_portfolio_values_of_random_all_and_deep_learning_models(portfolio_values_df)
    Reporter.print_resulting_portfolio_values_for_each_model(portfolio_values_df)
    Reporter.print_portfolio_metrics(portfolio_values_df, bull_bear_split_needed=dataset == 'crypto')
    Reporter.print_portfolio_asset_selection_accuracies(best_model_results + [random_results])

    returns_df = ReportTransformer.get_returns(portfolio_values_df, normalize=True, append_ones=False)
    Reporter.print_statistical_significance_metrics(portfolio_values_df, 'capital')
    Reporter.print_statistical_significance_metrics(returns_df, 'return')
    Reporter.plot_returns(returns_df)
    Reporter.plot_correlation_matrix_of_returns(returns_df)
    Reporter.plot_pairwise_returns(returns_df)

    Reporter.print_test_mapes()
    Reporter.print_stats_between_all_predicted_and_actual_prices(actual_prices, predicted_prices)
    Reporter.print_stats_between_all_predicted_and_actual_returns(actual_prices, predicted_returns)
    Reporter.print_mapes_between_actual_and_predicted_model_portfolios(best_model_results)

    Reporter.print_direction_based_metrics(actual_prices, predicted_prices, best_model_names)
    Reporter.plot_predicted_and_actual_portfolio_values(best_model_results)


if __name__ == '__main__':
    main(sys.argv[1])
