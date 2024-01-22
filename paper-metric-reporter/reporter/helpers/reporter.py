import itertools

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import config
from helpers import Portfolio, ReportTransformer
from models import Data
from .stats import Stats


class Reporter:

    @staticmethod
    def plot_standardized_prices(actual_prices, predicted_prices):
        if config.OutputFlags.plot_standardized_prices:
            standardized_actual_prices = Data(Stats.std_mean(actual_prices.data), actual_prices.label)
            standardized_predicted_prices = [Data(Stats.std_mean(pp.data), pp.label) for pp in predicted_prices]

            df = pd.DataFrame(standardized_actual_prices.data, columns=[standardized_actual_prices.label])
            for spp in standardized_predicted_prices:
                df[spp.label] = spp.data

            fig, ax = plt.subplots(figsize=(15, 10))
            df.plot(ax=ax)
            ax.set_xlabel('Weeks')
            ax.set_ylabel('Standardized Prices')
            ax.set_title('Standardized Prices of Actual and Predicted Prices')
            plt.show()

    @classmethod
    def print_stats_between_all_predicted_and_actual_prices(cls, actual_prices, predicted_prices):
        df = pd.DataFrame()
        df['Model'] = [pp.label for pp in predicted_prices]
        df['MAPEs Between Weekly Predicted and Actual Prices'] = [
            Stats.mape(actual_prices.data, pp.data) for pp in predicted_prices
        ]
        df['Mean Bias Deviation (MBD)'] = [Stats.mbd(actual_prices.data, pp.data) for pp in predicted_prices]
        df['Cumulative Forecast Error (CFE)'] = [Stats.cfe(actual_prices.data, pp.data) for pp in predicted_prices]
        cls.display(df.set_index('Model'))
        print()

    @classmethod
    def print_stats_between_all_predicted_and_actual_returns(cls, actual_prices, predicted_returns):
        actual_returns = actual_prices.data[1:] / actual_prices.data[:-1]
        df = pd.DataFrame()
        df['Model'] = [pp.label for pp in predicted_returns]
        df['MAPEs Between Weekly Predicted and Actual Returns'] = [
            Stats.mape(actual_returns, pp.data[:-1]) for pp in predicted_returns
        ]
        df['Mean Bias Deviation (MBD)'] = [Stats.mbd(actual_returns, pp.data[:-1]) for pp in predicted_returns]
        df['Cumulative Forecast Error (CFE)'] = [Stats.cfe(actual_returns, pp.data[:-1]) for pp in predicted_returns]
        cls.display(df.set_index('Model'))
        print()

    @staticmethod
    def plot_portfolio_values_of_random_all_and_deep_learning_models(portfolio_values_df):
        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        portfolio_values_df.plot(ax=ax)
        ax.set_xlabel('Weeks')
        ax.set_ylabel('Capital ($)')
        ax.set_title('Portfolio Values of Random, All and Deep Learning Models')
        plt.show()

    @classmethod
    def print_resulting_portfolio_values_for_each_model(cls, portfolio_values_df):
        df = portfolio_values_df.tail(1).T
        df.columns = ['Final Capital ($)']
        df['Initial Capital ($)'] = float(Portfolio.get_initial_capital())
        df = df[['Initial Capital ($)', 'Final Capital ($)']]
        cls.display(df)
        print()

    @staticmethod
    def print_statistical_significance_metrics(values_df, kind):
        print(f'Statistical significance analysis for weekly {kind}:')

        results = []

        for comp in itertools.combinations(values_df.columns, 2):
            t_stat, p_val = stats.ttest_rel(values_df[comp[0]], values_df[comp[1]])
            result = p_val < 0.05
            results.append({
                'Comparison': f'{comp[0]} vs {comp[1]}',
                'T-Statistic': t_stat,
                'P-Value': p_val,
                'P-Value < 0.05': result
            })

        Reporter.display(pd.DataFrame(results).set_index('Comparison'))
        print()

    @classmethod
    def plot_correlation_matrix_of_actual_prices(cls, actual_prices):
        cls.plot_correlation_matrix(pd.DataFrame(actual_prices.data), 'Correlation Matrix of Actual Prices')

    @classmethod
    def plot_correlation_matrix_of_returns(cls, returns_df):
        cls.plot_correlation_matrix(returns_df, 'Correlation Matrix of Returns')

    @staticmethod
    def plot_correlation_matrix(returns_df, title):
        fig, ax = plt.subplots(figsize=(15, 10))
        sns.heatmap(returns_df.corr(), annot=True, linewidths=.5, ax=ax)
        ax.set_title(title)
        plt.show()

    @classmethod
    def plot_pairwise_returns(cls, returns_df):
        if config.OutputFlags.plot_pairwise_returns:
            for comp in itertools.combinations(returns_df.columns, 2):
                cls.plot_returns(returns_df[[*comp]])

    @staticmethod
    def plot_returns(returns_df):
        fig, ax = plt.subplots(figsize=(15, 10))
        returns_df.plot(ax=ax)
        ax.set_xlabel('Weeks')
        ax.set_ylabel('Returns')
        ax.set_title('Returns of Portfolio Values')
        plt.show()

    @classmethod
    def print_portfolio_metrics(cls, portfolio_values_df, risk_free_rate_13_week, bull_bear_split_needed):

        def get_portfolio_metrics(df):
            model_name = df.columns[0]

            returns = (df / df.shift(1)) - 1
            weights = np.asarray([1 / df.shape[1]] * df.shape[1])
            expected_return = np.sum(weights * returns.mean()) * 52
            volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov(), weights))) * np.sqrt(52)

            quarterly_rate = risk_free_rate_13_week / 4
            weekly_rate = (1 + quarterly_rate)**(1 / 13) - 1
            sharpe_ratio = (expected_return - weekly_rate) / volatility

            roll_max = df[model_name].cummax()
            daily_drawdown = df[model_name] / roll_max - 1.0
            max_drawdown = daily_drawdown.cummin().to_numpy()[-1]

            return {
                "Model": model_name,
                "Expected Return": expected_return,
                "Volatility": volatility,
                "Sharpe Ratio": sharpe_ratio,
                "Maximum Drawdown": max_drawdown
            }

        portfolio_columns = ['Model', 'Expected Return', 'Volatility', 'Sharpe Ratio', 'Maximum Drawdown']

        if bull_bear_split_needed:
            all_bull_metrics = []
            all_bear_metrics = []
            for model in portfolio_values_df.columns:
                values_df = portfolio_values_df[[model]]
                middle = int(len(values_df) / 2)
                all_bull_metrics.append(get_portfolio_metrics(values_df.iloc[:middle, :]))
                all_bear_metrics.append(get_portfolio_metrics(values_df.iloc[middle:, :]))

            bull_metrics_df = pd.DataFrame(all_bull_metrics)
            bull_metrics_df["Market Condition"] = "Bull"

            bear_metrics_df = pd.DataFrame(all_bear_metrics)
            bear_metrics_df["Market Condition"] = "Bear"

            combined_df = pd.concat([bull_metrics_df, bear_metrics_df])
            combined_df = combined_df[['Market Condition'] + portfolio_columns]

            cls.display(combined_df.set_index('Market Condition'))
        else:
            portfolio_metrics = []
            for model in portfolio_values_df.columns:
                values_df = portfolio_values_df[[model]]
                portfolio_metrics.append(get_portfolio_metrics(values_df))

            portfolio_metrics_df = pd.DataFrame(portfolio_metrics)[portfolio_columns]

            cls.display(portfolio_metrics_df.set_index('Model'))
        print()

    @classmethod
    def print_portfolio_asset_selection_accuracies(cls, model_results):
        df = pd.DataFrame()
        df["Model"] = [m.label for m in model_results]
        df["Asset Selection Accuracy"] = [m.asset_selection_accuracy for m in model_results]
        df.set_index("Model", inplace=True)
        cls.display(df)
        print()

    @classmethod
    def print_direction_based_metrics(cls, actual_prices, predicted_prices, best_model_names):
        y_true = ReportTransformer.get_trading_decisions(actual_prices.data, actual_prices.data).ravel()

        overall = {}
        per_class = {}

        for prediction in filter(lambda p: p.label in best_model_names, predicted_prices):
            prediction = ReportTransformer.remove_model_prefix(prediction)
            y_pred = ReportTransformer.get_trading_decisions(actual_prices.data, prediction.data).ravel()
            overall[prediction.label] = cls._calculate_overall_results(y_true, y_pred)
            per_class[prediction.label] = cls._calculate_results_per_class(y_true, y_pred)

        cls.display(pd.DataFrame(overall).T)
        cls.display(pd.DataFrame(per_class).T)

    @staticmethod
    def plot_predicted_and_actual_portfolio_values(best_model_results):
        if config.OutputFlags.plot_predicted_and_actual_portfolio_values:

            def plot(model):
                fig, ax = plt.subplots(figsize=(15, 10))
                pd.DataFrame({"actual": model.data, "prediction": model.predicted_result_data}).plot(ax=ax)
                ax.set_xlabel('Weeks')
                ax.set_xlabel('Capital ($)')
                ax.set_title(f'Actual vs Prediction Capital ($) of {model.label}')
                plt.show()

            [plot(m) for m in best_model_results]

    @classmethod
    def print_mapes_between_actual_and_predicted_model_portfolios(cls, best_model_results):
        df = pd.DataFrame([{
            "Model":
            m.label,
            "MAPEs Between Weekly Predicted and Actual Portfolios":
            Stats.mape(m.data, m.predicted_result_data)
        } for m in best_model_results]).set_index("Model")
        cls.display(df)

    @classmethod
    def print_test_mapes(cls):
        models = ["Connecting the Dots", "DeepGLO", "LSTM"]
        mapes = [e * 100 for e in [.0603, .10028966475567343, .09663211554288864]] # They are copied from Colab logs, no calculation here
        df = pd.DataFrame()
        df["Models"] = models
        df["Test MAPEs"] = mapes
        cls.display(df)

    @staticmethod
    def _calculate_overall_results(y_pred, y_true):
        accuracy = accuracy_score(y_true, y_pred)
        precision_weighted = precision_score(y_true, y_pred, average='weighted')
        recall_weighted = recall_score(y_true, y_pred, average='weighted')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        return {
            'Accuracy': accuracy,
            'Weighted Precision': precision_weighted,
            'Weighted Recall': recall_weighted,
            'Weighted F1': f1_weighted
        }

    @classmethod
    def _calculate_results_per_class(cls, y_pred, y_true):
        classes = ['Buy', 'Hold', 'Sell']

        def class_metric_names(metric):
            return [f'{metric} of {c}' for c in classes]

        results_per_class = [
            dict(zip(class_metric_names(metric), result))
            for metric, result in zip(['Precision', 'Recall', 'F1'],
                                      cls._calculate_class_based_metrics(y_true, y_pred, classes))
        ]

        flattened_results_per_class = {k: v for d in results_per_class for k, v in d.items()}
        return flattened_results_per_class

    @staticmethod
    def _calculate_class_based_metrics(ground_truth_decisions, predicted_decisions_of_model, classes):
        precision_classes = precision_score(ground_truth_decisions,
                                            predicted_decisions_of_model,
                                            average=None,
                                            labels=classes)
        recall_classes = recall_score(ground_truth_decisions,
                                      predicted_decisions_of_model,
                                      average=None,
                                      labels=classes)
        f1_classes = f1_score(ground_truth_decisions, predicted_decisions_of_model, average=None, labels=classes)
        return precision_classes, recall_classes, f1_classes

    @staticmethod
    def display(df):
        try:
            display(df)
        except NameError:
            print(df)
