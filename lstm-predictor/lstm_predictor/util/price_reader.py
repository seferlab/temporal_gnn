import numpy as np
import pandas as pd

from lstm_predictor.core import config


def read_prices(data_path, week, num_weeks):
    raw_data = pd.read_csv(data_path)

    truncate_index = len(raw_data)
    stopping_point = -1
    truncation_mark = num_weeks - week

    for point in reversed(raw_data['split_point']):
        if stopping_point == truncation_mark:
            break

        if point:
            stopping_point += 1

        truncate_index -= 1

    return _convert_to_simple_returns(raw_data.loc[:truncate_index].drop(['split_point'], axis=1)).to_numpy()


def _convert_to_simple_returns(df):
    np_form = df.drop(['Date'], axis=1).to_numpy()
    simple_returns = np_form[config.horizon:] / np_form[:-config.horizon]
    simple_returns_with_initial_zero_row = np.vstack([np.full((config.horizon, np_form.shape[1]), .01), simple_returns])
    return pd.DataFrame(simple_returns_with_initial_zero_row, columns=df.columns[1:])
