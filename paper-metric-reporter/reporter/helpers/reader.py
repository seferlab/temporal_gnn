import numpy as np
import pandas as pd

from models import Data


class Reader:

    @staticmethod
    def get_actual_prices(path, num_weeks):
        actual_prices = pd.read_csv(path)
        return Data(
            actual_prices[actual_prices["split_point"]].drop(columns=["Date", "split_point"])[-num_weeks:].to_numpy(),
            "Actual Prices",
        )

    @staticmethod
    def get_predicted_returns(predictions_path, num_weeks, num_assets, data_type="simple-return"):
        predicted_prices = np.empty([0, num_assets])
        for i in range(1, num_weeks + 1):
            data = np.load(f"{predictions_path}/{i}.npy")[-1:]
            data = np.exp(data) if data_type == "log-return" else data
            predicted_prices = np.concatenate((predicted_prices, data))
        return predicted_prices
