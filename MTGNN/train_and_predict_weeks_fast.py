import sys

import pandas as pd
from tqdm import tqdm

from train_single_step import SingleStep


class Config:
    conv_channels = 4
    epoch = 2000
    layers = 3
    lr = 0.00014215115225721316
    num_of_weeks_in_window = 2
    weight_decay = 3.485648994073535e-05


def main(device, data_path, horizon, num_weeks):
    num_assets = len(pd.read_csv(data_path).columns) - 2
    end_indexes = [50, 75, 90, num_weeks]
    i = 0
    for week in tqdm([25, 50, 75, 90], desc='train_and_predict_weeks_fast.py > week'):
        print(f"Training week: {week}")
        get_trainer(data_path, horizon, num_assets, week, num_weeks, device).run_train_only()
        for predicting_week in range(week, end_indexes[i]):
            print(f"Predicting week: {predicting_week}")
            get_predictor(data_path, horizon, num_assets, predicting_week, num_weeks,
                          device).predict_with_the_best_model()
        i += 1


def get_trainer(data_path, horizon, num_assets, week, num_weeks, device):
    return SingleStep(data_path=data_path,
                      week=week,
                      num_weeks=num_weeks,
                      device=device,
                      num_nodes=num_assets,
                      subgraph_size=int(num_assets * .4),
                      seq_in_len=Config.num_of_weeks_in_window * horizon,
                      horizon=horizon,
                      batch_size=30,
                      epochs=Config.epoch,
                      lr=Config.lr,
                      weight_decay=Config.weight_decay,
                      layers=Config.layers,
                      conv_channels=Config.conv_channels,
                      residual_channels=Config.conv_channels,
                      skip_channels=Config.conv_channels * 2,
                      end_channels=Config.conv_channels * 4,
                      training_split=.74,
                      validation_split=.24)


def get_predictor(data_path, horizon, num_assets, week, num_weeks, device):
    return SingleStep(data_path=data_path,
                      week=week,
                      num_weeks=num_weeks,
                      device=device,
                      num_nodes=num_assets,
                      subgraph_size=int(num_assets * .4),
                      seq_in_len=Config.num_of_weeks_in_window * horizon,
                      horizon=horizon,
                      batch_size=30,
                      epochs=Config.epoch,
                      lr=Config.lr,
                      weight_decay=Config.weight_decay,
                      layers=Config.layers,
                      conv_channels=Config.conv_channels,
                      residual_channels=Config.conv_channels,
                      skip_channels=Config.conv_channels * 2,
                      end_channels=Config.conv_channels * 4,
                      training_split=.74,
                      validation_split=.24,
                      run_for_prediction=True)


if __name__ == "__main__":
    device, data_path, horizon, num_weeks = sys.argv[1:]
    main(device, data_path, int(horizon), int(num_weeks))
