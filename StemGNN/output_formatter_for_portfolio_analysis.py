import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='crypto_daily_marked')
    parser.add_argument('--num_weeks', type=int, default=104)
    return parser.parse_args()


def main(processing_time, dataset, num_weeks):
    dataset_type = 'simple_returns'
    output_dir = f'prediction/{processing_time}/__{dataset}_{dataset_type}'
    os.makedirs(output_dir, exist_ok=True)
    for week_plus_1 in tqdm(range(2, num_weeks + 1), desc='output_formatter_for_portfolio_analysis.py > week'):
        week = week_plus_1 - 1
        path = f'output/__{dataset}_{dataset_type}_week_{week_plus_1}/test/predict.csv'

        # StemGNN removes the last `horizon - 1` days from the test target, so we consider the second-to-last
        # element of next week as the current week's prediction.
        prediction = pd.read_csv(path)[-2:-1].to_numpy()
        output_path = f'{output_dir}/{week}.npy'
        np.save(output_path, prediction)


if __name__ == '__main__':
    args = parse_args()
    processing_time = datetime.now().strftime('%Y%m%d%H%M')
    main(processing_time, args.dataset, args.num_weeks)
