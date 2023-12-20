import argparse
import os

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='crypto_daily_marked')
    parser.add_argument('--horizon', type=int, default=7)
    parser.add_argument('--num_weeks', type=int, default=104)
    return parser.parse_args()


def main(dataset, horizon, num_weeks):
    # since 1st week doesn't contain the last test line of first week in `ForecastDataset`, I start from the second
    # week, and then I parse the first week's output in the `output_formatter_for_portfolio_analysis.py`
    for week in tqdm(range(2, num_weeks + 1), desc='runner.py > week'):
        dataset_name = f'__{dataset}_simple_returns_week_{week}'.replace('_marked', '')
        print(f'Running for {dataset_name}')
        os.system(f'python main.py --dataset {dataset_name} --horizon {horizon} --epoch 30')
    os.system(
        f'python output_formatter_for_portfolio_analysis.py --dataset {dataset.replace("_marked", "")} --num_weeks {num_weeks}')


if __name__ == '__main__':
    args = parse_args()
    main(args.dataset, args.horizon, args.num_weeks)
