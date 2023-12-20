from tqdm import tqdm

from lstm_predictor.core import config
from lstm_predictor.forecasting.runner import Runner


def main():
    for i in tqdm(range(config.num_weeks_to_train)):
        runner = Runner(week=i)
        runner.run()


if __name__ == '__main__':
    main()
