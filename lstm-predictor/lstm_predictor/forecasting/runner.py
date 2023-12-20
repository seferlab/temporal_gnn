import os

import numpy as np
from hyperopt import fmin, space_eval, tpe

from lstm_predictor.core import config, constants
from lstm_predictor.forecasting.trainer import Trainer
from lstm_predictor.util import log
# noinspection PyPep8Naming
from lstm_predictor.util.price_reader import read_prices


class Runner:

    def __init__(self, week):
        self._week = week
        self.space = config.hpo_space

    def run(self):
        next_week_prediction = self.process()
        self.save_predictions(next_week_prediction)

    def process(self):
        best_hparams = self.train()
        return self.predict_next_week(best_hparams)

    def train(self):
        log(self._week, 'Started running.')
        best = fmin(self.objective, self.space, algo=tpe.suggest, max_evals=config.hpo_max_evals)
        best_hparams = space_eval(self.space, best)
        log(self._week, f'Best: {best}, with hparams: {best_hparams}')
        return best_hparams

    def objective(self, hparams):
        trainer = Trainer(read_prices(config.prices_path, self._week, config.total_weeks), self._week)
        return trainer.train(num_epochs=config.num_epochs_to_run,
                             hparams=hparams,
                             training_ratio=config.training_ratio,
                             validation_ratio=config.validation_ratio)

    def predict_next_week(self, best_hparams):
        best_trainer = Trainer(read_prices(config.prices_path, self._week, config.total_weeks), self._week)
        return best_trainer.predict_with_best_hparams(num_epochs=config.num_epochs_to_run,
                                                      best_hparams=best_hparams,
                                                      **self._normalize_training_validation_ratios())

    @staticmethod
    def _normalize_training_validation_ratios():
        total = config.training_ratio + config.validation_ratio
        return {
            "training_ratio": config.training_ratio / total - .01,
            "validation_ratio": config.validation_ratio / total - .01
        }

    def save_predictions(self, predictions):
        dir_path = f'{config.predictions_base_path}/{constants.PREDICTION_TIME}/weeks'
        os.makedirs(dir_path, exist_ok=True)
        np.save(f'{dir_path}/{self._week + 1}', predictions)
