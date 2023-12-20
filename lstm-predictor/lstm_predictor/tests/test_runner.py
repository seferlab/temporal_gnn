from unittest import TestCase

import numpy as np
from hyperopt import hp

import test_config
from lstm_predictor.core import config
from lstm_predictor.forecasting.runner import Runner


class TestRunner(TestCase):

    def setUp(self) -> None:
        config.prices_path = test_config.prices_path
        config.num_weeks_to_train = 3
        config.total_weeks = 4
        config.training_ratio = .6
        config.validation_ratio = .2
        config.num_epochs_to_run = 1
        config.hpo_max_evals = 1
        config.hpo_space = {
            'lr': hp.loguniform('lr', np.log(0.01), np.log(0.02)),
            "dropout": hp.uniform("dropout", 0, .5),
            "hidden_size": hp.choice("hidden_size", [16]),
            "num_layers": hp.choice("num_layers", [2]),
            "batch_size": hp.choice("batch_size", [16]),
            "seq_len": hp.choice("seq_len", [7])
        }

    def test_process(self):
        runner = Runner(3)
        predictions = runner.process()
        # Just print for the moment, no assertions
        print(predictions)
