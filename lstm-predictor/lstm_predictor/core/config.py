import numpy as np
from hyperopt import hp

num_epochs_to_run = 100
early_stopping_patience = 5

hpo_max_evals = 100
hpo_space = {
    'lr': hp.loguniform('lr', np.log(0.0001), np.log(0.01)),
    "dropout": hp.uniform("dropout", 0, .5),
    "hidden_size": hp.choice("hidden_size", [16, 32, 64, 128]),
    "num_layers": hp.choice("num_layers", [2, 3]),
    "batch_size": hp.choice("batch_size", [16, 32, 64, 128]),
    "seq_len": hp.choice("seq_len", [7, 14, 21, 28])
}

training_ratio = .6
validation_ratio = .2

log_active = False
horizon = 7
prices_path = '../ticker-collector/out/crypto/daily_20_2189_marked.csv'
predictions_base_path = 'prediction'
num_weeks_to_train = 104
total_weeks = 104
