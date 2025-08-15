import sys

import numpy as np
import pandas as pd
from hyperopt import fmin, hp, space_eval, tpe

from train_single_step import SingleStep


def main(device, data_path, horizon):
    hpo_max_evals = 100
    hpo_space = {
        "epoch": hp.choice("epoch", [30]),
        "num_of_weeks_in_window": hp.choice("num_of_weeks_in_window", [2, 3, 4, 5]),
        "lr": hp.loguniform("lr", np.log(1e-6), np.log(1e-3)),
        "weight_decay": hp.loguniform("weight_decay", np.log(1e-6), np.log(1e-4)),
        "layers": hp.choice("layers", [2, 3, 5]),
        "conv_channels": hp.choice("conv_channels", [4, 8, 16]),
    }

    def objective(hparams):
        print(f"Trying hyperparameters: {hparams}")
        return get_trainer(data_path, horizon, 1, 104, device, **hparams).run_train_only()

    best = fmin(objective, hpo_space, algo=tpe.suggest, max_evals=hpo_max_evals)
    best_hparams = space_eval(hpo_space, best)
    print(f"Best hparams: {best_hparams}")


def get_trainer(
    data_path,
    horizon,
    week,
    num_weeks,
    device,
    epoch,
    num_of_weeks_in_window,
    lr,
    weight_decay,
    layers,
    conv_channels,
):
    num_assets = len(pd.read_csv(data_path).columns) - 2
    return SingleStep(
        data_path=data_path,
        week=week,
        num_weeks=num_weeks,
        device=device,
        num_nodes=num_assets,
        subgraph_size=int(num_assets * 0.4),
        seq_in_len=num_of_weeks_in_window * horizon,
        horizon=horizon,
        batch_size=30,
        epochs=epoch,
        lr=lr,
        weight_decay=weight_decay,
        layers=layers,
        conv_channels=conv_channels,
        residual_channels=conv_channels,
        skip_channels=conv_channels * 2,
        end_channels=conv_channels * 4,
        training_split=0.6,
        validation_split=0.2,
    )


if __name__ == "__main__":
    device, data_path, horizon = sys.argv[1:]
    main(device, data_path, int(horizon))
