import argparse
import os
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from hyperopt import fmin, hp, space_eval, tpe
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import data_dict, data_information, device
from model.FourierGNN import FGN
from utils.utils import evaluate, load_model, save_model


def main():
    args = parse_arguments()

    hparams = None
    for week in tqdm(range(args.starting_week, 105), desc="week"):
        print(f"Running week: {week}")
        if week % args.hparam_search_freq == 0 or not hparams:
            hparams = search_hyperparameters(
                args.data,
                args.pre_length,
                args.train_epochs,
                args.batch_size,
                args.train_ratio,
                args.val_ratio,
                week,
            )
        hparams_as_args = SimpleNamespace(**hparams)
        run(hparams_as_args, week)


def search_hyperparameters(data, pre_length, train_epochs, batch_size, train_ratio, val_ratio, week):
    print(f"Searching hyperparameters at week: {week}")
    hpo_max_evals = 100

    def objective(hparams):
        hparams_as_args = SimpleNamespace(**hparams)
        return run(hparams_as_args, week, hparam_search=True)

    args = {
        "data": hp.choice("data", [data]),
        "pre_length": hp.choice("pre_length", [pre_length]),
        "train_epochs": hp.choice("train_epochs", [train_epochs]),
        "batch_size": hp.choice("batch_size", [batch_size]),
        "train_ratio": hp.choice("train_ratio", [train_ratio]),
        "val_ratio": hp.choice("val_ratio", [val_ratio]),
        "validate_freq": hp.choice("validate_freq", [1]),
        "seq_length": hp.choice("seq_length", [pre_length, pre_length * 2, pre_length * 3, pre_length * 4]),
        "embed_size": hp.choice("embed_size", [8, 16, 32, 64, 128]),
        "hidden_size": hp.choice("hidden_size", [8, 16, 32, 64, 128]),
        "learning_rate": hp.loguniform("learning_rate", np.log(1e-6), np.log(1e-3)),
        "exponential_decay_step": hp.choice("exponential_decay_step", [2, 3, 5, 8, 13]),
        "decay_rate": hp.choice("decay_rate", [0.2, 0.3, 0.5, 0.8, 0.9]),
    }

    best = fmin(objective, args, algo=tpe.suggest, max_evals=hpo_max_evals)
    best_hparams = space_eval(args, best)
    print(f"Best hparams: {best_hparams}")

    return best_hparams


def run(args, week, hparam_search=False):
    result_train_file = create_output_directories(args.data)
    data_info = data_information[args.data]

    Data = data_dict[args.data]

    train_set = Data(
        root_path=data_info["root_path"],
        week=week,
        flag="train",
        seq_len=args.seq_length,
        pre_len=args.pre_length,
        type=data_info["type"],
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    val_set = Data(
        root_path=data_info["root_path"],
        week=week,
        flag="val",
        seq_len=args.seq_length,
        pre_len=args.pre_length,
        type=data_info["type"],
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    test_set = Data(
        root_path=data_info["root_path"],
        week=week,
        flag="test",
        seq_len=args.seq_length,
        pre_len=args.pre_length,
        type=data_info["type"],
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    train_dataloader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )
    val_dataloader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )
    test_dataloader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    model = FGN(
        pre_length=args.pre_length,
        embed_size=args.embed_size,
        seq_length=args.seq_length,
        hidden_size=args.hidden_size,
    ).to(device)
    my_optim = torch.optim.RMSprop(params=model.parameters(), lr=args.learning_rate, eps=1e-08)
    my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=my_optim, gamma=args.decay_rate)
    forecast_loss = nn.MSELoss(reduction="mean").to(device)

    return execute_training_and_prediction(
        args,
        week,
        model,
        train_dataloader,
        forecast_loss,
        my_optim,
        my_lr_scheduler,
        val_dataloader,
        result_train_file,
        test_dataloader,
        test_set,
        hparam_search,
    )


def parse_arguments():
    parser = argparse.ArgumentParser(description="fourier graph network for multivariate time series forecasting")
    parser.add_argument(
        "--hparam_search_freq",
        type=int,
        default=10,
        help="hyperparameter search frequency",
    )
    parser.add_argument("--data", type=str, default="ECG", help="data set")
    parser.add_argument("--starting_week", type=int, default=2, help="starting week")
    parser.add_argument("--seq_length", type=int, default=12, help="sequence length")
    parser.add_argument("--pre_length", type=int, default=12, help="predict length")
    parser.add_argument("--embed_size", type=int, default=128, help="hidden dimensions")
    parser.add_argument("--hidden_size", type=int, default=256, help="hidden dimensions")
    parser.add_argument("--train_epochs", type=int, default=100, help="train epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="input data batch size")
    parser.add_argument("--learning_rate", type=float, default=0.00001, help="optimizer learning rate")
    parser.add_argument("--exponential_decay_step", type=int, default=5)
    parser.add_argument("--validate_freq", type=int, default=1)
    parser.add_argument("--decay_rate", type=float, default=0.5)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    args = parser.parse_args()

    print(f"Training arguments: {args}")
    return args


def create_output_directories(data):
    result_train_file = os.path.join("output", data, "train")
    result_test_file = os.path.join("output", data, "test")
    if not os.path.exists(result_train_file):
        os.makedirs(result_train_file)
    if not os.path.exists(result_test_file):
        os.makedirs(result_test_file)
    return result_train_file


def execute_training_and_prediction(
    args,
    week,
    model,
    train_dataloader,
    forecast_loss,
    my_optim,
    my_lr_scheduler,
    val_dataloader,
    result_train_file,
    test_dataloader,
    test_set,
    hparam_search,
):
    best_val_loss = np.inf
    early_stop_patience = 20
    early_stop_counter = 0
    val_loss = np.nan

    for epoch in tqdm(range(args.train_epochs * 2), desc=f"epoch at week {week}"):
        epoch_start_time = time.time()
        model.train()
        loss_total = 0
        cnt = 0
        for index, (x, y) in enumerate(train_dataloader):
            cnt += 1
            y = y.float().to(device)
            x = x.float().to(device)
            forecast = model(x)
            y = y.permute(0, 2, 1).contiguous()
            loss = forecast_loss(forecast, y)
            loss.backward()
            my_optim.step()
            loss_total += float(loss)

        if (epoch + 1) % args.exponential_decay_step == 0:
            my_lr_scheduler.step()
        if (epoch + 1) % args.validate_freq == 0:
            val_loss = validate(model, val_dataloader, forecast_loss)
            if val_loss < best_val_loss:
                early_stop_counter = 0
                best_val_loss = val_loss
                save_model(model, result_train_file)
            else:
                early_stop_counter += 1

        print("| end of epoch {:3d} | time: {:5.2f}s | train_total_loss {:5.4f} | val_loss {:5.4f}".format(
            epoch, (time.time() - epoch_start_time), loss_total / cnt, val_loss))

        if early_stop_counter >= early_stop_patience:
            print(f"Early stopping at epoch {epoch}!")
            break
    print(f"Training has been completed at week {week}.")

    if not hparam_search:
        test(args, week, test_dataloader, test_set)

    return best_val_loss


def validate(model, vali_loader, forecast_loss):
    model.eval()
    cnt = 0
    loss_total = 0
    preds = []
    trues = []
    for i, (x, y) in enumerate(vali_loader):
        cnt += 1
        y = y.float().to(device)
        x = x.float().to(device)
        forecast = model(x)
        y = y.permute(0, 2, 1).contiguous()
        loss = forecast_loss(forecast, y)
        loss_total += float(loss)
        forecast = forecast.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        preds.append(forecast)
        trues.append(y)
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    score = evaluate(trues, preds)
    print(f"valid RAW : MAPE {score[0]:7.9%}; MAE {score[1]:7.9f}; RMSE {score[2]:7.9f}; A20 {score[3]:7.9f}.")
    model.train()
    return loss_total / cnt


def test(args, week, test_dataloader, test_set):
    result_test_file = "output/" + args.data + "/train"
    model = load_model(result_test_file)
    model.eval()
    preds = []
    trues = []

    for index, (x, y) in enumerate(test_dataloader):
        y = y.float().to(device)
        x = x.float().to(device)
        forecast = model(x)
        y = y.permute(0, 2, 1).contiguous()
        forecast = forecast.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        preds.append(forecast)
        trues.append(y)

    preds = np.concatenate(preds, axis=0).transpose(0, 2, 1)[:, -1, :]
    preds = test_set.standard_scaler.inverse_transform(preds)
    trues = np.concatenate(trues, axis=0).transpose(0, 2, 1)[:, -1, :]
    trues = test_set.standard_scaler.inverse_transform(trues)
    score = evaluate(trues, preds)
    print(f"test RAW : MAPE {score[0]:7.9%}; MAE {score[1]:7.9f}; RMSE {score[2]:7.9f}; A20 {score[3]:7.9f}.")
    np.save(f"output/{args.data}/{week - 1}.npy", preds[-args.pre_length].reshape((1, -1)))


if __name__ == "__main__":
    main()
