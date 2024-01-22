import datetime
import os

import numpy as np
import pandas as pd
import torch
from hyperopt import fmin, hp, space_eval, tpe
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# TODO This file has the up-to-date code, prepared so that it can be directly copied and run on Colab
# TODO The code is messy and awaits refactoring and moving under lstm_predictor package


# noinspection PyPep8Naming
def format_time_as_YYYYMMddHHmm(time):
    return time.isoformat(sep='/', timespec='minutes').replace('-', '').replace(':', '').replace('/', '')


class Constants:
    PREDICTION_TIME = format_time_as_YYYYMMddHHmm(datetime.datetime.now())
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# noinspection DuplicatedCode
class Config:
    num_epochs_to_run = 500
    early_stopping_patience = 100
    horizon = 7

    hpo_max_evals = 100
    hpo_space = {
        'lr':
        hp.loguniform('lr', np.log(0.0001), np.log(0.01)),
        "dropout":
        hp.uniform("dropout", 0, .5),
        "hidden_size":
        hp.choice("hidden_size", [8, 16, 32, 64, 128]),
        "num_layers":
        hp.choice("num_layers", [2, 3, 5, 8]),
        "batch_size":
        hp.choice("batch_size", [32, 64, 128, 256]),
        "seq_len":
        hp.choice(
            "seq_len",
            [horizon, horizon * 3, horizon * 7, horizon * 10, horizon * 24, horizon * 30, horizon * 34, horizon * 40])
    }

    training_ratio = .6
    validation_ratio = .2

    log_active = False
    prices_path = '../ticker-collector/out/crypto/daily_20_2189_marked.csv'
    predictions_base_path = 'prediction'
    num_weeks_to_train = 104
    total_weeks = 104


def log(week, message, log_anyway=False):
    if log_anyway or Config.log_active:
        print(f'Week: {week} | {message}')


def to_torch(np_array):
    return torch.from_numpy(np_array).to(Constants.DEVICE)


# noinspection DuplicatedCode
def read_return_ratios(data_path, week, num_weeks, training_ratio, fill_zero_for_the_first_horizon_samples=False):
    """
    :return: Log returns, the last line is the last split point in the data when week == num_weeks.
             If week == num_weeks -1, the last line is the second last split point in the data, and so on.
             If fill_zero_for_the_first_horizon_samples is False, then the first output np array will have `horizon`
             fewer rows.
    """
    assert 0 <= week <= num_weeks, f"week must be between 0 and num_weeks (inclusive): 0 <= {week} <= {num_weeks}"
    assert num_weeks >= 0, f"num_weeks must be >= 0, got {num_weeks}"

    raw_data = pd.read_csv(data_path)

    truncate_index = len(raw_data)
    stopping_point = -1
    truncation_mark = num_weeks - week

    for point in reversed(raw_data['split_point']):
        if stopping_point == truncation_mark:
            break

        if point:
            stopping_point += 1

        truncate_index -= 1

    return _convert_to_simple_returns(raw_data.loc[:truncate_index].drop(['split_point'], axis=1),
                                      fill_zero_for_the_first_horizon_samples).to_numpy()


# noinspection DuplicatedCode
def _convert_to_simple_returns(df, fill_zero_for_the_first_horizon_samples):
    np_form = df.drop(['Date'], axis=1).to_numpy()
    simple_returns = np_form[Config.horizon:] / np_form[:-Config.horizon]
    if fill_zero_for_the_first_horizon_samples:
        simple_returns = np.vstack([np.full((Config.horizon, np_form.shape[1]), .01), simple_returns])
    return pd.DataFrame(simple_returns, columns=df.columns[1:])


# noinspection DuplicatedCode
class LSTMModel(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


# noinspection DuplicatedCode
class WindowedSplitDataLoader:

    def __init__(self, data_np, seq_len, horizon, training_ratio, validation_ratio):
        self._seq_length = seq_len
        self._horizon = horizon
        self._dat = data_np
        self._num_rows, self._num_cols = self._dat.shape
        self._split(int(training_ratio * self._num_rows), int((training_ratio + validation_ratio) * self._num_rows))
        self._last_day = self._dat[-seq_len:].reshape((1, seq_len, -1)).astype(np.float32)

    @property
    def training(self):
        return self._training

    @property
    def validation(self):
        return self._validation

    @property
    def training_and_validation(self):
        return self._training_and_validation

    @property
    def test(self):
        return self._test

    @property
    def last_day(self):
        return self._last_day

    def _split(self, train_end_index, valid_end_index):
        training_set_indices = range(self._seq_length + self._horizon - 1, train_end_index)
        validation_set_indices = range(train_end_index, valid_end_index)
        test_set = range(valid_end_index, self._num_rows)
        self._training = self._batchify(training_set_indices)
        self._validation = self._batchify(validation_set_indices)
        self._training_and_validation = {
            'X': np.concatenate((self.training['X'], self.validation['X'])),
            'y': np.concatenate((self.training['y'], self.validation['y']))
        }
        self._test = self._batchify(test_set)

    # noinspection PyPep8Naming
    def _batchify(self, idx_set):
        n = len(idx_set)
        X = np.zeros((n, self._seq_length, self._num_cols))
        y = np.zeros((n, self._num_cols))
        for i in range(n):
            end = idx_set[i] - self._horizon + 1
            start = end - self._seq_length
            X[i, :, :] = self._dat[start:end, :]
            y[i, :] = self._dat[idx_set[i], :]
        return {'X': X.astype(np.float32), 'y': y.astype(np.float32)}


class Runner:

    def __init__(self, week):
        self._week = week
        self.space = Config.hpo_space

    def get_best_hparams(self):
        """
        Trains the model multiple times, finds the best hparams and returns.
        """
        best = fmin(self.objective, self.space, algo=tpe.suggest, max_evals=Config.hpo_max_evals)
        best_hparams = space_eval(self.space, best)
        log(self._week, f'Best: {best}, with hparams: {best_hparams}')
        return best_hparams

    def predict_with_best_hparams(self, hparams, train_new_model):
        log(self._week,
            f"Training {'on' if train_new_model else 'off'}. Predicting with hyperparams: {hparams}",
            log_anyway=True)

        hparams_str = self._create_hparams_str(hparams)
        model_path = f'models/experiment_{Constants.PREDICTION_TIME}/{hparams_str}/model.pt'
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        normalized_training_ratio = Config.training_ratio / (Config.training_ratio + Config.validation_ratio) - .01
        normalized_validation_ratio = Config.validation_ratio / (Config.training_ratio + Config.validation_ratio) - .01

        data_np = read_return_ratios(Config.prices_path, self._week, Config.total_weeks, normalized_training_ratio)
        num_assets = data_np.shape[1]

        dl = WindowedSplitDataLoader(data_np, hparams['seq_len'], Config.horizon, normalized_training_ratio,
                                     normalized_validation_ratio)
        training_targets = to_torch(dl.training['y'])
        validation_targets = to_torch(dl.validation['y'])

        model = LSTMModel(input_size=num_assets,
                          hidden_size=hparams['hidden_size'],
                          num_layers=hparams['num_layers'],
                          dropout=hparams['dropout'])

        model = model.to(Constants.DEVICE)
        criterion = nn.MSELoss()
        optimizer = torch.optim.RMSprop(model.parameters(), lr=hparams['lr'], weight_decay=0.01)
        training_loader = DataLoader(TensorDataset(to_torch(dl.training['X']), training_targets),
                                     batch_size=hparams['batch_size'])
        patience = Config.early_stopping_patience
        early_stop_counter = 0
        best_val_loss = float('inf')
        training_loss = float('inf')
        val_loss = float('inf')
        progress_bar = tqdm(
            range(Config.num_epochs_to_run),
            desc=f'Losses (MSE) | Train: {training_loss:.3f}, Val: {val_loss:.3f}, Best-Val: {best_val_loss:.3f}')

        if train_new_model:
            for epoch in progress_bar:
                model.train()
                for i, data in enumerate(training_loader):
                    inputs, targets = data
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    training_loss = loss.item()

                model.eval()
                with torch.no_grad():
                    validation_outputs = model(to_torch(dl.validation['X']))
                    val_loss = criterion(validation_outputs, to_torch(dl.validation['y'])).item()
                    progress_bar.set_description(
                        f'Losses (MSE) | Train: {training_loss:.3f}, Val: {val_loss:.3f}, Best-Val: {best_val_loss:.3f}'
                    )
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        early_stop_counter = 0
                        torch.save(model.state_dict(), model_path)
                    else:
                        early_stop_counter += 1
                        if early_stop_counter >= patience:
                            log(
                                self._week,
                                f'Validation loss has not improved for {patience} epochs. Early stopping at epoch: {epoch}'
                            )
                            break

        model.load_state_dict(torch.load(model_path))
        model.eval()
        with torch.no_grad():
            training_outputs = model(to_torch(dl.training['X']))
            validation_outputs = model(to_torch(dl.validation['X']))
            last_day_outputs = model(to_torch(dl.last_day))

            training_mape = torch.mean(torch.abs((training_targets - training_outputs) / (training_targets + 1e-6)))
            validation_mape = torch.mean(
                torch.abs((validation_targets - validation_outputs) / (validation_targets + 1e-6)))

            log(self._week,
                f'[TEST_EVAL] Training / Validation MAPEs: {training_mape:.5f} / {validation_mape:.5f}',
                log_anyway=True)
            progress_bar.set_description(progress_bar.desc +
                                         f', Train MAPE: {training_mape:.3f}, Val MAPE: {validation_mape:.3f}')
            progress_bar.display()
            return last_day_outputs.cpu().numpy()

    def objective(self, hparams):
        """
        :return: Validation loss
        """
        log(self._week, f"Trying hyperparams: {hparams}")

        hparams_str = self._create_hparams_str(hparams)
        model_path = f'models/experiment_{Constants.PREDICTION_TIME}_week{self._week}/{hparams_str}/model.pt'
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        self.create_figs_directory(hparams_str)
        writer = self._create_summary_writer(hparams_str)

        data_np = read_return_ratios(Config.prices_path, self._week, Config.total_weeks, Config.training_ratio)
        num_assets = data_np.shape[1]
        dl = WindowedSplitDataLoader(data_np, hparams['seq_len'], Config.horizon, Config.training_ratio,
                                     Config.validation_ratio)
        training_targets = to_torch(dl.training['y'])
        validation_targets = to_torch(dl.validation['y'])
        test_targets = to_torch(dl.test['y'])

        model = LSTMModel(input_size=num_assets,
                          hidden_size=hparams['hidden_size'],
                          num_layers=hparams['num_layers'],
                          dropout=hparams['dropout'])
        model = model.to(Constants.DEVICE)
        criterion = nn.MSELoss()
        optimizer = torch.optim.RMSprop(model.parameters(), lr=hparams['lr'], weight_decay=0.01)
        # optimizer = torch.optim.Adam(model.parameters(), lr=hparams['lr'])
        training_loader = DataLoader(TensorDataset(to_torch(dl.training['X']), training_targets),
                                     batch_size=hparams['batch_size'])

        patience = Config.early_stopping_patience
        early_stop_counter = 0
        best_val_loss = float('inf')
        training_loss = float('inf')
        val_loss = float('inf')
        progress_bar = tqdm(
            range(Config.num_epochs_to_run),
            desc=f'Losses (MSE) | Train: {training_loss:.3f}, Val: {val_loss:.3f}, Best-Val: {best_val_loss:.3f}')
        for epoch in progress_bar:
            model.train()
            for i, data in enumerate(training_loader):
                inputs, targets = data
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                training_loss = loss.item()
                writer.add_scalar('Training Loss', training_loss, epoch)

            model.eval()
            with torch.no_grad():
                validation_outputs = model(to_torch(dl.validation['X']))
                val_loss = criterion(validation_outputs, to_torch(dl.validation['y'])).item()
                writer.add_scalar('Validation Loss', val_loss, epoch)
                progress_bar.set_description(
                    f'Losses (MSE) | Train: {training_loss:.3f}, Val: {val_loss:.3f}, Best-Val: {best_val_loss:.3f}')
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    early_stop_counter = 0
                    torch.save(model.state_dict(), model_path)
                else:
                    early_stop_counter += 1
                    if early_stop_counter >= patience:
                        log(
                            self._week,
                            f'Validation loss has not improved for {patience} epochs. Early stopping at epoch: {epoch}')
                        break

        model.load_state_dict(torch.load(model_path))
        model.eval()
        with torch.no_grad():
            training_outputs = model(to_torch(dl.training['X']))
            validation_outputs = model(to_torch(dl.validation['X']))
            test_outputs = model(to_torch(dl.test['X']))

            val_loss = criterion(validation_outputs, to_torch(dl.validation['y'])).item()

            # MAPE
            training_mape = torch.mean(torch.abs((training_targets - training_outputs) / (training_targets + 1e-6)))
            validation_mape = torch.mean(
                torch.abs((validation_targets - validation_outputs) / (validation_targets + 1e-6)))
            test_mape = torch.mean(torch.abs((test_targets - test_outputs) / (test_targets + 1e-6)))

            # MAE
            training_mae = torch.mean(torch.abs(training_targets - training_outputs))
            validation_mae = torch.mean(torch.abs(validation_targets - validation_outputs))
            test_mae = torch.mean(torch.abs(test_targets - test_outputs))

            # RMSE
            training_rmse = self.rmse(training_outputs, training_targets)
            validation_rmse = self.rmse(validation_outputs, validation_targets)
            test_rmse = self.rmse(test_outputs, test_targets)

            # Logging
            log(self._week, f'Training MAPE/MAE/RMSE: {training_mape:.3f}/{training_mae:.3f}/{training_rmse:.3f} - '
                f'Validation MAPE/MAE/RMSE: {validation_mape:.3f}/{validation_mae:.3f}/{validation_rmse:.3f} - '
                f'Test MAPE/MAE/RMSE: {test_mape:.3f}/{test_mae:.3f}/{test_rmse:.3f}',
                log_anyway=True)
            return val_loss

    @staticmethod
    def rmse(predictions, targets):
        return torch.sqrt(torch.mean((predictions - targets)**2))

    def create_figs_directory(self, hparams_str):
        os.makedirs(f"figs/experiment_{Constants.PREDICTION_TIME}_week{self._week}/{hparams_str}")

    def _create_summary_writer(self, hparams_str):
        return SummaryWriter(log_dir=f'runs/experiment_{Constants.PREDICTION_TIME}_week{self._week}/{hparams_str}')

    @staticmethod
    def _create_hparams_str(hparams):
        return f"bs{hparams['batch_size']}-dr{hparams['dropout']}-hid{hparams['hidden_size']}-lr{hparams['lr']}-numl{hparams['num_layers']}-seql{hparams['seq_len']}"

    def save_predictions(self, predictions):
        dir_path = f'{Config.predictions_base_path}/{Constants.PREDICTION_TIME}/weeks'
        os.makedirs(dir_path, exist_ok=True)
        np.save(f'{dir_path}/{self._week + 1}', predictions)


def main_quick():
    if os.path.exists('figs'):
        os.system('rm -rf figs')
    if os.path.exists('runs'):
        os.system('rm -rf runs')
    if os.path.exists('models'):
        os.system('rm -rf models')
    runner = Runner(week=0)
    best_hparams = runner.get_best_hparams()
    print(f'Best hyperparameters of week {0}: {best_hparams}')
    for week in tqdm(range(Config.num_weeks_to_train)):
        runner = Runner(week=week)
        if week % 5 == 0:
            last_day_outputs = runner.predict_with_best_hparams(best_hparams, True)
        else:
            last_day_outputs = runner.predict_with_best_hparams(best_hparams, False)
        runner.save_predictions(last_day_outputs)


if __name__ == '__main__':
    main_quick()
