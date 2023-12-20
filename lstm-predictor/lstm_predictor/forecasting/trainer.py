import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from lstm_predictor.core import config, constants
from lstm_predictor.core.constants import PREDICTION_TIME
from lstm_predictor.forecasting.lstm_model import LSTMModel
from lstm_predictor.util import log
# noinspection PyPep8Naming
from lstm_predictor.util.data_loader import DataLoader as DL


class Trainer:

    def __init__(self, prices_np, week):
        self._prices_np = prices_np
        self._num_assets = prices_np.shape[1]
        self._week = week
        self._writer = SummaryWriter(log_dir=f'runs/experiment_{PREDICTION_TIME}_{self._week}')

    def __del__(self):
        self._writer.close()

    def train(self, num_epochs, hparams, training_ratio, validation_ratio):
        trainer = self._train(num_epochs, training_ratio, validation_ratio, **hparams)
        trainer.model.eval()
        with torch.no_grad():
            validation_loss = trainer.get_validation_loss()
            test_loss, test_mape = trainer.get_test_loss_and_mape()
            log(self._week, f'Validation Loss: {validation_loss} | Test Loss: {test_loss} | Test Mape: {test_mape}')
            self._writer.add_hparams(hparams, {
                "validation_loss": validation_loss,
                "test_loss": test_loss,
                "test_mape": test_mape
            })
            return validation_loss

    def predict_with_best_hparams(self, num_epochs, best_hparams, training_ratio, validation_ratio):
        trainer = self._train(num_epochs, training_ratio, validation_ratio, **best_hparams)
        trainer.model.eval()
        with torch.no_grad():
            validation_loss = trainer.get_validation_loss()
            log(self._week, f'Prediction validation loss: {validation_loss}')
            self._writer.add_hparams(best_hparams, {"validation_loss": validation_loss})
            return trainer.get_last_day_predictions()

    def _train(self, num_epochs, training_ratio, validation_ratio, batch_size, dropout, hidden_size, lr, num_layers,
               seq_len):
        trainer = _Trainer(self._week, self._prices_np, seq_len, training_ratio, validation_ratio)
        trainer.train(self._writer, num_epochs, hidden_size, num_layers, lr, dropout, batch_size)
        return trainer


class _Trainer:

    def __init__(self, week, prices_np, seq_len, training_ratio, validation_ratio):
        self._week = week
        self._num_assets = prices_np.shape[1]
        self._seq_len = seq_len
        self._training_ratio = training_ratio
        self._validation_ratio = validation_ratio
        self._dl = DL(data_np=prices_np,
                      seq_len=seq_len,
                      training_ratio=training_ratio,
                      validation_ratio=validation_ratio)
        self._load_tensors()
        self.model = None
        self._criterion = None

    def train(self, writer, num_epochs, hidden_size, num_layers, lr, dropout, batch_size):
        self.model = LSTMModel(input_size=self._num_assets,
                               hidden_size=hidden_size,
                               num_layers=num_layers,
                               dropout=dropout)
        self.model = self.model.to(constants.DEVICE)
        self._criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        training_loader = DataLoader(TensorDataset(self._training_X, self._training_y), batch_size=batch_size)

        best_val_loss = float('inf')
        patience = config.early_stopping_patience
        early_stop_counter = 0

        for epoch in range(num_epochs):
            self.model.train()
            for i, data in enumerate(training_loader):
                inputs, targets = data
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self._criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                training_loss = loss.item()
                writer.add_scalar('Training Loss', training_loss, epoch)
            self.model.eval()
            with torch.no_grad():
                val_loss = self._criterion(self.model(self._validation_X), self._validation_y).item()
                writer.add_scalar('Validation Loss', val_loss, epoch)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                    if early_stop_counter >= patience:
                        log(
                            self._week, f'Validation loss has not improved for {patience} epochs. '
                            f'Stopping training at epoch {epoch}.')
                        break

    def get_validation_loss(self):
        return self._criterion(self.model(self._validation_X), self._validation_y).item()

    def get_test_loss_and_mape(self):
        test_output = self.model(self._test_X)
        test_loss = self._criterion(test_output, self._test_y).item()
        test_mape = torch.mean(torch.abs((test_output - self._test_y) / test_output))
        return test_loss, test_mape

    def get_last_day_predictions(self):
        return self.model(torch.from_numpy(self._dl.last_day).to(constants.DEVICE)).cpu().numpy()

    def _load_tensors(self):
        self._training_X = torch.from_numpy(self._dl.training['X']).to(constants.DEVICE)
        self._training_y = torch.from_numpy(self._dl.training['y']).to(constants.DEVICE)
        self._validation_X = torch.from_numpy(self._dl.validation['X']).to(constants.DEVICE)
        self._validation_y = torch.from_numpy(self._dl.validation['y']).to(constants.DEVICE)
        self._test_X = torch.from_numpy(self._dl.test['X']).to(constants.DEVICE)
        self._test_y = torch.from_numpy(self._dl.test['y']).to(constants.DEVICE)
