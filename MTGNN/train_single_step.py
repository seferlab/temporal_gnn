import math
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from constants import Constants
from net import gtnet
from trainer import Optim
from util import *


@dataclass
class SingleStep:
    data_path: str
    week: int
    num_weeks: int
    device: str
    num_nodes: int
    subgraph_size: int
    seq_in_len: int
    horizon: int
    batch_size: int
    epochs: int
    run_for_prediction: bool = False
    normalize: int = 2
    training_split: float = .6
    validation_split: float = .2
    conv_channels: int = 16
    residual_channels: int = 16
    skip_channels: int = 32
    end_channels: int = 64
    layers: int = 2
    lr: float = .0001
    weight_decay: float = .00001
    log_interval = 2000
    optim = 'adam'
    L1Loss = True
    gcn_true = True
    buildA_true = True
    gcn_depth = 2
    dropout = .3
    node_dim = 40
    dilation_exponential = 2
    in_dim = 1
    seq_out_len = 1
    clip = 5
    propalpha = .05
    tanhalpha = 3
    num_split = 1
    step_size = 100

    def __post_init__(self):
        file_name = self.data_path.replace(".csv", "")
        if 'out/' in file_name:
            file_name = file_name.split('out/')[1]
        model_calculation = 25 if self.week < 50 else (50 if 50 <= self.week < 75 else
                                                       (75 if 75 <= self.week < 90 else 90))
        self.model_path = self.init_path(f'model/{self.seq_in_len}/{file_name}/weeks', str(model_calculation), '.pt')
        self.figs_path = self.create_path(
            f'figs/{Constants.PREDICTION_TIME}/{self.seq_in_len}/{file_name}/week_{self.week}')
        self.prediction_path = self.init_prediction_path(file_name)
        self.data = DataLoaderS(self.data_path, self.week, self.num_weeks, self.training_split, self.validation_split,
                                self.device, self.horizon, self.seq_in_len, self.normalize, self.run_for_prediction)

        self.criterion = nn.L1Loss(size_average=False).to(self.device) if self.L1Loss else nn.MSELoss(
            size_average=False).to(self.device)
        self.evaluateL1 = nn.L1Loss(size_average=False).to(self.device)
        self.evaluateL2 = nn.MSELoss(size_average=False).to(self.device)
        self.summary_writer = self.create_summary_writer(file_name)
        self.model = self.create_model()
        self.optimizer = self.create_optimizer()
        self.print_num_of_model_parameters()

    def __del__(self):
        # noinspection PyBroadException
        try:
            self.summary_writer.close()
        except Exception as e:
            print("Failed to close summary writer.", e)

    def init_prediction_path(self, file_name):
        return self.init_path(f'prediction/{self.seq_in_len}', f'{file_name}/weeks/{self.week + 1}')

    @staticmethod
    def init_path(base_path, file_name, file_extension=''):
        os.makedirs(f'{base_path}/{os.path.dirname(file_name)}', exist_ok=True)
        return f'{base_path}/{file_name}{file_extension}'

    @staticmethod
    def create_path(path):
        os.makedirs(path, exist_ok=True)
        return path

    def create_summary_writer(self, file_name):
        path = self.create_path(f'runs/{Constants.PREDICTION_TIME}/{self.seq_in_len}/{file_name}/weeks/{self.week}')
        return SummaryWriter(log_dir=path)

    def create_model(self):
        return gtnet(self.gcn_true,
                     self.buildA_true,
                     self.gcn_depth,
                     self.num_nodes,
                     self.device,
                     dropout=self.dropout,
                     subgraph_size=self.subgraph_size,
                     node_dim=self.node_dim,
                     dilation_exponential=self.dilation_exponential,
                     conv_channels=self.conv_channels,
                     residual_channels=self.residual_channels,
                     skip_channels=self.skip_channels,
                     end_channels=self.end_channels,
                     seq_length=self.seq_in_len,
                     in_dim=self.in_dim,
                     out_dim=self.seq_out_len,
                     layers=self.layers,
                     propalpha=self.propalpha,
                     tanhalpha=self.tanhalpha,
                     layer_norm_affline=False).to(self.device)

    def create_optimizer(self):
        return Optim(self.model.parameters(), self.optim, self.lr, self.clip, lr_decay=self.weight_decay)

    def print_num_of_model_parameters(self):
        print('The recpetive field size is', self.model.receptive_field)
        nParams = sum([p.nelement() for p in self.model.parameters()])
        print('Number of model parameters is', nParams, flush=True)

    def run(self):
        self.run_train_only()
        return self.evaluate_the_best_model()

    def run_train_only(self):
        max_value_for_best_validation_loss = 10000000
        best_validation_mape_for_best_loss = 10000000

        # At any point you can hit Ctrl + C to break out of training early.
        try:
            print('begin training')
            early_stop_patience = 100
            early_stop_counter = 0
            for epoch in tqdm(range(1, self.epochs + 1), desc='train_single_step.py > epoch'):
                epoch_start_time = time.time()
                train_loss = self.train()
                val_loss, val_rae, val_corr, val_mape = evaluate(self.data, self.data.valid[0], self.data.valid[1],
                                                                 self.model, self.evaluateL2, self.evaluateL1,
                                                                 self.batch_size)
                self.add_to_summary(epoch, train_loss, val_loss, val_rae, val_corr, val_mape)
                print(
                    '| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.4f} | valid rse {:5.4f} | valid rae {:5.4f} | valid corr {:5.4f} | valid mape {:5.4f}'
                    .format(epoch, (time.time() - epoch_start_time), train_loss, val_loss, val_rae, val_corr, val_mape),
                    flush=True)
                # Save the model if the validation loss is the best we've seen so far.

                if val_loss < max_value_for_best_validation_loss:
                    with open(self.model_path, 'wb') as f:
                        torch.save(self.model, f)
                    max_value_for_best_validation_loss = val_loss
                    best_validation_mape_for_best_loss = val_mape
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                    if early_stop_counter == early_stop_patience:
                        print(f"Early stopping after {early_stop_patience} epochs without improvement.")
                        break
                if epoch % 5 == 0:
                    test_loss, test_rae, test_corr, test_mape = evaluate(self.data, self.data.test[0],
                                                                         self.data.test[1], self.model, self.evaluateL2,
                                                                         self.evaluateL1, self.batch_size)
                    self.add_test_metrics_to_summary(epoch, test_loss, test_rae, test_corr, test_mape)
                    print("test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f} | test mape {:5.4f}".format(
                        test_loss, test_rae, test_corr, test_mape),
                          flush=True)

        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')

        print(f'Validation mape for best loss: {best_validation_mape_for_best_loss}', flush=True)
        return best_validation_mape_for_best_loss

    def add_to_summary(self, epoch, train_loss, val_loss, val_rae, val_corr, val_mape):
        # noinspection PyBroadException
        try:
            self.summary_writer.add_scalar('train_loss', train_loss, epoch)
            self.summary_writer.add_scalar('val_loss', val_loss, epoch)
            self.summary_writer.add_scalar('val_rae', val_rae, epoch)
            self.summary_writer.add_scalar('val_corr', val_corr, epoch)
            self.summary_writer.add_scalar('val_mape', val_mape, epoch)
        except Exception as e:
            print("Failed to add to summary writer.", e)

    def add_test_metrics_to_summary(self, epoch, test_loss, test_rae, test_corr, test_mape):
        # noinspection PyBroadException
        try:
            self.summary_writer.add_scalar('test_loss', test_loss, epoch)
            self.summary_writer.add_scalar('test_rae', test_rae, epoch)
            self.summary_writer.add_scalar('test_corr', test_corr, epoch)
            self.summary_writer.add_scalar('test_mape', test_mape, epoch)
        except Exception as e:
            print("Failed to add to summary writer.", e)

    def train(self):
        X, Y = self.data.train[0], self.data.train[1]
        self.model.train()
        total_loss = 0
        n_samples = 0
        iter = 0
        for X, Y in self.data.get_batches(X, Y, self.batch_size, True):
            self.model.zero_grad()
            X = torch.unsqueeze(X, dim=1)
            X = X.transpose(2, 3)
            if iter % self.step_size == 0:
                perm = np.random.permutation(range(self.num_nodes))
            num_sub = int(self.num_nodes / self.num_split)

            for j in range(self.num_split):
                if j != self.num_split - 1:
                    id = perm[j * num_sub:(j + 1) * num_sub]
                else:
                    id = perm[j * num_sub:]
                id = torch.tensor(id).to(self.device)
                tx = X[:, :, id, :]
                ty = Y[:, id]
                output = self.model(tx, id)
                output = torch.squeeze(output)
                scale = self.data.scale.expand(output.size(0), self.data.num_cols)
                scale = scale[:, id]
                loss = self.criterion(output * scale, ty * scale)
                loss.backward()
                total_loss += loss.item()
                n_samples += (output.size(0) * self.data.num_cols)
                self.optimizer.step()

            if iter % 100 == 0:
                print('iter:{:3d} | loss: {:.3f}'.format(iter, loss.item() / (output.size(0) * self.data.num_cols)))
            iter += 1
        return total_loss / n_samples

    def evaluate_the_best_model(self):
        model = self.load_model()

        vtest_acc, vtest_rae, vtest_corr, vtest_mape = evaluate(self.data, self.data.valid[0], self.data.valid[1],
                                                                model, self.evaluateL2, self.evaluateL1,
                                                                self.batch_size)
        test_acc, test_rae, test_corr, test_mape = evaluate(self.data, self.data.test[0], self.data.test[1], model,
                                                            self.evaluateL2, self.evaluateL1, self.batch_size)
        print("final test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f} | test mape {:5.4f}".format(
            test_acc, test_rae, test_corr, test_mape))
        return vtest_acc, vtest_rae, vtest_corr, vtest_mape, test_acc, test_rae, test_corr, test_mape

    def predict_with_the_best_model(self):
        model = self.load_model()

        model.eval()
        prediction = None

        for X in self.data.get_inputs(self.data.all[0], self.batch_size):
            X = torch.unsqueeze(X, dim=1).transpose(2, 3)
            with torch.no_grad():
                output = model(X)
            output = torch.squeeze(output)
            if len(output.shape) == 1:
                output = output.unsqueeze(dim=0)
            if prediction is None:
                prediction = output
            else:
                prediction = torch.cat((prediction, output))

        prediction = self.denormalize(prediction).cpu().numpy()
        all_denormalized = self.denormalize(self.data.all[1]).cpu().numpy()

        for j in range(prediction.shape[1]):
            df = pd.DataFrame()
            df['actual'] = all_denormalized[:, j]
            df['predicted'] = prediction[:, j]
            plt.figure(figsize=(80, 40))
            df.plot()
            plt.savefig(f'{self.figs_path}/prediction_of_asset_{j}.png', dpi=300)
            plt.close()

        print(f"Saving prediction results: {self.prediction_path}")
        np.save(self.prediction_path, prediction)

    def load_model(self):
        print(f"Loading model from: {self.model_path}")
        with open(self.model_path, 'rb') as f:
            model = torch.load(f, map_location=torch.device(self.device))
        return model

    def denormalize(self, data):
        data = data.to(self.device) * self.data.scale
        return data


def evaluate(data, X, Y, model, evaluateL2, evaluateL1, batch_size):
    model.eval()
    total_loss = 0
    total_loss_l1 = 0
    total_mape = 0
    n_samples = 0
    predict = None
    test = None

    for X, Y in data.get_batches(X, Y, batch_size, False):
        X = torch.unsqueeze(X, dim=1)
        X = X.transpose(2, 3)
        with torch.no_grad():
            output = model(X)
        output = torch.squeeze(output)
        if len(output.shape) == 1:
            output = output.unsqueeze(dim=0)
        if predict is None:
            predict = output
            test = Y
        else:
            predict = torch.cat((predict, output))
            test = torch.cat((test, Y))

        scale = data.scale.expand(output.size(0), data.num_cols)
        total_loss += evaluateL2(output * scale, Y * scale).item()
        total_loss_l1 += evaluateL1(output * scale, Y * scale).item()
        total_mape += torch.abs(((output * scale) - (Y * scale)) / (Y * scale)).sum().item()
        n_samples += (output.size(0) * data.num_cols)

    rse = math.sqrt(total_loss / n_samples) / data.rse
    rae = (total_loss_l1 / n_samples) / data.rae
    mape = total_mape / n_samples

    predict = predict.data.cpu().numpy()
    Ytest = test.data.cpu().numpy()
    sigma_p = (predict).std(axis=0)
    sigma_g = (Ytest).std(axis=0)
    mean_p = predict.mean(axis=0)
    mean_g = Ytest.mean(axis=0)
    index = (sigma_g != 0)
    correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
    correlation = (correlation[index]).mean()
    return rse, rae, correlation, mape


def print_results(runs, acc, corr, mape, rae, vacc, vcorr, vmape, vrae):
    print('\n\n')
    print(f'{runs} runs average')
    print('\n\n')
    print("valid\trse\trae\tcorr\tmape")
    print_means(vacc, vcorr, vmape, vrae)
    print_stds(vacc, vcorr, vmape, vrae)
    print('\n\n')
    print("test\trse\trae\tcorr\tmape")
    print_means(acc, corr, mape, rae)
    print_stds(acc, corr, mape, rae)


def print_means(acc, corr, mape, rae):
    print("mean\t{:5.4f}\t{:5.4f}\t{:5.4f}\t{:5.4f}".format(np.mean(acc), np.mean(rae), np.mean(corr), np.mean(mape)))


def print_stds(acc, corr, mape, rae):
    print("std\t{:5.4f}\t{:5.4f}\t{:5.4f}\t{:5.4f}".format(np.std(acc), np.std(rae), np.std(corr), np.std(mape)))
