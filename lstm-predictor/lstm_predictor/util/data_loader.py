import numpy as np

from lstm_predictor.core import config


class DataLoader:

    def __init__(self, data_np, seq_len, training_ratio, validation_ratio):
        self._seq_length = seq_len
        self._horizon = config.horizon
        self._dat = data_np
        self._num_rows, self._num_cols = self._dat.shape
        self._split(int(training_ratio * self._num_rows), int((training_ratio + validation_ratio) * self._num_rows))
        self.last_day = self._dat[-seq_len:].reshape((1, seq_len, -1)).astype(np.float32)

    def _split(self, train_end_index, valid_end_index):
        training_set_indices = range(self._seq_length + self._horizon - 1, train_end_index)
        validation_set_indices = range(train_end_index, valid_end_index)
        test_set = range(valid_end_index, self._num_rows)
        self.training = self._batchify(training_set_indices)
        self.validation = self._batchify(validation_set_indices)
        self.training_and_validation = {
            'X': np.concatenate((self.training['X'], self.validation['X'])),
            'y': np.concatenate((self.training['y'], self.validation['y']))
        }
        self.test = self._batchify(test_set)

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
