from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Data:
    data: np.ndarray
    label: str

    def __repr__(self):
        return f'Data(label={self.label}, data={self.data})'

    def __getitem__(self, key):
        if isinstance(key, slice):
            return Data(self.data[key], self.label)
        else:
            return Data(self.data[key:key + 1], self.label)


@dataclass
class DataPath:
    path: str
    label: str
    data_type: Optional[str] = None

    def __repr__(self):
        return f'Path(label={self.label}, path={self.path})'


@dataclass
class PortfolioResult:
    data: np.ndarray
    label: str
    long_short_combination: tuple
    predicted_result_data: Optional[np.ndarray] = None
    asset_selection_accuracy: Optional[float] = None

    def __repr__(self):
        if self.predicted_result_data is not None:
            return f'PortfolioResult(label={self.label}, change={self.data[0]}->{self.data[-1]}, predicted_change={self.data[0]}->{self.predicted_result_data[-1]}, long_short_combination={self.long_short_combination})'
        else:
            return f'PortfolioResult(label={self.label}, change={self.data[0]}->{self.data[-1]}, long_short_combination={self.long_short_combination})'

    def final_portfolio_value(self):
        return self.data[-1]


@dataclass
class PortfolioValues:
    actual: np.ndarray
    prediction: Optional[np.ndarray] = None

    def __repr__(self):
        return f'PortfolioValues(actual={self.actual}, prediction={self.prediction})'
