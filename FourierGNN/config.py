import torch

from data.data_loader import (
    Dataset_Dhfm,
    Dataset_ECG,
    Dataset_Solar,
    Dataset_Wiki,
    DatasetFinancial,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_information = {
    "traffic": {
        "root_path": "data/traffic.npy",
        "type": "0"
    },
    "ECG": {
        "root_path": "data/ECG_data.csv",
        "type": "1"
    },
    "COVID": {
        "root_path": "data/covid.csv",
        "type": "1"
    },
    "electricity": {
        "root_path": "data/electricity.csv",
        "type": "1"
    },
    "solar": {
        "root_path": "/data/solar",
        "type": "1"
    },
    "metr": {
        "root_path": "data/metr.csv",
        "type": "1"
    },
    "wiki": {
        "root_path": "data/wiki.csv",
        "type": "1"
    },
    "crypto": {
        "root_path": "../ticker-collector/out/crypto/daily_20_2189_marked.csv",
        "type": "1",
    },
    "fx": {
        "root_path": "../ticker-collector/out/fx/daily_10_4506_marked.csv",
        "type": "1",
    },
}

data_dict = {
    "ECG": Dataset_ECG,
    "COVID": Dataset_ECG,
    "traffic": Dataset_Dhfm,
    "solar": Dataset_Solar,
    "wiki": Dataset_Wiki,
    "electricity": Dataset_ECG,
    "metr": Dataset_ECG,
    "crypto": DatasetFinancial,
    "fx": DatasetFinancial,
}
