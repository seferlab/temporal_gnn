import datetime

import torch

from lstm_predictor.util import format_time_as_YYYYMMddHHmm

PREDICTION_TIME = format_time_as_YYYYMMddHHmm(datetime.datetime.now())
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
