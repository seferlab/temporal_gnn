from datetime import datetime

from util import format_time_as_YYYYMMddHHmm


class Constants:
    PREDICTION_TIME = format_time_as_YYYYMMddHHmm(datetime.now())
