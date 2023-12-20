from lstm_predictor.core import config


# noinspection PyPep8Naming
def format_time_as_YYYYMMddHHmm(time):
    return time.isoformat(sep='/', timespec='minutes').replace('-', '').replace(':', '').replace('/', '')


def log(week, message):
    if config.log_active:
        print(f'Week: {week} | {message}')
