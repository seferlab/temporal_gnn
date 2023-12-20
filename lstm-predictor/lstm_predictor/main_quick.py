from lstm_predictor.core import config


def main():
    config.num_weeks_to_train = 1
    config.num_epochs_to_run = 1
    config.hpo_max_evals = 1
    config.predictions_base_path = 'test_prediction'
    config.log_active = True

    from lstm_predictor.main import main as go
    go()


if __name__ == '__main__':
    main()
