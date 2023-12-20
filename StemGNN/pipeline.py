import os
import sys


def run_command(command):
    print('Running command: ' + command, flush=True)
    if os.system(command) != 0:
        sys.exit("Error executing command: " + command)


if __name__ == '__main__':
    # configs
    dataset = 'crypto_daily_marked'
    horizon = 7
    num_weeks = 104
    num_weeks_needed = 104

    # @formatter:off
    run_command(f'python dataset_preparer.py --dataset {dataset} --horizon {horizon} --num_weeks {num_weeks} --num_weeks_needed {num_weeks_needed}')
    run_command(f'python runner.py --dataset {dataset} --horizon {horizon} --num_weeks {num_weeks_needed}')
    run_command('rm dataset/*_week_*.csv')
    # @formatter:on
