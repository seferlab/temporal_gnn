import itertools
import random

from tqdm import tqdm

from pipeline import run_command

dataset_name = '__fx_daily_simple_returns_week_104'
horizon = 5

search_space = {
    'epoch': [50],
    'window_size': [12, 15, 20, 24],
    'lr': [1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
    'multi_layer': [2, 3, 5, 8, 13],
    'exponential_decay_step': [2, 3, 5, 8, 13],
    'decay_rate': [.2, .3, .5, .8, .9],
    'dropout_rate': [0, .25, .5, .75]
}

all_combinations = list(itertools.product(*search_space.values()))
random_combinations = random.sample(all_combinations, 100)
for combination in tqdm(random_combinations, desc='hyperparameter_finder.py > combination'):
    param_set = zip(search_space.keys(), combination)
    hyperparameters_text = ''
    for key, value in param_set:
        hyperparameters_text += f'--{key} {value} '
    run_command(f'python main.py --device cuda --dataset {dataset_name} --horizon {horizon} {hyperparameters_text}')
