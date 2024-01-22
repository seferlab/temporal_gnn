# TODO Integrate this file into main.py later.

import itertools

import numpy as np
from scipy.stats import levene, permutation_test, ttest_ind


def statistic(x, y, axis):
    return np.mean(x, axis=axis) - np.mean(y, axis=axis)


lstm_scores = {
    'name': 'LSTM',
    'Cryptocurrency': {
        'MAPE': [0.113092, 0.109872, 0.109632, 0.106940, 0.112002, 0.113703, 0.107926, 0.109201, 0.106156, 0.109668],
        'MAE': [0.106717, 0.103861, 0.103621, 0.101415, 0.105607, 0.107486, 0.102267, 0.103282, 0.100790, 0.103691],
        'RMSE': [0.154527, 0.152756, 0.152704, 0.150883, 0.154379, 0.155501, 0.151845, 0.152315, 0.150509, 0.152577]
    },
    'Forex': {
        'MAPE': [0.015425, 0.018386, 0.016254, 0.016021, 0.018472, 0.016153, 0.016249, 0.015639, 0.017104, 0.020973],
        'MAE': [0.015320, 0.018164, 0.016087, 0.015880, 0.018244, 0.016032, 0.016079, 0.015504, 0.016917, 0.020763],
        'RMSE': [0.153210, 0.153982, 0.153406, 0.153332, 0.154020, 0.153318, 0.153368, 0.153235, 0.153525, 0.154581]
    }
}

mtgnn_scores = {
    'name': 'MTGNN',
    'Cryptocurrency': {
        'MAPE': [0.0625, 0.0619, 0.0603, 0.0619, 0.0600, 0.0608, 0.0659, 0.0632, 0.0594, 0.0615],
        'MAE': [0.0611, 0.0606, 0.0595, 0.0614, 0.0589, 0.0596, 0.0642, 0.0635, 0.0585, 0.0604],
        'RMSE': [0.0839, 0.0840, 0.0836, 0.0854, 0.0823, 0.0834, 0.0873, 0.0874, 0.0827, 0.0841]
    },
    'Forex': {
        'MAPE': [0.0134, 0.0136, 0.0128, 0.0121, 0.0121, 0.0123, 0.0152, 0.0124, 0.0147, 0.0123],
        'MAE': [0.0135, 0.0138, 0.0129, 0.0122, 0.0122, 0.0124, 0.0154, 0.0125, 0.0148, 0.0125],
        'RMSE': [0.0213, 0.0208, 0.0199, 0.0193, 0.0190, 0.0191, 0.0267, 0.0192, 0.0221, 0.0194]
    }
}

stemgnn_scores = {
    'name': 'StemGNN',
    'Cryptocurrency': {
        'MAPE': [0.07924, 0.07985, 0.08078, 0.08060, 0.08277, 0.07893, 0.07923, 0.07866, 0.07909, 0.08584],
        'MAE': [0.07458, 0.07488, 0.07619, 0.07583, 0.07753, 0.07418, 0.07483, 0.07430, 0.07499, 0.08067],
        'RMSE': [0.10682, 0.10701, 0.10878, 0.10800, 0.10925, 0.10623, 0.10721, 0.10644, 0.10711, 0.11562]
    },
    'Forex': {
        'MAPE': [0.01865, 0.01757, 0.01718, 0.01809, 0.02041, 0.01703, 0.02192, 0.01683, 0.01947, 0.01788],
        'MAE': [0.02094, 0.01994, 0.01937, 0.02035, 0.02261, 0.01934, 0.02401, 0.01912, 0.02167, 0.02005],
        'RMSE': [0.22261, 0.22281, 0.22204, 0.22272, 0.22308, 0.22221, 0.22378, 0.22214, 0.22222, 0.22217]
    }
}


def apply_tests(dataset, combinations, significance_test_fn):
    print(f"# {dataset} Results")
    print()
    for metric in ['MAPE', 'MAE', 'RMSE']:
        print(f"## {metric} Results")
        for a, b in combinations:
            test_result = significance_test_fn((a[dataset][metric], b[dataset][metric]))
            print(
                f'{a["name"]} vs {b["name"]}: P-Value: {test_result.pvalue} and T-Stat: {test_result.statistic}{" => Significant" if test_result.pvalue < 0.05 else ""}'
            )
        print()
    print()


def check_variances(data):
    stat, p_value = levene(data[0], data[1])
    print(f"Levene's test: statistic={stat}, p-value={p_value}")
    if p_value < 0.05:
        print("Warning: Variances are significantly different.")
    else:
        print("Variances are not significantly different.")


def permutation_test_algorithm(data):
    print(data[0], 'vs', data[1])
    return permutation_test(data,
                            statistic,
                            vectorized=True,
                            n_resamples=100000,
                            alternative='less',
                            permutation_type='independent')


def t_test_ind_algorithm(data):
    return ttest_ind(data[0], data[1], alternative='two-sided')


result_combinations = list(itertools.combinations((mtgnn_scores, stemgnn_scores, lstm_scores), 2))
apply_tests('Cryptocurrency', result_combinations, t_test_ind_algorithm)
apply_tests('Forex', result_combinations, t_test_ind_algorithm)
