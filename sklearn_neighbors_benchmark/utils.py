import numpy as np
import os
import pandas as pd
import time

from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import NearestNeighbors

from tqdm import tqdm


__all__ = [
    'ParameterGen',
    'run_experiments',
    'save_results',
]


class ParameterGen(ParameterGrid):
    
    def __init__(self, param_grid):
        params_given = sorted(list(param_grid.keys()))
        params_expected = [
            'algorithm',
            'dataset',
            'n_features',
            'n_neighbors',
            'n_samples'
        ]
        if params_given != params_expected:
            raise ValueError(
                f'Parameters given ({params_given}) != '
                f'parameters expected ({params_expected})'
            )
            
        super().__init__(param_grid)


def run_experiments(datasets, param_gen):
    results = []
    param_gen = list(param_gen)  # tqdm needs the length
    print('Running experiments...')
    for params in tqdm(param_gen):
        X_train, X_test = datasets[params['dataset']]

        # Skip iteration when the dataset doesn't have enough samples/features
        n_samples, n_features = X_train.shape
        if n_samples < params['n_samples'] or n_features < params['n_features']:
            continue

        result = _run_single_experiment(X_train, X_test, params, repeat=3)
        results.append({**params, **result})
    
    return results


def _run_single_experiment(X_train, X_test, params, repeat=3):
    # Alias parameters for readability
    n_samples, n_features = params['n_samples'], params['n_features']
    algorithm, n_neighbors = params['algorithm'], params['n_neighbors']
    
    X_train = X_train[:n_samples, :n_features]
    X_test = X_test[:, :n_features]
    
    model = NearestNeighbors(n_neighbors, algorithm=algorithm, n_jobs=1)
    
    times_construction, times_querying = [], []
    for _ in range(repeat):
        t0 = time.time()
        model.fit(X_train)
        t1 = time.time()
        model.kneighbors(X_test, return_distance=False)
        t2 = time.time()
        
        times_construction.append(t1 - t0)
        times_querying.append(t2 - t1)
    
    return {
        'time_construction_mean': np.mean(times_construction),
        'time_construction_std': np.std(times_construction),
        'time_querying_mean': np.mean(times_querying),
        'time_querying_std': np.std(times_querying),
    }


def save_results(results, filepath='results.csv'):
    print('Saving results...')
    if os.path.exists(filepath):
        params = [
            param_name
            for param_name in results[0].keys()
            if not param_name.startswith('time')
        ]
        results_new = pd.DataFrame(results)
        results_old = pd.read_csv(filepath)
        results_all = results_old.merge(results_new, how='outer', on=params)
    else:
        results_all = pd.DataFrame(results)

    results_all.to_csv(filepath, index=False)
    print('Done!')
