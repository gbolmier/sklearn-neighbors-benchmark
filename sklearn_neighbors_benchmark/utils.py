import numpy as np
import os
import pandas as pd
import time

from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import NearestNeighbors

from threadpoolctl import threadpool_limits
from tqdm import tqdm


__all__ = [
    'ParameterGen',
    'run_experiments',
    'save_results',
]


class ParameterGen(ParameterGrid):
    params_expected = [
        'algorithm',    # algorithm used to compute the nearest neighbors
        'dataset',      # dataset name
        'n_features',   # number of features
        'n_jobs',       # number of parallel jobs to run for neighbors search
        'n_neighbors',  # number of querying neighbors
        'n_samples',    # number of samples at construction time
        'n_threads'     # number of threads that can be used in OpenMP/BLAS thread pools
    ]

    def __init__(self, param_grid):
        params_given = list(param_grid.keys())
        self.check_params(params_given)
        super().__init__(param_grid)

    def check_params(self, params_given):
        params_given = sorted(params_given)
        if params_given != self.params_expected:
            raise ValueError(
                f'Parameters given ({params_given}) != '
                f'parameters expected ({params_expected})'
            )


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


def _feature_subsampling(X_train, X_test, n_features):
    mask = np.random.choice(X_train.shape[1], size=n_features, replace=False)
    return X_train[:, mask], X_test[:, mask]


def _run_single_experiment(X_train, X_test, params, repeat=3):
    # Alias parameters for readability
    n_samples, n_features = params['n_samples'], params['n_features']
    algorithm, n_neighbors = params['algorithm'], params['n_neighbors']
    n_jobs, n_threads = params['n_jobs'], params['n_threads']
    dataset = params['dataset']

    X_train = X_train[:n_samples]
    times_construction, times_querying = [], []
    model = NearestNeighbors(n_neighbors, algorithm=algorithm, n_jobs=n_jobs)

    with threadpool_limits(limits=n_threads):
        for _ in range(repeat):

            if dataset.startswith('synthetic_'):  # no need to subsample iid random variables
                X_train_ = X_train[:, :n_features]
                X_test_ = X_test[:, :n_features]
            else:
                X_train_, X_test_ = _feature_subsampling(X_train, X_test,
                                                         n_features)

            t0 = time.time()
            model.fit(X_train_)
            t1 = time.time()
            model.kneighbors(X_test_, return_distance=False)
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
        results_all = results_old.merge(results_new, how='outer')
        results_all.drop_duplicates(params, inplace=True)
    else:
        results_all = pd.DataFrame(results)

    results_all.to_csv(filepath, index=False)
    print('Done!')
