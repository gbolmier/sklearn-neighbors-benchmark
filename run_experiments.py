from sklearn_neighbors_benchmark.datasets import Datasets
from sklearn_neighbors_benchmark.utils import ParameterGen
from sklearn_neighbors_benchmark.utils import run_experiments
from sklearn_neighbors_benchmark.utils import save_results


# Load and preprocess data
datasets = Datasets().datasets  # dict -> {'dataset_1': (X_train, X_test), ...}

# Define a grid and a generator of parameters to explore
# Each entry is mandatory and its value must be an iterable
param_grid = {
    'dataset': datasets.keys(),
    'n_samples': [10_000, 50_000, 100_000],  # At construction time
    'n_features': list(range(5, 31, 5)),
    'algorithm': ['brute', 'kd_tree', 'ball_tree'],
    'n_neighbors': [10, 100],
}

param_gen = ParameterGen(param_grid)

# Run experiments and save the results
results = run_experiments(datasets, param_gen)
save_results(results)
