# :bar_chart: Scikit-learn nearest neighbors benchmark

This repository contains scripts and notebooks for benchmarking scikit-learn [nearest neighbors algorithms](https://scikit-learn.org/dev/modules/neighbors.html#nearest-neighbor-algorithms) (**brute force**, **k-d tree** and **ball tree**). This work is related to sklearn neighbors heuristic issue [#8213](https://github.com/scikit-learn/scikit-learn/issues/8213) , and being addressed in the pull request [#17148](https://github.com/scikit-learn/scikit-learn/pull/17148).

Scikit-learn `0.22.2.post1` version is used.

## Usage

`sklearn_neighbors_benchmark` directory contains utilities to run experiments and save the results to `results.csv`.

`run_experiments.py` allows you to run a set of experiments, saving the results in `results.csv`. Please note that duplicated experiments will be run, but not saved.

`jakevdp_benchmark` directory contains Jake VanderPlas [benchmark](https://jakevdp.github.io/blog/2013/04/29/benchmarking-nearest-neighbor-searches-in-python/) with a modified version computing brute force instead of estimating it.

## Protocol

Datasets used:
- `covertype`: consists of forest cartographic variables, shape `(110_393, 54)`, version 1 from [OpenML](https://www.openml.org/d/1596).
- `creditcard`: consists of credit cards transactions PCA transformed variables, shape `(284_807, 29)`, version 1 from [OpenML](https://www.openml.org/d/1597).
- `mnist_pca`: consists of the 100 first MNIST PCA transformed variables (explaining 70% of the variance), shape `(70_000, 100)`, version 1 from [OpenML](https://www.openml.org/d/554).
- `synthetic_low_intrinsic_dim`: consists of standard normal sampled variables divided by 1,000 (except 5 of them), shape `(110_000, 100)`.
- `synthetic_standard_normal`: consists of standard normal sampled variables, shape `(110_000, 100)`.

Parameters studied:
- `algorithm`
- `dataset`
- `n_samples` at construction time
- `n_features`
- `n_neighbors`
- `n_jobs`, number of parallel jobs to run for neighbors search
- `n_threads`, number of threads that can be used in OpenMP/BLAS thread pools

Results saved:
- `time_construction_mean`
- `time_construction_std`
- `time_querying_mean`
- `time_querying_std`

Miscellaneous:
- In order to get robust results, the number of query points is fixed to 10,000
- Each experiment is repeated 3 times — with random feature sampling for real world datasets
- Real world datasets are standardized
- `metric` is fixed to `euclidean`

## Results analysis

- [sklearn_nn_heuristic.ipynb](https://nbviewer.jupyter.org/github/gbolmier/sklearn-neighbors-benchmark/blob/master/sklearn_nn_heuristic.ipynb)
- [sklearn_nn_heuristic_proposal.ipynb](https://nbviewer.jupyter.org/github/gbolmier/sklearn-neighbors-benchmark/blob/master/sklearn_nn_heuristic_proposal.ipynb)

## Resources

- [Scikit-learn user guide - nearest neighbor algorithms](https://scikit-learn.org/dev/modules/neighbors.html#nearest-neighbor-algorithms)
- [Jake Vanderplas benchmark](https://jakevdp.github.io/blog/2013/04/29/benchmarking-nearest-neighbor-searches-in-python/) 04/2013
- [Erik Bernhardsson benchmark](https://github.com/erikbern/ann-benchmarks)
