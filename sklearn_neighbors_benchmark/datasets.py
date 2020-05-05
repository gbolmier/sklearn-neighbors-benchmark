import numpy as np

from tqdm import tqdm

from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


__all__ = ['Datasets']


class Datasets:
    names = [
        'covertype',
        'creditcard',
        'mnist_pca',
        'synthetic_low_intrinsic_dim',
        'synthetic_standard_normal',
    ]

    def __init__(self, select='all', random_state=32):
        if select != 'all':
            self.names = select

        self.random_state = random_state
        self.datasets = self.load_and_preprocess_datasets()

    def load_and_preprocess_datasets(self):
        print('Loading and preprocessing datasets...')
        datasets = {
            name: getattr(self, 'get_' + name)()
            for name in tqdm(self.names)
        }
        return datasets

    def standardize(self, X_train, X_test):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        return X_train, X_test

    def get_covertype(self):
        X, y = fetch_openml('covertype', version=1, return_X_y=True)

        X_train, X_test, _, _ = train_test_split(
            X, y, train_size=100_000, test_size=10_000,
            random_state=self.random_state
        )

        # standardize data for meaningful distance calculations
        X_train, X_test = self.standardize(X_train, X_test)

        return X_train, X_test

    def get_creditcard(self):
        X, y = fetch_openml('creditcard', version=1, return_X_y=True)

        X_train, X_test, _, _ = train_test_split(
            X, y, train_size=100_000, test_size=10_000,
            random_state=self.random_state
        )

        # standardize data for meaningful distance calculations
        X_train, X_test = self.standardize(X_train, X_test)

        return X_train, X_test

    def get_mnist_pca(self):
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

        X_train, X_test, _, _ = train_test_split(
            X, y, train_size=50_000, test_size=10_000,
            random_state=self.random_state
        )

        # standardize data before PCA
        X_train, X_test = self.standardize(X_train, X_test)

        # apply PCA
        pca = PCA(n_components=100, random_state=self.random_state)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

        # standardize data again for meaningful distance calculations
        X_train, X_test = self.standardize(X_train, X_test)

        return X_train, X_test

    def get_synthetic_low_intrinsic_dim(self, intrinsic_dim=5):
        np.random.seed(self.random_state)
        X_signal = np.random.randn(110_000, intrinsic_dim)
        X_noise = np.random.randn(110_000, 100 - intrinsic_dim) / 1_000
        X = np.hstack((X_signal, X_noise))
        X_train, X_test = X[:100_000], X[-10_000:]
        return X_train, X_test

    def get_synthetic_standard_normal(self):
        np.random.seed(self.random_state)
        X_train = np.random.randn(100_000, 100)
        X_test = np.random.randn(10_000, 100)
        return X_train, X_test
