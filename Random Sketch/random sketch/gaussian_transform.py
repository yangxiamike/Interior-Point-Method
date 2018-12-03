"""
Fast Johnson-Lindenstrauss Transform (FJLT)
http://people.inf.ethz.ch/kgabriel/software.html
"""

from __future__ import division, print_function
import numpy as np
import os
from numpy.linalg import norm
from clarkson_transform import clarkson_woodruff_transform


class RandomizedGaussianTransform(object):
    def __init__(self, k, rows=True):
        self.rows = rows
        self.k = k

    def fit(self, X, y=None):
        assert (y is None) or self.rows, 'If over features, cant use y'
        if not self.rows:
            X = X.T

        self.n = X.shape[0]
        self.G = np.random.randn(self.k, self.n) * np.sqrt(1 / self.k)


    def transform_1d(self, x):
        return np.dot(self.G, x)

    def inverse_transform_1d(self, a):
        return np.dot(self.G.T, a)

    def transform(self, X, y=None):
        assert (y is None) or self.rows, 'If over features, cant use y'

        if y is not None:
            X = np.c_[X, y]
            Ab = np.dot(self.G, X)
            return Ab[:, 0:-1], Ab[:, -1]
        else:
            if self.rows:
                return np.dot(self.G, X)
            else:
                return np.dot(self.G, X.T).T

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)


def test():
    n, d, k = 10000, 100, 9000
    X = np.random.randn(n, d)
    y = np.random.randn(n)

    """
    # Over rows, full dataset
    rt = RandomizedGaussianTransform(k)
    print(rt.fit_transform(X))
    print(rt.fit_transform(X, y))
    """

    # Over cols, full dataset
    rt = RandomizedGaussianTransform(k, rows=False)
    A = rt.fit_transform(X)
    #F = clarkson_woodruff_transform(X, k)
    #print(A)

    test_C = []
    test_G = []

    """
    # 1d transforms, inverse
    rt = RandomizedGaussianTransform(k)
    x = X[0, :]
    rt.fit(x)
    a = rt.transform_1d(x)
    """

    #print(norm(X,2), norm(A,2),norm(F,2))

    #print(rt.inverse_transform_1d(a))


if __name__ == '__main__':
    test()