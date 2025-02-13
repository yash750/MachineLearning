import numpy as np


class SimpleRegression:
    def __init__(self, X, Y):
        self.X = np.array(X)
        self.Y = np.array(Y)
        self.m = None
        self.c = None

    def fit(self):
        mean_x = np.mean(self.X)
        mean_y = np.mean(self.Y)

        num = 0
        den = 0

        for i in range(len(self.X)):
            num += (self.X[i] - mean_x) * (self.Y[i] - mean_y)
            den += (self.X[i] - mean_x) ** 2

        self.m = num / den
        self.c = mean_y - (self.m * mean_x)

    def train(self):
        self.fit()

    def predict(self, x_value):
        if self.m is None or self.c is None:
            raise ValueError("Model is not trained. Call train() before predicting.")
        return self.m * x_value + self.c


class MultipleLinearRegression:
    def __init__(self, X, Y):
        self.X = np.array(X)
        self.Y = np.array(Y)
        self.weights = None

    def fit(self):
        X_bias = np.concatenate((np.ones((self.X.shape[0], 1)), self.X), axis=1)
        self.weights = np.dot(
            np.linalg.inv(np.dot(X_bias.T, X_bias)), np.dot(X_bias.T, self.Y)
        )

    def train(self):
        self.fit()

    def predict(self, X_test):
        X_test_bias = np.concatenate((np.ones((X_test.shape[0], 1)), X_test), axis=1)
        return np.dot(X_test_bias, self.weights)


class PolynomialRegression:
    def __init__(self, degree):
        self.degree = degree
        self.weights = None

    def polynomial_features(self, X):
        X_poly = np.ones((X.shape[0], 1))
        for d in range(1, self.degree + 1):
            X_poly = np.hstack((X_poly, X**d))
        return X_poly

    def fit(self, X, y):
        X_poly = self.polynomial_features(X)
        self.weights = np.dot(
            np.linalg.inv(np.dot(X_poly.T, X_poly)), np.dot(X_poly.T, y)
        )

    def predict(self, X):
        X_poly = self.polynomial_features(X)
        return np.dot(X_poly, self.weights)
