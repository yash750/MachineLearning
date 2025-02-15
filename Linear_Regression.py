import numpy as np


class SimpleRegression:
    def __init__(self, X, Y):
        """
        Simple Linear Regression using Ordinary Least Squares (OLS) method.
        :X: Independent variable (1D array)
        :Y: Dependent variable (1D array)
        """
        self.X = np.array(X)
        self.Y = np.array(Y)
        self.m = None  # Slope
        self.c = None  # Intercept

    def fit(self):
        """
        Compute the best fit line using the OLS formula.
        """
        mean_x = np.mean(self.X)
        mean_y = np.mean(self.Y)

        num = 0
        den = 0

        for i in range(len(self.X)):
            num += (self.X[i] - mean_x) * (self.Y[i] - mean_y)
            den += (self.X[i] - mean_x) ** 2

        self.m = num / den
        self.c = mean_y - (self.m * mean_x)

    def predict(self, x_value):
        """
        Predict the dependent variable for a given independent variable.
        :x_value: Input value(s) for prediction
        :return: Predicted value(s)
        """
        if self.m is None or self.c is None:
            raise ValueError("Model is not trained. Call fit() before predicting.")
        return self.m * x_value + self.c


class MultipleLinearRegression:
    def __init__(self, X, Y):
        """
        Multiple Linear Regression using OLS method.
        :X: Feature matrix (2D array)
        :Y: Target values (1D array)
        """
        self.X = np.array(X)
        self.Y = np.array(Y)
        self.weights = None

    def fit(self):
        """
        Compute the optimal weights using the Normal Equation.
        """
        X_bias = np.concatenate((np.ones((self.X.shape[0], 1)), self.X), axis=1)
        self.weights = np.dot(
            np.linalg.inv(np.dot(X_bias.T, X_bias)), np.dot(X_bias.T, self.Y)
        )

    def predict(self, X_test):
        """
        Predict values using the trained model.
        :X_test: Input features for prediction (2D array)
        :return: Predicted target values (1D array)
        """
        X_test_bias = np.concatenate((np.ones((X_test.shape[0], 1)), X_test), axis=1)
        return np.dot(X_test_bias, self.weights)


class PolynomialRegression:
    def __init__(self, degree):
        """
        Polynomial Regression using OLS method.
        :degree: Degree of the polynomial.
        """
        self.degree = degree
        self.weights = None

    def polynomial_features(self, X):
        """
        Expand the feature matrix to include polynomial terms.
        :X: Original feature matrix (1D or 2D array)
        :return: Expanded feature matrix (2D array)
        """
        X_poly = np.ones((X.shape[0], 1))
        for d in range(1, self.degree + 1):
            X_poly = np.hstack((X_poly, X**d))
        return X_poly

    def fit(self, X, y):
        """
        Train the polynomial regression model using OLS.
        :X: Input feature(s) (1D or 2D array)
        :y: Target values (1D array)
        """
        X_poly = self.polynomial_features(X)
        self.weights = np.dot(
            np.linalg.inv(np.dot(X_poly.T, X_poly)), np.dot(X_poly.T, y)
        )

    def predict(self, X):
        """
        Predict values using the trained polynomial regression model.
        :X: Input feature(s) for prediction (1D or 2D array)
        :return: Predicted values (1D array)
        """
        X_poly = self.polynomial_features(X)
        return np.dot(X_poly, self.weights)
