import numpy as np


class PolynomialRegressionGD:
    def __init__(self, degree=2, learning_rate=0.01, iterations=1000):
        """
        Initialize the Polynomial Regression model using Gradient Descent.
        :degree: Degree of the polynomial.
        :learning_rate: Step size for weight updates.
        :iterations: Number of iterations for gradient descent.
        """
        self.degree = degree
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None

    def polynomial_features(self, X):
        """
        Generate polynomial features up to the specified degree.
        : X: Input features (1D or 2D array)
        :return: Transformed feature matrix with polynomial terms.
        """
        X_poly = np.ones((X.shape[0], 1))  # Start with bias term (ones column)
        for d in range(1, self.degree + 1):
            X_poly = np.hstack((X_poly, X**d))
        return X_poly

    def fit(self, X, Y):
        """
        Train the polynomial regression model using Gradient Descent.
        :X: Input features (1D or 2D array)
        :Y: Target values (1D array)
        """
        X = np.array(X).reshape(-1, 1)  # Ensure X is 2D
        Y = np.array(Y)
        X_poly = self.polynomial_features(X)
        n_samples, n_features = X_poly.shape

        self.weights = np.zeros(n_features)  # Initialize weights

        # Gradient Descent loop
        for _ in range(self.iterations):
            y_pred = X_poly @ self.weights  # Compute predictions
            error = y_pred - Y  # Compute error
            gradient = (2 / n_samples) * (X_poly.T @ error)  # Compute gradient
            self.weights -= self.learning_rate * gradient  # Update weights

    def predict(self, X_test):
        """
        Predict target values for new data.
        :X_test: Input features for prediction (1D or 2D array)
        :return: Predicted target values (1D array)
        """
        X_test = np.array(X_test).reshape(-1, 1)  # Ensure X_test is 2D
        X_test_poly = self.polynomial_features(X_test)
        return X_test_poly @ self.weights  # Compute predictions
