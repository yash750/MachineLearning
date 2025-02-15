import numpy as np


class LinearRegressionGD:
    def __init__(self, learning_rate=0.001, iterations=1000):
        """
        Initialize the Linear Regression model using Gradient Descent.
        :learning_rate: Step size for weight updates.
        :iterations: Number of iterations for gradient descent.
        """
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None  # Model parameters

    def fit(self, X, Y):
        """
        Train the model using Gradient Descent.
        :X: Input features (2D array)
        :Y: Target values (1D array)
        """
        X = np.array(X)
        Y = np.array(Y)
        n_samples, n_features = X.shape

        # Adding bias term (column of ones) to X
        X_bias = np.c_[np.ones(n_samples), X]

        # Initialize weights to zeros
        self.weights = np.zeros(n_features + 1)

        # Gradient Descent loop
        for _ in range(self.iterations):
            y_pred = X_bias @ self.weights  # Predicted values
            error = y_pred - Y  # Compute error
            gradient = (2 / n_samples) * (X_bias.T @ error)  # Compute gradient
            self.weights -= self.learning_rate * gradient  # Update weights

    def predict(self, X_test):
        """
        Predict target values for new data.
        :X_test: Input features for prediction (2D array)
        :return: Predicted target values (1D array)
        """
        X_test = np.array(X_test)
        n_samples = X_test.shape[0]

        # Adding bias term to test data
        X_test_bias = np.c_[np.ones(n_samples), X_test]

        return X_test_bias @ self.weights  # Compute predictions
