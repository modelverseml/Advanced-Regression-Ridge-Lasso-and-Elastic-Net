"""Ridge regression (L2-regularised), closed-form.

Adds an L2 penalty on the coefficients to the OLS loss:

    L(beta) = ||y - X beta||^2  +  alpha * ||beta||^2

Setting the gradient to zero gives a closed-form solution:

    beta = (X^T X + alpha * I')^(-1) X^T y

where I' is the identity matrix with a zero in the (0, 0) slot so the intercept
is NOT penalised (this matches scikit-learn's Ridge).

Ridge shrinks every coefficient toward zero but never exactly to zero, so it
does not do feature selection. It is most useful when features are highly
correlated, where it spreads coefficient weight more evenly across them.
"""

import numpy as np
import pandas as pd


class RidgeRegression:
    def __init__(self, X_train, y_train, alpha=1.0):
        self.X_train = X_train
        self.y_train = y_train
        self.alpha = alpha
        self.feature_names = list(X_train.columns)
        self.coefficients = None

    def build_model(self):
        """Fit by solving the L2-regularised normal equation."""
        X = self.X_train.to_numpy(dtype=float)
        y = self.y_train.to_numpy(dtype=float)

        # Prepend a column of 1s so coefficient 0 is the intercept.
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        self.feature_names = ["Intercept"] + self.feature_names

        # Regularisation matrix: do NOT penalise the intercept term.
        n_features = X.shape[1]
        reg_matrix = self.alpha * np.eye(n_features)
        reg_matrix[0, 0] = 0.0

        # np.linalg.solve is more numerically stable than inverting explicitly.
        self.coefficients = np.linalg.solve(X.T @ X + reg_matrix, X.T @ y)

    def predict(self, X):
        """Predict y for new inputs X (same feature order as training)."""
        X = np.asarray(X, dtype=float)
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        return X @ self.coefficients

    def get_parameters(self):
        """Return a DataFrame of (feature, coefficient) pairs."""
        return pd.DataFrame({
            "Feature": self.feature_names,
            "Coefficient": self.coefficients.round(3),
        })
