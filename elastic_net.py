"""
Elastic Net Regression — Coordinate Descent
-------------------------------------------
Combines L1 (Lasso) and L2 (Ridge) penalties with a mixing parameter
`l1_ratio` in [0, 1]:

    L(beta) = (1/2n) * ||y - X beta||^2
              +       alpha *      l1_ratio   * ||beta||_1
              + 0.5 * alpha * (1 - l1_ratio)  * ||beta||^2

This matches the convention used by scikit-learn's ElasticNet. The two
extreme cases are:

    l1_ratio = 1  ->  pure Lasso  (L1 only)
    l1_ratio = 0  ->  pure Ridge  (L2 only — closed-form is faster, see ridge.py)

Coordinate descent solves each per-coefficient sub-problem analytically.
For centred features:

    rho_j  = (1/n) x_j^T (y - X beta + x_j beta_j)
    z_j    = (1/n) x_j^T x_j
    beta_j = soft(rho_j, alpha * l1_ratio) / ( z_j + alpha * (1 - l1_ratio) )

The L2 term shows up as an extra positive constant in the denominator —
it shrinks coefficients smoothly. The L1 term shows up as the soft-
threshold — it can zero coefficients out exactly. Together they get
Lasso's sparsity plus Ridge's stability across correlated features.

The intercept is recovered after the loop by centring X and y, then
back-solving `intercept = mean(y) - mean(X) . beta`.
"""

import numpy as np
import pandas as pd


def _soft_threshold(x, threshold):
    """Coordinate-wise soft thresholding — the proximal operator of the L1 norm."""

    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)


class ElasticNetRegression:

    def __init__(self, X_train, y_train, alpha=1.0, l1_ratio=0.5, max_iter=1000, tol=1e-4):

        self.X_train = X_train
        self.y_train = y_train
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.feature_names = list(X_train.columns)
        self.coefficients = None  # includes the intercept at index 0

    def build_model(self):
        """Fit with coordinate descent on the centred problem."""

        X = self.X_train.to_numpy(dtype=float)
        y = self.y_train.to_numpy(dtype=float)

        x_mean = X.mean(axis=0)
        y_mean = y.mean()
        X_c = X - x_mean
        y_c = y - y_mean

        n_samples, n_features = X_c.shape
        beta = np.zeros(n_features)
        z = (X_c ** 2).sum(axis=0) / n_samples

        # Decompose the overall penalty into its L1 and L2 contributions once.
        l1 = self.alpha * self.l1_ratio
        l2 = self.alpha * (1 - self.l1_ratio)

        residual = y_c.copy()

        for _ in range(self.max_iter):
            beta_old = beta.copy()

            for j in range(n_features):
                residual += X_c[:, j] * beta[j]
                rho_j = X_c[:, j] @ residual / n_samples

                denom = z[j] + l2
                if denom == 0:
                    beta[j] = 0.0
                else:
                    beta[j] = _soft_threshold(rho_j, l1) / denom

                residual -= X_c[:, j] * beta[j]

            if np.max(np.abs(beta - beta_old)) < self.tol:
                break

        intercept = y_mean - x_mean @ beta
        self.coefficients = np.concatenate(([intercept], beta))
        self.feature_names = ["Intercept"] + self.feature_names

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
