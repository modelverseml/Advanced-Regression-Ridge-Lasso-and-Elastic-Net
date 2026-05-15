"""
Lasso Regression (L1-Regularized) — Coordinate Descent
------------------------------------------------------
Minimises the L1-penalised least-squares loss:

    L(beta) = (1/2n) * ||y - X beta||^2  +  alpha * ||beta||_1

The L1 term is non-differentiable at zero, so there is no closed-form
solution. We use **coordinate descent**: cycle through one coefficient at a
time, holding the others fixed. Each per-coordinate sub-problem has an
analytic solution via the **soft-thresholding operator**:

    soft(x, t) = sign(x) * max(|x| - t, 0)

For features centred to zero mean, the update for coefficient j is:

    r_j    = y - X beta + x_j beta_j           # partial residual ignoring feature j
    rho_j  = (1/n) x_j^T r_j
    z_j    = (1/n) x_j^T x_j                   # precomputed once
    beta_j = soft(rho_j, alpha) / z_j

The intercept is recovered after the loop by centring X and y first and
then back-solving `intercept = mean(y) - mean(X) . beta`. This avoids a
special-case "don't shrink the intercept" rule inside the inner loop.

The L1 penalty drives some coefficients exactly to zero — Lasso doubles
as a feature-selection method.
"""

import numpy as np
import pandas as pd


def _soft_threshold(x, threshold):
    """Coordinate-wise soft thresholding — the proximal operator of the L1 norm."""

    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)


class LassoRegression:

    def __init__(self, X_train, y_train, alpha=1.0, max_iter=1000, tol=1e-4):

        self.X_train = X_train
        self.y_train = y_train
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.feature_names = list(X_train.columns)
        self.coefficients = None  # includes the intercept at index 0

    def build_model(self):
        """Fit with coordinate descent on the centred problem."""

        X = self.X_train.to_numpy(dtype=float)
        y = self.y_train.to_numpy(dtype=float)

        # Centring lets us drop the intercept from the inner optimisation;
        # we add it back at the end as a closed-form correction.
        x_mean = X.mean(axis=0)
        y_mean = y.mean()
        X_c = X - x_mean
        y_c = y - y_mean

        n_samples, n_features = X_c.shape
        beta = np.zeros(n_features)

        # Precompute (1/n) * ||x_j||^2 for each feature — used inside every update.
        z = (X_c ** 2).sum(axis=0) / n_samples

        # Maintain the full residual y_c - X_c beta so each update only touches one column.
        residual = y_c.copy()

        for _ in range(self.max_iter):
            beta_old = beta.copy()

            for j in range(n_features):
                # Add feature j back into the residual so we get the partial residual r_j.
                residual += X_c[:, j] * beta[j]
                rho_j = X_c[:, j] @ residual / n_samples

                if z[j] == 0:
                    beta[j] = 0.0
                else:
                    beta[j] = _soft_threshold(rho_j, self.alpha) / z[j]

                # Subtract the new contribution of feature j to restore the full residual.
                residual -= X_c[:, j] * beta[j]

            # Stop when the largest coefficient change in a full pass is below tol.
            if np.max(np.abs(beta - beta_old)) < self.tol:
                break

        # Recover the intercept from the centring shift.
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
