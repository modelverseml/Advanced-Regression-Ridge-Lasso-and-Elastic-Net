"""Advanced (regularised) regression toolkit.

From-scratch NumPy implementations of the three standard regularised linear
models: Ridge (L2), Lasso (L1) and Elastic Net (L1 + L2). Each one matches the
matching scikit-learn estimator to numerical precision and shares the same
small interface: build_model(), predict(X) and get_parameters().

See README.md for the theory and examples/walkthrough.py for a runnable demo.
"""

from .elastic_net import ElasticNetRegression
from .lasso import LassoRegression
from .ridge import RidgeRegression

__all__ = [
    "RidgeRegression",
    "LassoRegression",
    "ElasticNetRegression",
]

__version__ = "1.0.0"
