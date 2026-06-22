# Advanced Regression — Ridge, Lasso & Elastic Net

Plain linear regression (OLS) can struggle when you have many features or
features that are correlated: it gives big, unstable coefficients and overfits
the training data. **Regularized** regression fixes this by adding a penalty
that keeps the coefficients small.

This repo implements the three standard regularized models from scratch with
NumPy — no `sklearn.linear_model` — and checks each one against scikit-learn
(they match to numerical precision).

## A few terms first

If some of these are new, here is the short version:

- **Coefficient (β):** the weight a model gives a feature. Bigger weight = the
  feature moves the prediction more.
- **Overfitting:** the model memorises noise in the training data and does
  worse on new data. Large coefficients are a common symptom.
- **Regularization:** add a penalty on the size of the coefficients to the
  thing we minimise, so the model prefers smaller, simpler weights.
- **Penalty strength `alpha`:** how hard we push coefficients toward zero.
  `alpha = 0` is ordinary regression; larger `alpha` = more shrinkage.
- **L1 penalty (sum of |β|):** can push some coefficients to *exactly* zero,
  which drops those features. This is **feature selection**.
- **L2 penalty (sum of β²):** shrinks every coefficient toward zero but never
  exactly to zero.
- **Multicollinearity:** two or more features carry the same information
  (highly correlated). It makes OLS coefficients jump around.

## The three models at a glance

| Model | Penalty | What it does | Use it when |
|---|---|---|---|
| **Ridge** | L2 | shrinks all coefficients, keeps them all | features are correlated; you want stability |
| **Lasso** | L1 | shrinks and zeroes out weak features | you want a smaller, automatically selected set of features |
| **Elastic Net** | L1 + L2 | a mix of both, set by `l1_ratio` | many correlated features and you still want some selection |

<p align="center"><img src="Images/regularization.webp" alt="regularization" width="50%"/></p>

## Repository structure

```
Advanced-Regression/
├── README.md
├── requirements.txt
├── main.py                       # runs the full walkthrough
├── advanced_regression/          # the package (importable)
│   ├── __init__.py
│   ├── ridge.py                  # Ridge (L2), closed-form
│   ├── lasso.py                  # Lasso (L1), coordinate descent
│   └── elastic_net.py            # Elastic Net (L1 + L2), coordinate descent
├── examples/
│   └── walkthrough.py            # end-to-end demo, validated against sklearn
└── Images/
```

Every model shares the same small interface: `build_model()` to fit,
`predict(X)`, and `get_parameters()` returning a `(feature, coefficient)` table.

## Getting started

```bash
git clone https://github.com/modelverseml/Advanced-Regression.git
cd Advanced-Regression
pip install -r requirements.txt

# Run the full walkthrough (data -> 3 models -> compare vs sklearn -> alpha sweep)
python main.py

# Same, plus save the diagnostic plots into Images/
python main.py --plot
```

## Using the models

```python
from advanced_regression import RidgeRegression, LassoRegression, ElasticNetRegression

# Ridge (L2) — has a closed-form solution
model = RidgeRegression(X_train, y_train, alpha=1.0)
model.build_model()
y_pred = model.predict(X_test)
model.get_parameters()        # (feature, coefficient) table

# Lasso (L1) — many coefficients come out exactly 0
model = LassoRegression(X_train, y_train, alpha=1.0, max_iter=5000, tol=1e-6)
model.build_model()

# Elastic Net (L1 + L2) — l1_ratio mixes the two penalties
model = ElasticNetRegression(X_train, y_train, alpha=1.0, l1_ratio=0.5)
model.build_model()
```

`l1_ratio` runs from 0 to 1: `l1_ratio = 1` is pure Lasso, `l1_ratio = 0` is
pure Ridge (for that case use `RidgeRegression` — its closed form is faster).

**Scale your features first.** L1 and L2 penalise the raw size of each
coefficient, so a single `alpha` is only fair when the features are on the same
scale. Standardise (e.g. `sklearn.preprocessing.StandardScaler`) before fitting.
The walkthrough does this.

## The math

OLS picks the coefficients that minimise the sum of squared errors (RSS):

$$
\min_{\beta}\ \sum_{i=1}^n\left( y_i - \beta_0 - \sum_{j=1}^p \beta_j X_{ij} \right)^2
$$

Each regularized model adds a penalty term on top, scaled by `alpha` (often
written λ):

**Ridge (L2):** add the sum of squared coefficients.

$$
\min_{\beta}\ \text{RSS} + \alpha \sum_{j=1}^p \beta_j^2
$$

**Lasso (L1):** add the sum of absolute coefficients. The sharp corner of
`|β|` at zero is what lets it set coefficients to exactly zero.

$$
\min_{\beta}\ \text{RSS} + \alpha \sum_{j=1}^p |\beta_j|
$$

**Elastic Net:** use both, with `l1_ratio` (written as $r$ below) deciding the balance.

$$
\min_{\beta}\ \text{RSS} + \alpha \left[ r \sum_{j=1}^p |\beta_j| + \frac{1 - r}{2} \sum_{j=1}^p \beta_j^2 \right]
$$

where $r$ is `l1_ratio` in the code: $r = 1$ is pure Lasso, $r = 0$ is pure Ridge.

### How they are solved

- **Ridge** is differentiable everywhere, so there is a closed-form formula:
  `β = (XᵀX + αI)⁻¹ Xᵀy` (the intercept is left unpenalised). See
  [`ridge.py`](advanced_regression/ridge.py).
- **Lasso and Elastic Net** have the non-differentiable `|β|` term, so there is
  no formula. They use **coordinate descent**: update one coefficient at a
  time, each step using the *soft-thresholding* rule
  `soft(x, t) = sign(x)·max(|x| − t, 0)`, which is exactly what shrinks small
  coefficients to zero. See [`lasso.py`](advanced_regression/lasso.py) and
  [`elastic_net.py`](advanced_regression/elastic_net.py).

## What the walkthrough shows

Running `python main.py` on synthetic data (10 features, only 5 of them
actually useful) demonstrates the expected behaviour:

- All three from-scratch models match scikit-learn to ~6 decimal places.
- Lasso sets the 5 useless features to exactly 0; Ridge keeps all 10 but small;
  Elastic Net does a mix.
- The `alpha` sweep shows the bias–variance trade-off: too small overfits, too
  large underfits, and there is a sweet spot in between.

With `--plot`, three figures are saved to `Images/`: coefficients per penalty,
the regularization paths (coefficients vs `alpha`), and test MSE vs `alpha`.

> In a real project, pick `alpha` with k-fold cross-validation rather than a
> single train/test split.
