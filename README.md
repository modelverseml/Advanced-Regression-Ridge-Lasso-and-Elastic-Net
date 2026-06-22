# Advanced Regression — Ridge, Lasso & Elastic Net

This repo implements the three standard **regularized** linear regression models
from scratch with NumPy — Ridge, Lasso, and Elastic Net — and checks each one
against scikit-learn (they match to numerical precision). The sections below
explain what regularization is, why we need it, and how the three models differ.

## The problem: why plain regression isn't enough

Ordinary least squares (OLS) regression fits a line by choosing coefficients
(the weights on each feature) that make the squared prediction error as small as
possible on the training data. That sounds ideal, but it runs into trouble in
two common situations:

1. **Too many features, or noisy ones.** OLS will happily use *every* feature to
   shave off a little training error, even features that are really just noise.
   It ends up fitting the random quirks of the training set instead of the real
   pattern. This is **overfitting**: great scores on training data, poor scores
   on new data.

2. **Correlated features (multicollinearity).** If two features carry almost the
   same information (say, "height in cm" and "height in inches"), OLS can't
   decide how to split the weight between them. It often lands on huge
   coefficients that cancel each other out — for example `+5000` on one and
   `-4980` on the other. These are unstable: a tiny change in the data flips them
   around, and the model becomes unreliable.

The common thread is **large, unstable coefficients**. They are the warning sign
that a model has fit the noise rather than the signal.

## The fix: regularization

**Regularization** means adding a penalty for large coefficients to the quantity
the model minimises. Instead of only "make the training error small", the model
now optimises:

```
total cost  =  prediction error  +  alpha × (penalty on coefficient sizes)
```

Because big coefficients now cost something, the model only keeps a coefficient
large if that feature genuinely pays for itself by reducing error a lot. Weak or
noisy features get pushed toward zero. The result is a simpler model that
generalises better to new data.

<p align="center"><img src="Images/regularization.webp" alt="regularization" width="50%"/></p>

The knob `alpha` (often written λ) controls how strong the penalty is:

- `alpha = 0` → no penalty, you are back to ordinary OLS.
- small `alpha` → light touch, coefficients barely shrink.
- large `alpha` → heavy shrinkage, coefficients are forced close to zero (push it
  too far and the model becomes too simple and *underfits*).

There are two standard ways to measure "coefficient size", and they behave
differently — this is the key idea behind the three models.

### L2 penalty (Ridge): sum of squared coefficients

L2 adds up the **squares** of the coefficients ($\sum \beta_j^2$). Squaring
punishes large coefficients much more than small ones, so L2 mainly attacks the
biggest weights and spreads them out. It shrinks every coefficient smoothly
toward zero but **never exactly to zero** — every feature stays in the model,
just with a smaller weight. This is exactly what tames the unstable, cancelling
coefficients caused by correlated features.

### L1 penalty (Lasso): sum of absolute coefficients

L1 adds up the **absolute values** of the coefficients ($\sum |\beta_j|$). The
absolute-value function has a sharp corner at zero, and that corner makes the
optimiser happy to park a coefficient at *exactly* zero rather than at some tiny
value. A coefficient of zero means the feature is dropped entirely — so Lasso
doesn't just shrink, it performs **automatic feature selection**, handing you a
smaller, simpler model.

### Elastic Net: a bit of both

Elastic Net adds the L1 and L2 penalties together, with a mixing knob
`l1_ratio`. It gets Lasso's ability to drop useless features *and* Ridge's
stability when features are correlated. It's the usual choice when you have many
correlated features but still want some selection — in that case pure Lasso
tends to arbitrarily pick one feature from a correlated group and zero the rest,
while Elastic Net handles the group more gracefully.

## The three models at a glance

| Model | Penalty | What it does to coefficients | Use it when |
|---|---|---|---|
| **Ridge** | L2 (sum of β²) | shrinks all of them, keeps all features | features are correlated; you want stability |
| **Lasso** | L1 (sum of \|β\|) | shrinks them and sets weak ones to exactly 0 | you want a smaller, automatically selected feature set |
| **Elastic Net** | L1 + L2 | a mix of both, controlled by `l1_ratio` | many correlated features and you still want some selection |

<p align="center"><img src="Images/lasso_ridge.webp" alt="Lasso vs Ridge" width="55%"/></p>

## Quick glossary

- **Coefficient (β):** the weight the model puts on a feature. A bigger
  coefficient means that feature has more influence on the prediction.
- **Intercept (β₀):** the baseline prediction when all features are zero. It is
  never penalised — we only shrink the feature weights, not the baseline.
- **Overfitting:** the model learns the noise in the training data, so it looks
  great in training but does poorly on new data.
- **Underfitting:** the opposite — the model is too simple (often from too much
  regularization) and misses the real pattern.
- **Regularization:** adding a penalty on coefficient size so the model prefers
  smaller, simpler weights.
- **`alpha` (λ):** penalty strength. `0` = plain OLS; larger = more shrinkage.
- **`l1_ratio`:** for Elastic Net, the share of the penalty that is L1 vs L2.
  `1` = pure Lasso, `0` = pure Ridge.
- **Multicollinearity:** when features are highly correlated and carry
  overlapping information; it makes OLS coefficients large and unstable.
- **Bias–variance trade-off:** more regularization → simpler, steadier model
  (more bias, less variance); less regularization → flexible model that can
  overfit (less bias, more variance). The best `alpha` sits in between.

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

**Ridge** has a clean formula. Its penalty (squared coefficients) is smooth, so
we can set the derivative to zero and solve directly:

```
β = (XᵀX + αI)⁻¹ Xᵀy
```

That single line gives the answer in one step — no iteration needed. The `αI`
term is what adds the L2 penalty (with the intercept left out so it isn't
shrunk). See [`ridge.py`](advanced_regression/ridge.py).

**Lasso and Elastic Net** can't be solved that way, because the absolute-value
penalty has that sharp corner at zero and so isn't differentiable there. Instead
they use **coordinate descent**, which is simpler than it sounds:

> Fix every coefficient except one, and find the best value for that single
> coefficient (an easy 1-D problem). Move to the next coefficient and repeat.
> Keep cycling through all coefficients until they stop changing.

The one-coefficient update uses the **soft-thresholding** rule:

```
soft(x, t) = sign(x) × max(|x| − t, 0)
```

In words: shrink the value `x` toward zero by an amount `t`, and if `x` was
already smaller than `t`, snap it all the way to **0**. That "snap to zero" step
is exactly what lets Lasso drop features. (Ridge's smooth penalty never snaps,
which is why it only shrinks and never eliminates.) See
[`lasso.py`](advanced_regression/lasso.py) and
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
