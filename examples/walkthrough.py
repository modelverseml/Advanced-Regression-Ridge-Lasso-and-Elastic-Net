"""End-to-end walkthrough for Ridge / Lasso / Elastic Net.

Reproduces the old notebook as a runnable script:

    1. build a synthetic regression dataset (some features are pure noise)
    2. check each from-scratch model against its scikit-learn equivalent
    3. compare the coefficients the three penalties produce
    4. sweep alpha to find the test-MSE-minimising value for each model
    5. (optional) save the diagnostic plots

Run from the repository root:

    python examples/walkthrough.py            # numbers only
    python examples/walkthrough.py --plot     # also save plots to Images/
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.linear_model import ElasticNet as SkElasticNet
from sklearn.linear_model import Lasso as SkLasso
from sklearn.linear_model import Ridge as SkRidge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from _helpers import add_repo_root_to_path

add_repo_root_to_path()

from advanced_regression import (  # noqa: E402  (import after the sys.path tweak)
    ElasticNetRegression,
    LassoRegression,
    RidgeRegression,
)

RANDOM_STATE = 42
IMAGES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Images"
)


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def build_dataset():
    """Synthetic data: 10 features, only 5 informative; standardised."""
    X, y, true_coef = make_regression(
        n_samples=200,
        n_features=10,
        n_informative=5,
        noise=10.0,
        coef=True,
        random_state=RANDOM_STATE,
    )
    feature_names = [f"f{i}" for i in range(X.shape[1])]
    X = pd.DataFrame(X, columns=feature_names)
    y = pd.Series(y, name="target")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # L1/L2 penalties are scale-sensitive, so standardise the features. Then a
    # single alpha is comparable across them.
    scaler = StandardScaler()
    X_train = pd.DataFrame(
        scaler.fit_transform(X_train), columns=feature_names, index=X_train.index
    )
    X_test = pd.DataFrame(
        scaler.transform(X_test), columns=feature_names, index=X_test.index
    )

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"True coefficients (pre-scaling): {true_coef.round(2)}")
    return X_train, X_test, y_train, y_test, feature_names


def compare(manual_model, sk_model, X_train, y_train, X_test, y_test, name):
    """Fit both models and show that the from-scratch one matches sklearn."""
    manual_model.build_model()
    sk_model.fit(X_train, y_train)

    manual_pred = manual_model.predict(X_test)
    sk_pred = sk_model.predict(X_test)

    print(f"=== {name} ===")
    print(f"Manual  MSE: {mean_squared_error(y_test, manual_pred):.4f}   "
          f"R2: {r2_score(y_test, manual_pred):.4f}")
    print(f"sklearn MSE: {mean_squared_error(y_test, sk_pred):.4f}   "
          f"R2: {r2_score(y_test, sk_pred):.4f}")
    coef_diff = np.max(np.abs(manual_model.coefficients[1:] - sk_model.coef_))
    intercept_diff = abs(manual_model.coefficients[0] - sk_model.intercept_)
    print(f"Max |manual - sklearn| coefficient diff: {coef_diff:.6f}")
    print(f"Intercept diff: {intercept_diff:.6f}\n")


def fit_all(X_train, X_test, y_train, y_test):
    """Fit and validate all three models at alpha=1.0."""
    section("Ridge (L2) vs scikit-learn")
    ridge = RidgeRegression(X_train, y_train, alpha=1.0)
    compare(ridge, SkRidge(alpha=1.0), X_train, y_train, X_test, y_test, "Ridge")

    section("Lasso (L1) vs scikit-learn")
    lasso = LassoRegression(X_train, y_train, alpha=1.0, max_iter=5000, tol=1e-6)
    compare(lasso, SkLasso(alpha=1.0, max_iter=5000, tol=1e-6),
            X_train, y_train, X_test, y_test, "Lasso")

    section("Elastic Net (L1 + L2) vs scikit-learn")
    en = ElasticNetRegression(X_train, y_train, alpha=1.0, l1_ratio=0.5,
                              max_iter=5000, tol=1e-6)
    compare(en, SkElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=5000, tol=1e-6),
            X_train, y_train, X_test, y_test, "Elastic Net")

    return ridge, lasso, en


def compare_coefficients(ridge, lasso, en, feature_names):
    """Line up the coefficients each penalty produces (note Lasso's zeros)."""
    section("Coefficients by penalty type")
    coef_df = pd.DataFrame({
        "Feature": feature_names,
        "Ridge": ridge.coefficients[1:].round(3),
        "Lasso": lasso.coefficients[1:].round(3),
        "ElasticNet": en.coefficients[1:].round(3),
    })
    print(coef_df.to_string(index=False))
    n_zero = int((np.abs(lasso.coefficients[1:]) < 1e-8).sum())
    print(f"\nLasso set {n_zero}/{len(feature_names)} coefficients to exactly 0.")
    return coef_df


def sweep_alpha(X_train, X_test, y_train, y_test):
    """Fit every model across a range of alpha and record test MSE + coef paths."""
    section("Alpha sweep: test MSE and best alpha")
    alphas = np.logspace(-2, 2, 25)
    ridge_coefs, lasso_coefs = [], []
    ridge_mse, lasso_mse, en_mse = [], [], []

    for a in alphas:
        rm = RidgeRegression(X_train, y_train, alpha=a)
        rm.build_model()
        lm = LassoRegression(X_train, y_train, alpha=a, max_iter=5000, tol=1e-6)
        lm.build_model()
        em = ElasticNetRegression(X_train, y_train, alpha=a, l1_ratio=0.5,
                                  max_iter=5000, tol=1e-6)
        em.build_model()

        ridge_coefs.append(rm.coefficients[1:])
        lasso_coefs.append(lm.coefficients[1:])
        ridge_mse.append(mean_squared_error(y_test, rm.predict(X_test)))
        lasso_mse.append(mean_squared_error(y_test, lm.predict(X_test)))
        en_mse.append(mean_squared_error(y_test, em.predict(X_test)))

    print(f"  Ridge      : best alpha = {alphas[np.argmin(ridge_mse)]:.3f}  "
          f"MSE = {min(ridge_mse):.3f}")
    print(f"  Lasso      : best alpha = {alphas[np.argmin(lasso_mse)]:.3f}  "
          f"MSE = {min(lasso_mse):.3f}")
    print(f"  Elastic Net: best alpha = {alphas[np.argmin(en_mse)]:.3f}  "
          f"MSE = {min(en_mse):.3f}")

    return {
        "alphas": alphas,
        "ridge_coefs": np.array(ridge_coefs),
        "lasso_coefs": np.array(lasso_coefs),
        "ridge_mse": ridge_mse,
        "lasso_mse": lasso_mse,
        "en_mse": en_mse,
    }


def save_plots(coef_df, sweep, feature_names):
    """Save the coefficient bar chart, regularisation paths and MSE curve."""
    import matplotlib

    matplotlib.use("Agg")  # file output only, no display needed
    import matplotlib.pyplot as plt

    os.makedirs(IMAGES_DIR, exist_ok=True)

    # 1. Coefficient by feature, per penalty.
    ax = coef_df.set_index("Feature").plot.bar(figsize=(10, 4))
    ax.axhline(0, color="grey", linewidth=0.7)
    ax.set_ylabel("Coefficient")
    ax.set_title("Coefficient by feature, by penalty type")
    plt.tight_layout()
    path1 = os.path.join(IMAGES_DIR, "coefficients_by_penalty.png")
    plt.savefig(path1, dpi=150)
    plt.close()

    # 2. Regularisation paths: how each coefficient shrinks as alpha grows.
    alphas = sweep["alphas"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for i, name in enumerate(feature_names):
        axes[0].plot(alphas, sweep["ridge_coefs"][:, i], label=name)
        axes[1].plot(alphas, sweep["lasso_coefs"][:, i], label=name)
    for ax, title in zip(axes, ["Ridge path", "Lasso path"]):
        ax.set_xscale("log")
        ax.axhline(0, color="grey", linewidth=0.7)
        ax.set_xlabel("alpha (log scale)")
        ax.set_title(title)
    axes[0].set_ylabel("Coefficient")
    axes[1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    path2 = os.path.join(IMAGES_DIR, "regularization_paths.png")
    plt.savefig(path2, dpi=150)
    plt.close()

    # 3. Test MSE vs alpha.
    plt.figure(figsize=(8, 4))
    plt.plot(alphas, sweep["ridge_mse"], label="Ridge", marker="o")
    plt.plot(alphas, sweep["lasso_mse"], label="Lasso", marker="s")
    plt.plot(alphas, sweep["en_mse"], label="Elastic Net", marker="^")
    plt.xscale("log")
    plt.xlabel("alpha (log scale)")
    plt.ylabel("Test MSE")
    plt.title("Test-set MSE vs regularisation strength")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path3 = os.path.join(IMAGES_DIR, "test_mse_vs_alpha.png")
    plt.savefig(path3, dpi=150)
    plt.close()

    print(f"\nSaved plots to {IMAGES_DIR}:")
    for p in (path1, path2, path3):
        print(f"  {os.path.basename(p)}")


def main():
    parser = argparse.ArgumentParser(description="Run the regularised-regression walkthrough.")
    parser.add_argument("--plot", action="store_true", help="save diagnostic plots to Images/")
    args = parser.parse_args()

    np.random.seed(RANDOM_STATE)

    section("Dataset")
    X_train, X_test, y_train, y_test, feature_names = build_dataset()

    ridge, lasso, en = fit_all(X_train, X_test, y_train, y_test)
    coef_df = compare_coefficients(ridge, lasso, en, feature_names)
    sweep = sweep_alpha(X_train, X_test, y_train, y_test)

    if args.plot:
        save_plots(coef_df, sweep, feature_names)

    print()


if __name__ == "__main__":
    main()
