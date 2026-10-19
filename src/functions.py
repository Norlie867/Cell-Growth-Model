# =======================================================================
# FUNCTIONS
# Define reusable helper functions in this file.
# =======================================================================

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # No display needed: figures are saved to files
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import (GridSearchCV, cross_val_predict,
                                     cross_validate, train_test_split)
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from src.constants import (COLORS, CV_FOLDS, DATASET_PATH, EXTREME_FRACTION,
                           EXTREME_GRID_POINTS, FEATURES, FIG_HEIGHT, FIG_WIDTH,
                           FIG_WIDTH_WIDE, FIGURES_DIR, FONTSIZE_LEGEND,
                           FONTSIZE_TICKS, FONTSIZE_XY_LABEL, KNN_NEIGHBORS,
                           KNN_WEIGHTS, LC_SAMPLE_SIZES, LC_SEED, LINESTYLES,
                           N_GRID_POINTS, POLY_DEGREES, RANDOM_STATE,
                           RELIABILITY_K, RELIABILITY_PERCENTILE, TARGET,
                           TEST_SIZE)

# By default, matplotlib makes math in italics, but we want our units in regular.
plt.rcParams["mathtext.default"] = "regular"
STR_D1 = r"$d^{-1}$"  # Will show a proper superscript in the figures.
SCORING = "neg_root_mean_squared_error"


# ---------------------------------------------------------------------
# Data loading and preparation
# ---------------------------------------------------------------------
def load_data():
    """Load the dataset and check that it is complete."""
    df = pd.read_csv(DATASET_PATH)
    missing = [c for c in [TARGET] + FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")
    if df.isna().any().any():
        raise ValueError("The dataset contains missing values.")
    return df


def summarize_data(df):
    """Descriptive statistics (with a count of zeros) and correlation matrix."""
    stats = df.describe().T
    stats["n zeros"] = (df == 0).sum()
    return stats.round(3), df.corr().round(3)


def split_data(df):
    """Convert to numpy and hold out a test set that is never used for tuning."""
    x = df.loc[:, FEATURES].to_numpy()
    y = df.loc[:, TARGET].to_numpy()
    return train_test_split(x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------
def build_model_poly():
    """Polynomial regression. The degree is left undefined so that
    GridSearchCV can try different values."""
    return make_pipeline(
        StandardScaler(),
        PolynomialFeatures(include_bias=False),
        LinearRegression()
    )


def build_model_knn():
    """k-nearest neighbors regression on standardized features."""
    return make_pipeline(
        StandardScaler(),
        KNeighborsRegressor()
    )


def run_grid_search(model, param_grid, x_train, y_train):
    """Grid search with cross-validation on the training set only."""
    model_gscv = GridSearchCV(model, param_grid=param_grid, cv=CV_FOLDS,
                              scoring=SCORING, return_train_score=True)
    model_gscv.fit(x_train, y_train)
    return model_gscv


def tune_models(x_train, y_train):
    """Tune every candidate model. Returns a dictionary of fitted GridSearchCV
    objects: linear, polynomial, and kNN with each weighting scheme."""
    searches = {
        "Linear": run_grid_search(
            build_model_poly(), dict(polynomialfeatures__degree=[1]),
            x_train, y_train),
        "Polynomial": run_grid_search(
            build_model_poly(), dict(polynomialfeatures__degree=POLY_DEGREES),
            x_train, y_train),
    }
    for weight in KNN_WEIGHTS:
        param_grid = dict(kneighborsregressor__n_neighbors=KNN_NEIGHBORS,
                          kneighborsregressor__weights=[weight])
        searches[f"kNN ({weight})"] = run_grid_search(
            build_model_knn(), param_grid, x_train, y_train)
    return searches


def predict_non_negative(model, x):
    """A growth rate cannot be negative, so negative predictions are set to 0."""
    return np.maximum(model.predict(x), 0)


def best_params_string(model_gscv):
    params = {k.split("__")[-1]: v for k, v in model_gscv.best_params_.items()}
    if "degree" in params:
        return f"degree = {params['degree']}"
    return f"k = {params['n_neighbors']}, weights = {params['weights']}"


def cv_scores(model_gscv):
    """Mean and SD of the validation RMSE of the best hyperparameters."""
    i = model_gscv.best_index_
    return (-model_gscv.cv_results_["mean_test_score"][i],
            model_gscv.cv_results_["std_test_score"][i])


def select_final_model(searches):
    """Choose between the best polynomial and the best kNN model.

    The model with the lowest validation RMSE is preferred, except that the
    polynomial is kept (it is simpler and easier to interpret) whenever its
    validation RMSE is within one standard deviation of the kNN model."""
    name_knn = min((n for n in searches if n.startswith("kNN")),
                   key=lambda n: cv_scores(searches[n])[0])
    rmse_poly, sd_poly = cv_scores(searches["Polynomial"])
    rmse_knn, _ = cv_scores(searches[name_knn])
    if rmse_knn < rmse_poly - sd_poly:
        return name_knn, "Polynomial"
    return "Polynomial", name_knn


# ---------------------------------------------------------------------
# Result tables
# ---------------------------------------------------------------------
def grid_results_table(model_gscv):
    """All grid-search results as a tidy DataFrame (positive RMSE)."""
    res = pd.DataFrame(model_gscv.cv_results_)
    out = res[[c for c in res.columns if c.startswith("param_")]].copy()
    out.columns = [c.split("__")[-1] for c in out.columns]
    out["Train RMSE"] = -res["mean_train_score"]
    out["Valid RMSE"] = -res["mean_test_score"]
    out["Valid RMSE SD"] = res["std_test_score"]
    out["Rank"] = res["rank_test_score"]
    return out.sort_values("Rank").reset_index(drop=True).round(4)


def polynomial_degree_table(x_train, y_train):
    """Train and validation RMSE and R^2 for each polynomial degree."""
    rows = []
    for degree in POLY_DEGREES:
        model = build_model_poly().set_params(polynomialfeatures__degree=degree)
        cv = cross_validate(model, x_train, y_train, cv=CV_FOLDS,
                            scoring=(SCORING, "r2"), return_train_score=True)
        rows.append({
            "Degree": degree,
            "Train RMSE": -np.mean(cv["train_" + SCORING]),
            "Valid RMSE": -np.mean(cv["test_" + SCORING]),
            "Valid RMSE SD": np.std(cv["test_" + SCORING]),
            "Train R2": np.mean(cv["train_r2"]),
            "Valid R2": np.mean(cv["test_r2"]),
        })
    return pd.DataFrame(rows).round(4)


def evaluate(model, x, y):
    """RMSE, MAE and R^2 (with non-negative predictions)."""
    y_pred = predict_non_negative(model, x)
    return {"RMSE": root_mean_squared_error(y, y_pred),
            "MAE": mean_absolute_error(y, y_pred),
            "R2": r2_score(y, y_pred)}


def comparison_table(searches, x_train, y_train, x_test, y_test):
    """Summary of all candidate models: validation (CV) and test metrics."""
    baseline = DummyRegressor(strategy="mean")
    cv = cross_validate(baseline, x_train, y_train, cv=CV_FOLDS, scoring=SCORING)
    baseline.fit(x_train, y_train)
    rows = [{
        "Model": "Baseline (mean)", "Hyperparameters": "none",
        "Valid RMSE": -np.mean(cv["test_score"]),
        "Valid RMSE SD": np.std(cv["test_score"]),
        "Train RMSE": evaluate(baseline, x_train, y_train)["RMSE"],
        "Test RMSE": evaluate(baseline, x_test, y_test)["RMSE"],
        "Test MAE": evaluate(baseline, x_test, y_test)["MAE"],
        "Test R2": evaluate(baseline, x_test, y_test)["R2"],
    }]
    for name, model_gscv in searches.items():
        rmse, sd = cv_scores(model_gscv)
        test = evaluate(model_gscv, x_test, y_test)
        rows.append({
            "Model": name, "Hyperparameters": best_params_string(model_gscv),
            "Valid RMSE": rmse, "Valid RMSE SD": sd,
            "Train RMSE": evaluate(model_gscv, x_train, y_train)["RMSE"],
            "Test RMSE": test["RMSE"], "Test MAE": test["MAE"],
            "Test R2": test["R2"],
        })
    return pd.DataFrame(rows).round(4)


# ---------------------------------------------------------------------
# Learning curves (same procedure as the course: a random subset of the
# dataset, a new train/test split, and a new grid search at every size)
# ---------------------------------------------------------------------
def learning_curve_data(df):
    rng = np.random.default_rng(seed=LC_SEED)
    x_all = df.loc[:, FEATURES].to_numpy()
    y_all = df.loc[:, TARGET].to_numpy()
    rows = []
    for n_samples in LC_SAMPLE_SIZES:
        idxs = rng.choice(x_all.shape[0], size=n_samples, replace=False)
        x_train, x_test, y_train, y_test = train_test_split(
            x_all[idxs], y_all[idxs], test_size=TEST_SIZE,
            random_state=RANDOM_STATE)

        # A training fold has 75 % of the training samples, which limits k
        k_max = min(max(KNN_NEIGHBORS), int(0.75 * x_train.shape[0]))
        gscv_poly = run_grid_search(
            build_model_poly(), dict(polynomialfeatures__degree=POLY_DEGREES),
            x_train, y_train)
        gscv_knn = run_grid_search(
            build_model_knn(),
            dict(kneighborsregressor__n_neighbors=list(range(1, k_max + 1)),
                 kneighborsregressor__weights=KNN_WEIGHTS),
            x_train, y_train)

        rows.append({
            "N train": x_train.shape[0],
            "Poly": best_params_string(gscv_poly),
            "Poly train RMSE": evaluate(gscv_poly, x_train, y_train)["RMSE"],
            "Poly test RMSE": evaluate(gscv_poly, x_test, y_test)["RMSE"],
            "kNN": best_params_string(gscv_knn),
            "kNN train RMSE": evaluate(gscv_knn, x_train, y_train)["RMSE"],
            "kNN test RMSE": evaluate(gscv_knn, x_test, y_test)["RMSE"],
        })
    return pd.DataFrame(rows).round(4)


# ---------------------------------------------------------------------
# Interpretation
# ---------------------------------------------------------------------
def short_name(feature):
    """Remove the units, e.g. 'Glucose [mM]' becomes 'Glucose'."""
    return feature.split(" [")[0]


def polynomial_weights(model):
    """Weights of a fitted polynomial model, printed like the course does.
    The features are standardized, so all weights are on the same scale."""
    powers = model["polynomialfeatures"].powers_
    coefs = model["linearregression"].coef_
    intercept = model["linearregression"].intercept_

    terms = []
    for i in range(powers.shape[0]):
        parts = []
        for j in range(powers.shape[1]):
            name = short_name(FEATURES[j])
            if powers[i, j] == 1:
                parts.append(f"[{name}]")
            elif powers[i, j] > 1:
                parts.append(f"[{name}]^{powers[i, j]}")
        terms.append(" * ".join(parts))

    table = pd.DataFrame({"Term": terms, "Weight": coefs})
    table = table.reindex(table["Weight"].abs().sort_values(ascending=False).index)
    baseline = pd.DataFrame({"Term": ["Intercept (baseline)"], "Weight": [intercept]})
    return pd.concat([baseline, table], ignore_index=True).round(4)


def one_factor_profiles(model, x):
    """Predicted growth rate when one feature varies and the others are held
    at their mean (the visual technique the course uses for kNN)."""
    means = np.mean(x, axis=0)
    profiles = []
    for j in range(x.shape[1]):
        grid = np.linspace(np.min(x[:, j]), np.max(x[:, j]), N_GRID_POINTS)
        x_plot = np.tile(means, (N_GRID_POINTS, 1))
        x_plot[:, j] = grid
        profiles.append((grid, predict_non_negative(model, x_plot)))
    return profiles


def fit_domain(x):
    """Describe where predictions can be trusted. A condition is 'reliable' if
    its distance to the 5th nearest sample (standardized units) is no larger
    than what 95 % of the samples have to their own 5th nearest neighbor."""
    scaler = StandardScaler().fit(x)
    nn = NearestNeighbors().fit(scaler.transform(x))
    dist = nn.kneighbors(scaler.transform(x), n_neighbors=RELIABILITY_K + 1)[0][:, -1]
    return scaler, nn, np.percentile(dist, RELIABILITY_PERCENTILE)


def is_reliable(domain, x_new):
    scaler, nn, threshold = domain
    dist = nn.kneighbors(scaler.transform(x_new), n_neighbors=RELIABILITY_K)[0][:, -1]
    return dist <= threshold


def find_extreme_conditions(model, x, domain):
    """Search a 4-D grid (5th to 95th percentile of each feature) for the highest
    and lowest predicted growth rates, keeping only reliable conditions."""
    axes = [np.linspace(np.percentile(x[:, j], 5), np.percentile(x[:, j], 95),
                        EXTREME_GRID_POINTS) for j in range(x.shape[1])]
    mesh = np.meshgrid(*axes, indexing="ij")
    grid = np.column_stack([m.ravel() for m in mesh])
    grid = grid[is_reliable(domain, grid)]
    pred = predict_non_negative(model, grid)

    order = np.argsort(pred)[::-1]
    n = max(int(len(order) * EXTREME_FRACTION), 1)
    groups = {"Best single condition": order[:1],
              f"High growth (top {EXTREME_FRACTION:.0%})": order[:n],
              f"Low growth (bottom {EXTREME_FRACTION:.0%})": order[-n:]}

    rows = []
    for name, idx in groups.items():
        row = {"Group": name, "N": len(idx),
               "Predicted growth rate": np.mean(pred[idx])}
        for j, feature in enumerate(FEATURES):
            lo, mid, hi = np.percentile(grid[idx, j], [5, 50, 95])
            row[feature] = f"{mid:.2f}" if len(idx) == 1 else f"{mid:.2f} ({lo:.2f}-{hi:.2f})"
        rows.append(row)
    return pd.DataFrame(rows).round(3)


def reliability_by_range(model, x, y, n_bins=4):
    """Sample count and out-of-fold residual RMSE for each quartile of each
    feature, to spot regions with little data or larger errors."""
    y_pred = np.maximum(cross_val_predict(clone(model), x, y, cv=CV_FOLDS), 0)
    resid = y - y_pred
    rows = []
    for j, feature in enumerate(FEATURES):
        bins = pd.qcut(x[:, j], q=n_bins, duplicates="drop")
        df_bin = pd.DataFrame({"bin": bins, "r": resid})
        for interval, g in df_bin.groupby("bin", observed=True)["r"]:
            rows.append({"Feature": feature, "Range": str(interval), "N": len(g),
                         "Residual RMSE": np.sqrt(np.mean(g ** 2))})
    return pd.DataFrame(rows).round(4)


# ---------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------
def _style_axis(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel, fontsize=FONTSIZE_XY_LABEL)
    ax.set_ylabel(ylabel, fontsize=FONTSIZE_XY_LABEL)
    ax.tick_params(axis="both", which="major", labelsize=FONTSIZE_TICKS)


def plot_overview(df):
    """Distribution of every variable, and growth rate versus each feature."""
    columns = [TARGET] + FEATURES
    fig, axes = plt.subplots(1, 5, figsize=(FIG_WIDTH_WIDE, FIG_HEIGHT * 0.5),
                             dpi=200, layout="constrained")
    for ax, col in zip(axes, columns):
        ax.hist(df[col], bins=20, color="tab:blue", edgecolor="white")
        ax.set_xlabel(col)
    axes[0].set_ylabel("Count")
    fig.savefig(FIGURES_DIR / "figure_distributions.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(FIG_WIDTH_WIDE, FIG_HEIGHT * 0.6),
                             dpi=200, layout="constrained", sharey=True)
    for ax, col in zip(axes, FEATURES):
        ax.scatter(df[col], df[TARGET], s=12, alpha=0.5, color="tab:blue")
        binned = df.groupby(pd.qcut(df[col], 8, duplicates="drop"),
                            observed=True).agg(x=(col, "mean"), y=(TARGET, "mean"))
        ax.plot(binned["x"], binned["y"], "o-", color="tab:red", label="Binned mean")
        ax.set_xlabel(col)
    axes[0].set_ylabel(TARGET)
    axes[0].legend()
    fig.savefig(FIGURES_DIR / "figure_growth_vs_features.png")
    plt.close(fig)


def plot_validation_curve_poly(table):
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=200, layout="constrained")
    ax.plot(table["Degree"], table["Train RMSE"], label="Train (Poly)",
            color=COLORS[0], linestyle="-")
    ax.plot(table["Degree"], table["Valid RMSE"], label="Valid (Poly)",
            color=COLORS[0], linestyle="--")
    _style_axis(ax, "Polynomial Degree", f"Growth Rate RMSE [{STR_D1}]")
    ax.set_yscale("log")  # high degrees overfit so strongly that a log scale is needed
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.legend(fontsize=FONTSIZE_LEGEND)
    fig.savefig(FIGURES_DIR / "figure_validation_curve_poly.png")
    plt.close(fig)


def plot_validation_curve_knn(searches):
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=200, layout="constrained")
    for i, weight in enumerate(KNN_WEIGHTS, start=1):
        df_w = grid_results_table(searches[f"kNN ({weight})"]).sort_values("n_neighbors")
        ax.plot(df_w["n_neighbors"], df_w["Train RMSE"], color=COLORS[i],
                linestyle="-", label=f"Train (kNN: {weight})")
        ax.plot(df_w["n_neighbors"], df_w["Valid RMSE"], color=COLORS[i],
                linestyle="--", label=f"Valid (kNN: {weight})")
    _style_axis(ax, "k", f"Growth Rate RMSE [{STR_D1}]")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=FONTSIZE_LEGEND)
    fig.savefig(FIGURES_DIR / "figure_validation_curve_knn.png")
    plt.close(fig)


def plot_learning_curve(lc):
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=200, layout="constrained")
    ax.plot(lc["N train"], lc["Poly train RMSE"], label="RMSE Train (Poly)",
            color=COLORS[0], linestyle="-")
    ax.plot(lc["N train"], lc["Poly test RMSE"], label="RMSE Test (Poly)",
            color=COLORS[0], linestyle="--")
    ax.plot(lc["N train"], lc["kNN train RMSE"], label="RMSE Train (kNN)",
            color=COLORS[1], linestyle="-")
    ax.plot(lc["N train"], lc["kNN test RMSE"], label="RMSE Test (kNN)",
            color=COLORS[1], linestyle="--")
    _style_axis(ax, "Number of training samples", f"RMSE [{STR_D1}]")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=FONTSIZE_LEGEND)
    fig.savefig(FIGURES_DIR / "figure_learning_curve.png")
    plt.close(fig)


def plot_parity(models, x_test, y_test):
    """Measured versus predicted growth rate on the test set."""
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=200, layout="constrained")
    lims = [0, max(np.max(y_test), 0.8) * 1.05]
    ax.plot(lims, lims, color="black", linestyle="-", linewidth=0.5)
    for i, (name, model) in enumerate(models.items()):
        ax.scatter(y_test, predict_non_negative(model, x_test), s=25, alpha=0.7,
                   color=COLORS[i], edgecolors="black", linewidth=0.5, label=name)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    _style_axis(ax, f"Measured growth rate [{STR_D1}]", f"Predicted growth rate [{STR_D1}]")
    ax.legend(fontsize=FONTSIZE_LEGEND)
    fig.savefig(FIGURES_DIR / "figure_parity.png")
    plt.close(fig)


def plot_residuals(model, x_test, y_test, name):
    y_pred = predict_non_negative(model, x_test)
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=200, layout="constrained")
    ax.scatter(y_pred, y_test - y_pred, s=25, alpha=0.7, color=COLORS[0],
               edgecolors="black", linewidth=0.5, label=f"Test ({name})")
    ax.axhline(y=0, color="black", linewidth=0.5)
    _style_axis(ax, f"Predicted growth rate [{STR_D1}]", f"Residual [{STR_D1}]")
    ax.legend(fontsize=FONTSIZE_LEGEND)
    fig.savefig(FIGURES_DIR / "figure_residuals.png")
    plt.close(fig)


def plot_one_factor_effects(models, x):
    """One-factor effects of each model, with the training samples as a rug."""
    fig, axes = plt.subplots(1, 4, figsize=(FIG_WIDTH_WIDE, FIG_HEIGHT * 0.65),
                             dpi=200, layout="constrained", sharey=True)
    all_profiles = {name: one_factor_profiles(m, x) for name, m in models.items()}
    for j, ax in enumerate(axes):
        for i, (name, profiles) in enumerate(all_profiles.items()):
            grid, pred = profiles[j]
            ax.plot(grid, pred, color=COLORS[i], linestyle=LINESTYLES[i * 2], linewidth=2,
                    label=name)
        ax.plot(x[:, j], np.zeros(x.shape[0]), "|", color="black", alpha=0.3, markersize=8)
        ax.set_xlabel(FEATURES[j], fontsize=FONTSIZE_TICKS)
    axes[0].set_ylabel(f"Predicted growth rate [{STR_D1}]", fontsize=FONTSIZE_TICKS)
    axes[0].set_ylim(bottom=0)
    axes[0].legend(fontsize=FONTSIZE_LEGEND - 3)
    fig.savefig(FIGURES_DIR / "figure_one_factor_effects.png")
    plt.close(fig)


def plot_interaction_maps(model, x, domain):
    """Predicted growth rate in the temperature-pH and glucose-lactate planes
    (other features at their mean). The dashed line encloses the reliable region."""
    means = np.mean(x, axis=0)
    pairs = [(0, 1), (2, 3)]
    panels = []
    for a, b in pairs:
        ga, gb = np.meshgrid(np.linspace(np.min(x[:, a]), np.max(x[:, a]), 120),
                             np.linspace(np.min(x[:, b]), np.max(x[:, b]), 120))
        x_plot = np.tile(means, (ga.size, 1))
        x_plot[:, a], x_plot[:, b] = ga.ravel(), gb.ravel()
        panels.append((a, b, ga, gb,
                       predict_non_negative(model, x_plot).reshape(ga.shape),
                       is_reliable(domain, x_plot).reshape(ga.shape)))
    vmax = max(p[4].max() for p in panels)

    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH_WIDE, FIG_HEIGHT * 0.8),
                             dpi=200, layout="constrained")
    for ax, (a, b, ga, gb, z, mask) in zip(axes, panels):
        cs = ax.contourf(ga, gb, z, levels=np.linspace(0, vmax, 16), cmap="viridis")
        ax.contour(ga, gb, mask.astype(float), levels=[0.5], colors="white",
                   linestyles="--")
        ax.scatter(x[:, a], x[:, b], s=6, color="black", alpha=0.35)
        _style_axis(ax, FEATURES[a], FEATURES[b])
        others = [short_name(FEATURES[j]) for j in range(4) if j not in (a, b)]
        ax.set_title(f"{others[0]} and {others[1]} at their mean", fontsize=FONTSIZE_TICKS)
        fig.colorbar(cs, ax=ax, label=f"Predicted growth rate [{STR_D1}]")
    fig.savefig(FIGURES_DIR / "figure_interaction_maps.png")
    plt.close(fig)