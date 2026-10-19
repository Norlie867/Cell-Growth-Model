
from sklearn.base import clone

import src.functions as sf
from src.constants import FIGURES_DIR, TABLES_DIR


def main():
    FIGURES_DIR.mkdir(exist_ok=True)
    TABLES_DIR.mkdir(exist_ok=True)

    # 1. Load and explore the data ------------------------------------------
    df = sf.load_data()
    stats, corr = sf.summarize_data(df)
    stats.to_csv(TABLES_DIR / "data_summary.csv")
    corr.to_csv(TABLES_DIR / "correlation_matrix.csv")
    sf.plot_overview(df)
    print(f"\nLoaded {len(df)} samples.")

    # 2. Split the data (the test set is never used for tuning) --------------
    x_train, x_test, y_train, y_test = sf.split_data(df)
    x_all = df.loc[:, sf.FEATURES].to_numpy()
    y_all = df.loc[:, sf.TARGET].to_numpy()
    print(f"Train: {x_train.shape[0]} samples | Test: {x_test.shape[0]} samples")

    # 3. Tune the candidate models with grid search cross-validation ----------
    searches = sf.tune_models(x_train, y_train)

    degree_table = sf.polynomial_degree_table(x_train, y_train)
    degree_table.to_csv(TABLES_DIR / "polynomial_degrees.csv", index=False)
    sf.grid_results_table(searches["Polynomial"]).to_csv(
        TABLES_DIR / "grid_results_poly.csv", index=False)
    for weight in ("uniform", "distance"):
        sf.grid_results_table(searches[f"kNN ({weight})"]).to_csv(
            TABLES_DIR / f"grid_results_knn_{weight}.csv", index=False)
    print("\nPolynomial degree (4-fold cross-validation on the training set):")
    print(degree_table.to_string(index=False))

    sf.plot_validation_curve_poly(degree_table)
    sf.plot_validation_curve_knn(searches)

    # 4. Compare all candidates, then select the final model -----------------
    comparison = sf.comparison_table(searches, x_train, y_train, x_test, y_test)
    comparison.to_csv(TABLES_DIR / "model_comparison.csv", index=False)
    print("\nModel comparison:")
    print(comparison.to_string(index=False))

    name_final, name_other = sf.select_final_model(searches)
    print(f"\nSelected model: {name_final} ({sf.best_params_string(searches[name_final])})")

    # 5. Learning curves --------------------------------------------------------
    lc = sf.learning_curve_data(df)
    lc.to_csv(TABLES_DIR / "learning_curve.csv", index=False)
    sf.plot_learning_curve(lc)

    # 6. Evaluate the selected model on the test set ----------------------------
    models_test = {name_final: searches[name_final],
                   name_other: searches[name_other]}
    sf.plot_parity(models_test, x_test, y_test)
    sf.plot_residuals(searches[name_final], x_test, y_test, name_final)

    # 7. Refit on all the data, then interpret the models ----------------------
    final = clone(searches[name_final].best_estimator_).fit(x_all, y_all)
    other = clone(searches[name_other].best_estimator_).fit(x_all, y_all)

    if name_final == "Polynomial":
        weights = sf.polynomial_weights(final)
        weights.to_csv(TABLES_DIR / "polynomial_weights.csv", index=False)
        print("\nPolynomial weights (standardized features):")
        print(weights.to_string(index=False))

    sf.plot_one_factor_effects({name_final: final, name_other: other}, x_all)

    domain = sf.fit_domain(x_all)
    sf.plot_interaction_maps(final, x_all, domain)

    conditions = sf.find_extreme_conditions(final, x_all, domain)
    conditions.to_csv(TABLES_DIR / "extreme_conditions.csv", index=False)
    print("\nHigh and low growth conditions (reliable region only):")
    print(conditions.to_string(index=False))

    reliability = sf.reliability_by_range(final, x_all, y_all)
    reliability.to_csv(TABLES_DIR / "reliability_by_range.csv", index=False)

    print("\nDone. Tables saved in tables/, figures saved in figures/.")


if __name__ == "__main__":
    main()