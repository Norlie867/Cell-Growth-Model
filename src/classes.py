# =======================================================================
# CLASSES
# Define custom classes in this file.
# =======================================================================

from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

from src.constants import CV_FOLDS, RANDOM_STATE


class ModelCandidate:
    """A regression model family (pipeline + hyperparameter grid) that is
    tuned with cross-validation and evaluated on a held-out test set."""

    def __init__(self, name, pipeline, param_grid):
        self.name = name
        self.pipeline = pipeline
        self.param_grid = param_grid
        self.search = None

    def tune(self, x_train, y_train):
        """Run a cross-validated grid search on the training set only."""
        cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        self.search = GridSearchCV(
            self.pipeline,
            self.param_grid,
            scoring="neg_root_mean_squared_error",
            cv=cv,
            return_train_score=True,
            n_jobs=-1,
        )
        self.search.fit(x_train, y_train)

    @property
    def best_estimator(self):
        return self.search.best_estimator_

    @property
    def best_params(self):
        return self.search.best_params_

    @property
    def cv_rmse(self):
        return -self.search.best_score_

    def evaluate(self, x, y):
        """Return RMSE, MAE and R2 of the tuned model on the given data."""
        y_pred = self.best_estimator.predict(x)
        return {
            "RMSE": float(root_mean_squared_error(y, y_pred)),
            "MAE": float(mean_absolute_error(y, y_pred)),
            "R2": float(r2_score(y, y_pred)),
        }

    def display_info(self):
        print(f"{self.name:<12} | CV RMSE: {self.cv_rmse:6.4f} | "
              f"Best params: {self.best_params}")