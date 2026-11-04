"""Custom estimators. Kept in their own module (not model.py) so pickled models
always reference `estimators.X`, regardless of whether training was started via
`python model.py`, the API, or the unit tests."""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone


def baseline_pred(X):
    """Naive baseline: next 30 days == last 28 days scaled to 30 days."""
    return X['previous_28'].values * (30.0 / 28.0)


class BaselineModel(BaseEstimator, RegressorMixin):
    """The naive baseline wrapped as an estimator (so it can be deployed if
    no ML model beats it for a given country)."""

    def fit(self, X, y):
        return self

    def predict(self, X):
        return baseline_pred(X)


class ResidualModel(BaseEstimator, RegressorMixin):
    """Learns the *correction* to the naive baseline: y_hat = baseline + f(X).
    Far more stable than learning absolute revenue from ~600 daily rows."""

    def __init__(self, estimator=None):
        self.estimator = estimator

    def fit(self, X, y):
        self.est_ = clone(self.estimator).fit(X, np.asarray(y) - baseline_pred(X))
        return self

    def predict(self, X):
        return baseline_pred(X) + self.est_.predict(X)
