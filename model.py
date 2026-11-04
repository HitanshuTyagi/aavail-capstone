"""
Model training / prediction for 30-day forward revenue.

 - model_train(): compares several candidate models vs a naive baseline with
   cross-validation, keeps the best one (per country) and saves it.
 - model_predict(): loads the saved model and predicts for one country
   (or 'all') on a given date.
 - MODEL_VERSION / MODEL_VERSION_NOTE are written to the logs.
 - test=True everywhere => reads/writes go to models/test-* and logs/*-test.log
   so production artefacts are never touched by the unit tests.
"""

import os
import time
import joblib
from datetime import date

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ingestion import fetch_ts, engineer_features
from estimators import BaselineModel, ResidualModel, baseline_pred
from logger import update_train_log, update_predict_log

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_VERSION = 0.1
MODEL_VERSION_NOTE = "30-day revenue forecast, best of Ridge/RF/GBR"
if not os.path.isdir(MODEL_DIR):
    os.mkdir(MODEL_DIR)


def _model_path(country, test):
    prefix = "test" if test else "prod"
    return os.path.join(MODEL_DIR, f"{prefix}-{country}-v{MODEL_VERSION}.joblib")


_baseline_pred = baseline_pred


def _candidates():
    return {
        "ridge": (ResidualModel(Pipeline([('scaler', StandardScaler()), ('m', Ridge())])),
                  {'estimator__m__alpha': [10.0, 100.0, 1000.0]}),
        "random_forest": (ResidualModel(RandomForestRegressor(random_state=42)),
                          {'estimator__n_estimators': [100],
                           'estimator__max_depth': [3, 4],
                           'estimator__min_samples_leaf': [10, 20]}),
        "gradient_boosting": (ResidualModel(GradientBoostingRegressor(random_state=42, subsample=0.8)),
                              {'estimator__n_estimators': [100],
                               'estimator__learning_rate': [0.03, 0.05],
                               'estimator__max_depth': [2]}),
    }


def compare_models(X, y, n_splits=3):
    """Cross-validated RMSE for baseline + every candidate. Returns dict of results."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = {}
    base_rmse = [np.sqrt(mean_squared_error(y[te], _baseline_pred(X.iloc[te])))
                 for _, te in tscv.split(X)]
    results['baseline'] = {'rmse': float(np.mean(base_rmse)), 'estimator': BaselineModel()}

    for name, (est, grid) in _candidates().items():
        gs = GridSearchCV(est, grid, cv=tscv, scoring='neg_root_mean_squared_error', n_jobs=-1)
        gs.fit(X, y)
        results[name] = {'rmse': float(-gs.best_score_), 'estimator': gs.best_estimator_}
    return results


def model_train(data_dir, test=False, countries=None):
    """Train models per country; returns the comparison table (also saved)."""
    t0 = time.time()
    ts_data = fetch_ts(data_dir)
    if test:   # keep unit tests fast and isolated
        ts_data = {k: v for k, v in ts_data.items() if k in ('all', 'germany')}
    if countries:
        ts_data = {k: v for k, v in ts_data.items() if k in countries}

    rows = []
    for country, df in ts_data.items():
        X, y, dates = engineer_features(df, training=True)
        res = compare_models(X, y)
        # baseline competes too: if no ML model beats it, the baseline is deployed
        best = min(res, key=lambda k: res[k]['rmse'])
        final = res[best]['estimator'].fit(X, y)
        joblib.dump({'model': final, 'name': best, 'version': MODEL_VERSION,
                     'trained': str(date.today()),
                     'last_date': str(df['date'].max().date()),
                     'ts': df}, _model_path(country, test))
        for k, v in res.items():
            rows.append({'country': country, 'model': k, 'rmse': v['rmse'],
                         'selected': k == best})
        m, s = divmod(time.time() - t0, 60)
        h, m = divmod(m, 60)
        update_train_log(str(X.shape), {'rmse': round(res[best]['rmse'], 2), 'model': best},
                         "%d:%02d:%02d" % (h, m, s), MODEL_VERSION,
                         MODEL_VERSION_NOTE, test=test)
    table = pd.DataFrame(rows)
    out = os.path.join(MODEL_DIR, f"{'test' if test else 'prod'}-model-comparison.csv")
    table.to_csv(out, index=False)
    return table


def model_load(country, test=False):
    path = _model_path(country, test)
    if not os.path.exists(path):
        raise FileNotFoundError(f"no trained model for '{country}' (looked for {path}). Train first.")
    return joblib.load(path)


PROD_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "cs-production")


def _history(art, country, test):
    """Training history, extended with newer (production) data when available,
    so forecasts can be made for dates after the training window.
    Test mode never touches production data."""
    ts = art['ts']
    if test or not os.path.isdir(PROD_DATA_DIR):
        return ts
    try:
        newer = fetch_ts(PROD_DATA_DIR)[country]
    except (KeyError, RuntimeError):
        return ts
    return (pd.concat([ts, newer]).drop_duplicates('date')
              .sort_values('date').reset_index(drop=True))


def model_predict(country, year, month, day, test=False):
    """Predict revenue for the 30 days after the given date."""
    t0 = time.time()
    try:
        query_date = pd.Timestamp(year=int(year), month=int(month), day=int(day))
    except Exception:
        raise ValueError(f"invalid date: {year}-{month}-{day}")

    art = model_load(country, test)
    ts = _history(art, country, test)
    if query_date < ts['date'].min() or query_date > ts['date'].max():
        raise ValueError(f"date {query_date.date()} outside available range "
                         f"{ts['date'].min().date()} .. {ts['date'].max().date()}")

    X, _, dates = engineer_features(ts, training=False)
    idx = np.where(dates == query_date.strftime('%Y-%m-%d'))[0]
    if idx.size == 0:
        raise ValueError(f"date {query_date.date()} not found in data")
    y_pred = float(art['model'].predict(X.iloc[idx])[0])

    runtime = "%.3fs" % (time.time() - t0)
    update_predict_log(country, round(y_pred, 2), None,
                       f"{year}-{month}-{day}", runtime, art['version'], test=test)
    return {'country': country, 'date': query_date.strftime('%Y-%m-%d'),
            'y_pred': round(y_pred, 2), 'model': art['name'],
            'model_version': art['version']}


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "cs-train")
    print(model_train(data_dir).to_string(index=False))
