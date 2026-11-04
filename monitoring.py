"""
Performance monitoring.

 - model performance: compare predictions vs actuals (RMSE / MAPE) on the
   held-out production data, and compare with the baseline.
 - data drift: Kolmogorov-Smirnov test between train and production feature
   distributions (outlier / drift detection).
 - service performance: summary of runtimes from the predict logs.

    python monitoring.py            # prints a report and writes reports/monitoring.json
"""

import os
import json
import glob

import numpy as np
import pandas as pd
from scipy import stats

from ingestion import fetch_ts, engineer_features
from model import model_load, _baseline_pred

ROOT = os.path.dirname(os.path.abspath(__file__))


def _combined_features(country, train_dir, prod_dir):
    """Build features on train+production history (so rolling windows are
    complete), then split rows back into train / production periods."""
    tr = fetch_ts(train_dir)[country]
    pr = fetch_ts(prod_dir)[country]
    full = (pd.concat([tr, pr]).drop_duplicates('date')
              .sort_values('date').reset_index(drop=True))
    X, y, dates = engineer_features(full, training=True)
    is_prod = np.asarray(pd.to_datetime(dates) >= pr['date'].min())
    return X, y, is_prod


def performance_vs_actuals(country, train_dir, prod_dir, test=False):
    """Score the deployed model on production-period rows (true 30-day revenue known)."""
    art = model_load(country, test)
    X, y, is_prod = _combined_features(country, train_dir, prod_dir)
    X, y = X[is_prod], y[is_prod]
    pred = art['model'].predict(X)
    base = _baseline_pred(X)
    rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
    mape = lambda a, b: float(np.mean(np.abs(a - b) / np.maximum(a, 1e-9)))
    return {'country': country, 'n': int(len(y)),
            'model_rmse': rmse(y, pred), 'baseline_rmse': rmse(y, base),
            'model_mape': mape(y, pred), 'baseline_mape': mape(y, base)}


def drift_report(train_dir, prod_dir, country='all', alpha=0.01):
    """KS-test every engineered feature: training period vs production period."""
    X, _, is_prod = _combined_features(country, train_dir, prod_dir)
    # skip the first year of training rows (previous_year window incomplete)
    warm = X['has_prev_year'].values == 1
    out = {}
    for col in ['previous_7', 'previous_14', 'previous_28', 'previous_70',
                'previous_year', 'recent_invoices', 'recent_views']:
        stat, p = stats.ks_2samp(X.loc[~is_prod & warm, col], X.loc[is_prod, col])
        out[col] = {'ks_stat': float(stat), 'p_value': float(p), 'drift': bool(p < alpha)}
    return out


def service_report(log_dir=None, test=False):
    """Runtime summary from predict logs."""
    log_dir = log_dir or os.path.join(ROOT, 'logs')
    pattern = 'predict-test.log' if test else 'predict-[0-9]*.log'
    files = glob.glob(os.path.join(log_dir, pattern))
    if not files:
        return {'n_calls': 0}
    df = pd.concat([pd.read_csv(f, quotechar='|') for f in files])
    rt = df['runtime'].astype(str).str.rstrip('s').astype(float)
    return {'n_calls': int(len(df)), 'mean_runtime_s': float(rt.mean()),
            'p95_runtime_s': float(rt.quantile(0.95)), 'max_runtime_s': float(rt.max())}


if __name__ == "__main__":
    train_dir = os.path.join(ROOT, 'data', 'cs-train')
    prod_dir = os.path.join(ROOT, 'data', 'cs-production')
    report = {
        'performance': performance_vs_actuals('all', train_dir, prod_dir),
        'drift': drift_report(train_dir, prod_dir),
        'service': service_report(),
    }
    os.makedirs(os.path.join(ROOT, 'reports'), exist_ok=True)
    with open(os.path.join(ROOT, 'reports', 'monitoring.json'), 'w') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
