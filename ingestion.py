#!/usr/bin/env python
"""
Data ingestion for the AAVAIL revenue capstone.

Automation-ready: every step is a function, and the module can be run as a
script to (re)build the cached time-series CSVs:

    python ingestion.py data/cs-train
"""

import os
import re
import sys
import shutil
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd

COLUMNS = ['country', 'customer_id', 'day', 'invoice', 'month',
           'price', 'stream_id', 'times_viewed', 'year']


def fetch_data(data_dir):
    """Load all monthly JSON files in data_dir into one clean DataFrame."""
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"specified data dir does not exist: {data_dir}")
    files = sorted(os.path.join(data_dir, f) for f in os.listdir(data_dir)
                   if re.search(r"\.json$", f))
    if not files:
        raise RuntimeError(f"no json files found in {data_dir}")

    frames = []
    for f in files:
        df = pd.read_json(f)
        df = df.rename(columns={'StreamID': 'stream_id',
                                'TimesViewed': 'times_viewed',
                                'total_price': 'price'})
        missing = set(COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"{f} is missing columns {missing}")
        frames.append(df[COLUMNS])

    df = pd.concat(frames, sort=False, ignore_index=True)
    df['invoice_date'] = pd.to_datetime(df[['year', 'month', 'day']])
    df['invoice'] = df['invoice'].astype(str).str.replace(r"\D+", "", regex=True)
    df = df[df['price'] > 0]                      # drop refunds / bad rows
    df = df.sort_values('invoice_date').reset_index(drop=True)
    return df


def convert_to_ts(df_orig, country=None):
    """Aggregate transactions into a daily time series (optionally for one country)."""
    if country:
        if country not in df_orig['country'].unique():
            raise ValueError(f"country not found: {country}")
        df = df_orig[df_orig['country'] == country]
    else:
        df = df_orig

    start = df_orig['invoice_date'].min()
    stop = df_orig['invoice_date'].max()
    days = pd.date_range(start, stop, freq='D')

    g = df.groupby('invoice_date')
    ts = pd.DataFrame({
        'date': days,
        'purchases': g.size().reindex(days, fill_value=0).values,
        'unique_invoices': g['invoice'].nunique().reindex(days, fill_value=0).values,
        'unique_streams': g['stream_id'].nunique().reindex(days, fill_value=0).values,
        'total_views': g['times_viewed'].sum().reindex(days, fill_value=0).values,
        'revenue': g['price'].sum().reindex(days, fill_value=0).values,
    })
    ts['year_month'] = ts['date'].dt.strftime('%Y-%m')
    return ts


def fetch_ts(data_dir, clean=False, top_n=10):
    """Return {country_key: daily_ts}; caches to data_dir/ts-data for speed."""
    ts_dir = os.path.join(data_dir, "ts-data")
    if clean and os.path.exists(ts_dir):
        shutil.rmtree(ts_dir)
    if os.path.isdir(ts_dir) and os.listdir(ts_dir):
        print("... loading ts data from files")
        return {re.sub(r"\.csv$", "", re.sub(r"^ts-", "", f)):
                pd.read_csv(os.path.join(ts_dir, f), parse_dates=['date'])
                for f in os.listdir(ts_dir)}

    print("... processing data for loading")
    os.makedirs(ts_dir, exist_ok=True)
    df = fetch_data(data_dir)
    top = (df.groupby('country')['price'].sum()
             .sort_values(ascending=False).index[:top_n])

    dfs = {'all': convert_to_ts(df)}
    for c in top:
        dfs[re.sub(r"\s+", "_", c.lower())] = convert_to_ts(df, country=c)
    for key, ts in dfs.items():
        ts.to_csv(os.path.join(ts_dir, f"ts-{key}.csv"), index=False)
    return dfs


def engineer_features(df, training=True):
    """
    For every day build look-back revenue features; target = revenue in the
    next 30 days. When training=True the last 30 days (no complete target)
    are dropped.
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    dates = df['date'].values.astype('datetime64[D]')
    revenue = df['revenue'].values
    previous = [7, 14, 28, 70]

    eng = defaultdict(list)
    y = np.zeros(dates.size)
    for i, d in enumerate(dates):
        for n in previous:
            mask = (dates > d - np.timedelta64(n, 'D')) & (dates <= d)
            eng[f'previous_{n}'].append(revenue[mask].sum())
        # target: next 30 days
        mask = (dates > d) & (dates <= d + np.timedelta64(30, 'D'))
        y[i] = revenue[mask].sum()
        # same 30-day window one year ago
        start = d - np.timedelta64(365, 'D')
        mask = (dates >= start) & (dates < start + np.timedelta64(30, 'D'))
        eng['previous_year'].append(revenue[mask].sum())
        # recent activity (last 30 days averages)
        mask = (dates > d - np.timedelta64(30, 'D')) & (dates <= d)
        eng['recent_invoices'].append(df['unique_invoices'].values[mask].mean())
        eng['recent_views'].append(df['total_views'].values[mask].mean())

    X = pd.DataFrame(eng)
    # yearly seasonality (cyclical encoding of day-of-year)
    doy = pd.to_datetime(pd.Series(dates)).dt.dayofyear.values
    X['season_sin'] = np.sin(2 * np.pi * doy / 365.25)
    X['season_cos'] = np.cos(2 * np.pi * doy / 365.25)
    X['has_prev_year'] = (X['previous_year'] > 0).astype(int)
    if training:
        keep = dates <= dates.max() - np.timedelta64(30, 'D')
        X, y, dates = X[keep], y[keep], dates[keep]
    X = X.reset_index(drop=True)
    return X, y, pd.Series(dates).dt.strftime('%Y-%m-%d').values


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "cs-train")
    t0 = datetime.now()
    ts = fetch_ts(data_dir, clean=True)
    for k, v in ts.items():
        print(f"{k:16s} {v.shape}")
    print(f"ingestion finished in {datetime.now() - t0}")
