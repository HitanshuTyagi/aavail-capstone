#!/usr/bin/env python
"""
EDA with visualizations -> reports/figures/*.png

    python scripts/eda.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ingestion import fetch_data, fetch_ts

FIG = os.path.join(ROOT, "reports", "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})


def save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, name))
    plt.close()
    print("saved", name)


def main():
    data_dir = os.path.join(ROOT, "data", "cs-train")
    df = fetch_data(data_dir)
    ts = fetch_ts(data_dir)

    # 1. revenue by country (top 10) -> why we model countries separately
    rev = df.groupby("country")["price"].sum().sort_values()
    rev.plot.barh(color="#3b6ea5", figsize=(7, 4))
    plt.title("Total revenue by country (top 10)")
    plt.xlabel("Revenue")
    save("01_revenue_by_country.png")

    # 2. daily revenue with 28-day rolling mean
    a = ts["all"].set_index("date")["revenue"]
    ax = a.plot(figsize=(9, 3.5), alpha=.35, color="#3b6ea5", label="daily")
    a.rolling(28).mean().plot(ax=ax, color="#c0392b", label="28-day mean")
    plt.title("Daily revenue - all countries")
    plt.legend()
    save("02_daily_revenue.png")

    # 3. monthly revenue (seasonality / trend)
    m = ts["all"].groupby("year_month")["revenue"].sum()
    m.plot.bar(figsize=(9, 3.5), color="#3b6ea5")
    plt.title("Monthly revenue - all countries")
    plt.xlabel("")
    save("03_monthly_revenue.png")

    # 4. day-of-week pattern
    d = ts["all"].assign(dow=ts["all"]["date"].dt.day_name())
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    d.groupby("dow")["revenue"].mean().reindex(order).plot.bar(figsize=(6, 3.5), color="#3b6ea5")
    plt.title("Average revenue by day of week")
    plt.xlabel("")
    save("04_dow_revenue.png")

    # 5. distribution of invoice price (outliers)
    df["price"].clip(upper=df["price"].quantile(.99)).plot.hist(bins=50, figsize=(6, 3.5), color="#3b6ea5")
    plt.title("Line-item price distribution (clipped at p99)")
    save("05_price_hist.png")

    # 6. correlation of engineered features with the target
    from ingestion import engineer_features
    X, y, _ = engineer_features(ts["all"], training=True)
    corr = X.assign(target_next_30d=y).corr()["target_next_30d"].drop("target_next_30d").sort_values()
    corr.plot.barh(figsize=(6, 4), color="#3b6ea5")
    plt.title("Feature correlation with next-30-day revenue")
    save("06_feature_correlation.png")

    # 7. top countries daily revenue comparison (small multiples)
    keys = [k for k in ts if k != "all"][:6]
    fig, axes = plt.subplots(2, 3, figsize=(11, 5), sharex=True)
    for ax, k in zip(axes.ravel(), keys):
        ts[k].set_index("date")["revenue"].rolling(28).mean().plot(ax=ax, color="#3b6ea5")
        ax.set_title(k.replace("_", " ").title(), fontsize=9)
        ax.set_xlabel("")
    fig.suptitle("28-day rolling revenue - top countries")
    save("07_country_small_multiples.png")

    # findings table
    print("\nKey facts")
    print(f"  rows: {len(df):,}   date range: {df['invoice_date'].min().date()} -> {df['invoice_date'].max().date()}")
    top = rev.sort_values(ascending=False)
    print(f"  UK share of revenue: {top.iloc[0] / top.sum():.0%}")


if __name__ == "__main__":
    main()
