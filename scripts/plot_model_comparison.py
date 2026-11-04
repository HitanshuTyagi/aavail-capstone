#!/usr/bin/env python
"""
Visual comparison: every candidate model vs the baseline (cross-validated RMSE),
plus predicted vs actual on the production hold-out for the deployed model.

    python scripts/plot_model_comparison.py      (needs `python model.py` first)
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ingestion import fetch_ts, engineer_features
from model import model_load, _baseline_pred

FIG = os.path.join(ROOT, "reports", "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})


def main():
    t = pd.read_csv(os.path.join(ROOT, "models", "prod-model-comparison.csv"))

    # --- 1. all countries combined: bar chart of CV RMSE, baseline highlighted
    a = t[t["country"] == "all"].sort_values("rmse")
    colors = ["#c0392b" if m == "baseline" else "#3b6ea5" for m in a["model"]]
    plt.figure(figsize=(6, 3.5))
    plt.bar(a["model"], a["rmse"], color=colors)
    for i, v in enumerate(a["rmse"]):
        plt.text(i, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=8)
    plt.ylabel("CV RMSE (lower is better)")
    plt.title("Model vs baseline - all countries")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "08_model_vs_baseline_all.png"))
    plt.close()

    # --- 2. every country: % improvement of best model over baseline
    piv = t.pivot(index="country", columns="model", values="rmse")
    best = t[t["selected"]].set_index("country")["rmse"]
    gain = ((piv["baseline"] - best) / piv["baseline"] * 100).sort_values()
    plt.figure(figsize=(6.5, 4))
    plt.barh(gain.index, gain.values, color=["#3b6ea5" if g > 0 else "#999" for g in gain.values])
    plt.axvline(0, color="k", lw=.8)
    plt.xlabel("RMSE improvement of deployed model over baseline (%)")
    plt.title("Where the model beats the baseline")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "09_improvement_by_country.png"))
    plt.close()

    # --- 3. production hold-out: predicted vs actual vs baseline (all)
    art = model_load("all")
    train_ts = fetch_ts(os.path.join(ROOT, "data", "cs-train"))["all"]
    prod_ts = fetch_ts(os.path.join(ROOT, "data", "cs-production"))["all"]
    full = pd.concat([train_ts, prod_ts]).drop_duplicates("date").sort_values("date").reset_index(drop=True)
    X, y, dates = engineer_features(full, training=True)
    mask = pd.to_datetime(dates) >= prod_ts["date"].min()
    plt.figure(figsize=(9, 3.8))
    d = pd.to_datetime(dates)[mask]
    plt.plot(d, y[mask], color="k", label="actual (next 30d)")
    plt.plot(d, art["model"].predict(X[mask]), color="#3b6ea5", label=f"model ({art['name']})")
    plt.plot(d, _baseline_pred(X[mask]), color="#c0392b", ls="--", label="baseline")
    plt.legend()
    plt.title("Production hold-out: model vs baseline vs actual")
    plt.ylabel("30-day revenue")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "10_holdout_pred_vs_actual.png"))
    plt.close()
    print("saved figures 08, 09, 10")


if __name__ == "__main__":
    main()
