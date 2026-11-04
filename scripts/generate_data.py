#!/usr/bin/env python
"""
Generate synthetic AAVAIL-style invoice data (same JSON schema as the IBM
capstone "cs-train" / "cs-production" folders) so the project runs end-to-end
even without the original course data.

If you have the real data, just drop the JSON files into data/cs-train and
data/cs-production and skip this script.

usage: python scripts/generate_data.py
"""

import os
import json
import numpy as np
import pandas as pd

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

COUNTRIES = {
    "United Kingdom": 0.70, "EIRE": 0.06, "Germany": 0.05, "France": 0.04,
    "Norway": 0.03, "Spain": 0.025, "Hong Kong": 0.02, "Portugal": 0.02,
    "Singapore": 0.02, "Netherlands": 0.015,
}


def _make_month(year, month, rng, invoice_start):
    days = pd.date_range(f"{year}-{month:02d}-01", periods=31, freq="D")
    days = [d for d in days if d.month == month]
    rows, inv = [], invoice_start
    for d in days:
        # weekly + yearly seasonality and a mild upward trend
        season = 1 + 0.35 * np.sin(2 * np.pi * (d.dayofyear - 80) / 365)
        weekly = 0.6 if d.dayofweek == 5 else 1.0
        trend = 1 + 0.0004 * (d - pd.Timestamp("2017-11-01")).days
        n_inv = rng.poisson(18 * season * weekly * trend)
        for _ in range(n_inv):
            country = rng.choice(list(COUNTRIES), p=np.array(list(COUNTRIES.values())) / sum(COUNTRIES.values()))
            customer = int(rng.integers(12000, 18500))
            inv += 1
            for _ in range(rng.integers(1, 12)):
                rows.append({
                    "country": country,
                    "customer_id": customer,
                    "day": int(d.day),
                    "invoice": str(inv),
                    "month": int(month),
                    "price": round(float(rng.gamma(2.0, 2.2)), 2),
                    "stream_id": f"{rng.integers(20000, 23000)}",
                    "times_viewed": int(rng.integers(1, 25)),
                    "year": int(year),
                })
    return rows, inv


def main(seed=42):
    rng = np.random.default_rng(seed)
    inv = 489000
    periods = pd.period_range("2017-11", "2019-11", freq="M")
    for p in periods:
        rows, inv = _make_month(p.year, p.month, rng, inv)
        folder = "cs-production" if p >= pd.Period("2019-08", "M") else "cs-train"
        out = os.path.join(BASE, folder, f"invoices-{p.year}-{p.month:02d}.json")
        # mimic the inconsistent column names of the real data in a few files
        if p.month in (2, 6):
            rows = [{("StreamID" if k == "stream_id" else "TimesViewed" if k == "times_viewed" else k): v
                     for k, v in r.items()} for r in rows]
        with open(out, "w") as f:
            json.dump(rows, f)
        print(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
