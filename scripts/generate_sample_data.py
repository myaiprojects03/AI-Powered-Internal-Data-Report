"""Generate an anonymised, deliberately messy sample order export.

The data is synthetic. It contains no real customers, products or companies.
It mimics a typical ERP/CSV export: two months of orders with a few known
data-quality problems (duplicates, inconsistent labels, missing values,
invalid rows) so the cleaning step has real work to do.

Usage:
    python scripts/generate_sample_data.py --out data/sample/orders.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REGIONS = ["North", "South", "East", "West", "Central"]
CATEGORIES = ["Electronics", "Home", "Office", "Outdoor"]
SEGMENTS = ["Retail", "Wholesale"]
BASE_PRICE = {"Electronics": 180.0, "Home": 65.0, "Office": 40.0, "Outdoor": 95.0}
COST_RATIO = {"Electronics": 0.68, "Home": 0.55, "Office": 0.50, "Outdoor": 0.58}


def build(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    order_no = 100000
    for month, n_orders in (("2026-09", 1450), ("2026-10", 1520)):
        start = pd.Timestamp(f"{month}-01")
        days = start.days_in_month
        oct_ = month == "2026-10"
        for _ in range(n_orders):
            order_no += 1
            region = rng.choice(REGIONS, p=[0.22, 0.20, 0.21, 0.19, 0.18])
            category = rng.choice(CATEGORIES, p=[0.30, 0.28, 0.24, 0.18])
            segment = rng.choice(SEGMENTS, p=[0.65, 0.35])
            units = int(rng.integers(1, 6) if segment == "Retail" else rng.integers(5, 30))
            price = BASE_PRICE[category] * rng.uniform(0.85, 1.15)
            cost = BASE_PRICE[category] * COST_RATIO[category] * rng.uniform(0.95, 1.05)

            discount = rng.uniform(0.0, 0.08) if segment == "Retail" else rng.uniform(0.05, 0.12)
            # Story 1: heavier wholesale discounting in October erodes margin.
            if oct_ and segment == "Wholesale":
                discount += rng.uniform(0.04, 0.08)

            return_p = 0.04
            # Story 2: electronics returns spike in the South in October.
            if oct_ and region == "South" and category == "Electronics":
                return_p = 0.17
            returned = rng.random() < return_p

            delivery = float(np.clip(rng.normal(4.0, 1.2), 1, 12))
            # Story 3: East delivery slows down in October.
            if oct_ and region == "East":
                delivery += rng.uniform(0.4, 1.4)

            # Story 4: Outdoor demand grows in October.
            if oct_ and category == "Outdoor" and rng.random() < 0.25:
                units += int(rng.integers(1, 4))

            rows.append({
                "order_id": f"SO-{order_no}",
                "order_date": (start + pd.Timedelta(days=int(rng.integers(0, days)))).strftime("%Y-%m-%d"),
                "customer_id": f"C-{int(rng.integers(1000, 1600))}",
                "region": region,
                "category": category,
                "segment": segment,
                "units": units,
                "unit_price": round(price, 2),
                "unit_cost": round(cost, 2),
                "discount_pct": round(discount * 100, 1),
                "returned": "Y" if returned else "N",
                "delivery_days": round(delivery, 1),
            })

    df = pd.DataFrame(rows)

    idx = rng.choice(df.index, size=40, replace=False)
    df.loc[idx[:15], "region"] = df.loc[idx[:15], "region"].str.upper()
    df.loc[idx[15:25], "region"] = " " + df.loc[idx[15:25], "region"] + " "
    df.loc[idx[25:35], "delivery_days"] = np.nan
    df.loc[idx[35:38], "units"] = -1                    
    df.loc[idx[38:40], "returned"] = "yes"               
    dupes = df.sample(18, random_state=seed)             
    df = pd.concat([df, dupes]).sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sample/orders.csv")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    build(args.seed).to_csv(out, index=False)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
