"""
Generates data/retail_sales.csv — 2 years of daily sales for 5 products
across 3 stores, with realistic weekly seasonality, yearly seasonality,
trend, promo effects, and price elasticity baked in, so the ML forecasting
model and pricing simulation have genuine signal to learn from.

Schema mirrors common public retail datasets (Kaggle "Store Item Demand
Forecasting Challenge" / Walmart sales): date, store, item, price, promo,
units_sold. No live internet dataset download is reachable from this
sandboxed environment, so this generator stands in for one — swap in a
real CSV with the same columns and everything downstream keeps working.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

rng = np.random.default_rng(42)

STORES = ["Store_A", "Store_B", "Store_C"]
PRODUCTS = [
    {"item": "Milk_1L",   "base_price": 50.0, "base_demand": 100, "elasticity": -1.8, "category": "Dairy"},
    {"item": "Rice_5kg",  "base_price": 60.0, "base_demand": 70,  "elasticity": -1.2, "category": "Staples"},
    {"item": "Bread_400g","base_price": 35.0, "base_demand": 150, "elasticity": -1.5, "category": "Bakery"},
    {"item": "Eggs_12pk", "base_price": 80.0, "base_demand": 60,  "elasticity": -1.3, "category": "Dairy"},
    {"item": "Cooking_Oil_1L","base_price": 140.0,"base_demand": 40,"elasticity": -1.0, "category": "Staples"},
]

STORE_MULTIPLIER = {"Store_A": 1.3, "Store_B": 1.0, "Store_C": 0.7}

START = datetime(2024, 1, 1)
DAYS = 730

rows = []
for store in STORES:
    for prod in PRODUCTS:
        # slow upward trend per product/store
        trend_slope = rng.uniform(0.00, 0.04)
        promo_days = set(rng.choice(DAYS, size=int(DAYS * 0.08), replace=False))
        current_price = prod["base_price"]
        for d in range(DAYS):
            date = START + timedelta(days=d)
            dow = date.weekday()  # 0=Mon
            month = date.month

            weekly_factor = {0: 0.9, 1: 0.85, 2: 0.9, 3: 0.95, 4: 1.15, 5: 1.35, 6: 1.2}[dow]
            yearly_factor = 1 + 0.15 * np.sin(2 * np.pi * (date.timetuple().tm_yday / 365) + 1.2)
            festive_bump = 1.4 if month in (10, 11) else (1.2 if month == 12 else 1.0)

            # price varies fairly often across a wide range so the model can
            # actually learn price elasticity (demand response to price),
            # not just seasonality
            price_shock = 1.0
            if rng.random() < 0.20:
                price_shock = rng.uniform(0.80, 1.15)
            price = round(prod["base_price"] * price_shock, 2)

            promo = 1 if d in promo_days else 0
            promo_factor = 1.35 if promo else 1.0

            price_ratio = price / prod["base_price"]
            elasticity_factor = price_ratio ** prod["elasticity"]

            base = prod["base_demand"] * STORE_MULTIPLIER[store]
            trend = 1 + trend_slope * (d / 30)

            noise = rng.normal(1.0, 0.08)
            demand = (
                base * weekly_factor * yearly_factor * festive_bump
                * promo_factor * elasticity_factor * trend * noise
            )
            units_sold = max(0, int(round(demand)))

            rows.append({
                "date": date.strftime("%Y-%m-%d"),
                "store": store,
                "item": prod["item"],
                "category": prod["category"],
                "price": price,
                "base_price": prod["base_price"],
                "promo": promo,
                "units_sold": units_sold,
            })

df = pd.DataFrame(rows)
df.to_csv("/home/claude/shelfsense-ai/data/retail_sales.csv", index=False)
print(df.shape)
print(df.head())
