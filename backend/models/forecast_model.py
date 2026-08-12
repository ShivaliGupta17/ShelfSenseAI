"""
Real ML forecasting model (not an LLM). Trains a RandomForestRegressor per
store-item on lag/rolling/seasonal features, used by the Demand Agent to
predict future daily units_sold, including under hypothetical price/promo
scenarios (what powers the pricing & inventory simulations).
"""
import os
from functools import lru_cache
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "retail_sales.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "trained")
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURES = [
    "dow", "month", "promo", "price_ratio",
    "lag_1", "lag_7", "lag_14", "rolling_mean_7", "rolling_mean_14",
]


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").copy()
    df["date"] = pd.to_datetime(df["date"])
    df["dow"] = df["date"].dt.weekday
    df["month"] = df["date"].dt.month
    df["price_ratio"] = df["price"] / df["base_price"]
    df["lag_1"] = df["units_sold"].shift(1)
    df["lag_7"] = df["units_sold"].shift(7)
    df["lag_14"] = df["units_sold"].shift(14)
    df["rolling_mean_7"] = df["units_sold"].shift(1).rolling(7).mean()
    df["rolling_mean_14"] = df["units_sold"].shift(1).rolling(14).mean()
    return df


def train_all_models(verbose: bool = True):
    raw = pd.read_csv(DATA_PATH)
    metrics = {}
    for (store, item), group in raw.groupby(["store", "item"]):
        feat = _build_features(group).dropna()
        X = feat[FEATURES]
        y = feat["units_sold"]
        split = int(len(X) * 0.85)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        mape = float(np.mean(np.abs((y_test - preds) / np.clip(y_test, 1, None))) * 100)
        metrics[f"{store}|{item}"] = {"mae": round(mae, 2), "mape_pct": round(mape, 2)}

        joblib.dump(model, os.path.join(MODEL_DIR, f"{store}_{item}.joblib"))
        if verbose:
            print(f"Trained {store}/{item}: MAE={mae:.2f} MAPE={mape:.1f}%")

    return metrics


@lru_cache(maxsize=64)
def load_model(store: str, item: str):
    path = os.path.join(MODEL_DIR, f"{store}_{item}.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No trained model for {store}/{item}. Run train_all_models() first.")
    return joblib.load(path)


@lru_cache(maxsize=32)
def _load_item_history(store: str, item: str):
    raw = pd.read_csv(DATA_PATH)
    return raw[(raw["store"] == store) & (raw["item"] == item)].copy()


def predict_next_days(store: str, item: str, days: int = 7, price_change_pct: float = 0.0, promo: bool = False):
    """Predicts units_sold for the next `days` days, optionally under a
    hypothetical price change (%) or promo flag — this is what the
    Pricing/Inventory agents call to run what-if scenarios."""
    group = _load_item_history(store, item)
    if group.empty:
        raise ValueError(f"No data for {store}/{item}")

    feat = _build_features(group).dropna()
    history = list(feat["units_sold"].tail(14))
    base_price = float(group["base_price"].iloc[0])
    hypothetical_price = base_price * (1 + price_change_pct / 100)
    model = load_model(store, item)

    last_date = pd.to_datetime(group["date"]).max()
    preds = []
    for i in range(1, days + 1):
        date = last_date + pd.Timedelta(days=i)
        row = {
            "dow": date.weekday(),
            "month": date.month,
            "promo": int(promo),
            "price_ratio": hypothetical_price / base_price,
            "lag_1": history[-1],
            "lag_7": history[-7] if len(history) >= 7 else history[-1],
            "lag_14": history[-14] if len(history) >= 14 else history[0],
            "rolling_mean_7": np.mean(history[-7:]),
            "rolling_mean_14": np.mean(history[-14:]),
        }
        X = pd.DataFrame([row])[FEATURES]
        pred = max(0.0, float(model.predict(X)[0]))
        preds.append({"date": date.strftime("%Y-%m-%d"), "predicted_units": round(pred, 1)})
        history.append(pred)

    return preds


if __name__ == "__main__":
    train_all_models()
