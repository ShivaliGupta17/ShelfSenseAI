"""
Simulation Engine — pure calculation layer used by the agents.
No LLM involved here; this is deterministic what-if math on top of the
ML demand forecasts.
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import pandas as pd
from backend.models.forecast_model import predict_next_days, _load_item_history

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "retail_sales.csv")


def _current_stats(store: str, item: str):
    group = _load_item_history(store, item).sort_values("date")
    last = group.iloc[-1]
    avg_7 = group["units_sold"].tail(7).mean()
    return {
        "current_price": float(last["price"]),
        "base_price": float(last["base_price"]),
        "avg_daily_sales_7d": round(float(avg_7), 1),
    }


def simulate_price_scenarios(store: str, item: str, price_changes_pct: list, unit_cost_ratio: float = 0.65):
    """Compares multiple price-change % scenarios and returns expected
    sales/revenue/profit for each, using the trained demand model to
    capture price elasticity."""
    stats = _current_stats(store, item)
    base_price = stats["base_price"]
    unit_cost = base_price * unit_cost_ratio

    results = []
    for pct in price_changes_pct:
        preds = predict_next_days(store, item, days=7, price_change_pct=pct)
        avg_demand = sum(p["predicted_units"] for p in preds) / len(preds)
        new_price = round(base_price * (1 + pct / 100), 2)
        revenue = round(avg_demand * new_price, 2)
        profit = round(avg_demand * (new_price - unit_cost), 2)
        margin_pct = round((new_price - unit_cost) / new_price * 100, 1) if new_price > 0 else 0
        results.append({
            "price_change_pct": pct,
            "price": new_price,
            "expected_daily_sales": round(avg_demand, 1),
            "expected_daily_revenue": revenue,
            "expected_daily_profit": profit,
            "margin_pct": margin_pct,
        })
    return {"store": store, "item": item, "current_price": base_price, "scenarios": results}


def simulate_inventory(store: str, item: str, current_stock: int, demand_change_pct_scenarios=(0, 20, 40)):
    """Given current stock, projects stock-out day under several demand
    growth scenarios."""
    results = []
    for demand_bump in demand_change_pct_scenarios:
        preds = predict_next_days(store, item, days=30)
        stock = current_stock
        stockout_day = None
        for i, p in enumerate(preds, start=1):
            daily_demand = p["predicted_units"] * (1 + demand_bump / 100)
            stock -= daily_demand
            if stock <= 0 and stockout_day is None:
                stockout_day = i
                break
        results.append({
            "demand_change_pct": demand_bump,
            "projected_stockout_in_days": stockout_day if stockout_day else "30+",
        })
    return {"store": store, "item": item, "current_stock": current_stock, "scenarios": results}


def simulate_supplier_delay(store: str, item: str, current_stock: int, delay_days: int):
    preds = predict_next_days(store, item, days=max(delay_days, 1))
    projected_demand_during_delay = sum(p["predicted_units"] for p in preds[:delay_days])
    stock_after_delay = current_stock - projected_demand_during_delay
    risk = "High" if stock_after_delay < 0 else ("Medium" if stock_after_delay < current_stock * 0.15 else "Low")
    return {
        "store": store,
        "item": item,
        "delay_days": delay_days,
        "projected_demand_during_delay": round(projected_demand_during_delay, 1),
        "current_stock": current_stock,
        "projected_stock_after_delay": round(stock_after_delay, 1),
        "stockout_risk": risk,
    }


def simulate_store_transfer(store_stock: dict, item: str, avg_daily_demand: dict, coverage_target_days: int = 5):
    """store_stock: {store: units}, avg_daily_demand: {store: units/day}.
    Recommends transfers from surplus stores to stores below coverage target."""
    surplus, deficit = [], []
    for store, stock in store_stock.items():
        demand = max(avg_daily_demand.get(store, 1), 0.1)
        coverage = stock / demand
        if coverage > coverage_target_days * 2:
            surplus.append({"store": store, "coverage_days": round(coverage, 1), "stock": stock, "demand": demand})
        elif coverage < coverage_target_days:
            deficit.append({"store": store, "coverage_days": round(coverage, 1), "stock": stock, "demand": demand})

    transfers = []
    for d in deficit:
        needed = int((coverage_target_days - d["coverage_days"]) * d["demand"])
        for s in surplus:
            available = int(s["stock"] - coverage_target_days * 2 * s["demand"])
            if available > 0 and needed > 0:
                move = min(available, needed)
                if move > 0:
                    transfers.append({"item": item, "from": s["store"], "to": d["store"], "units": move})
                    s["stock"] -= move
                    needed -= move
    return {"item": item, "transfers": transfers}
