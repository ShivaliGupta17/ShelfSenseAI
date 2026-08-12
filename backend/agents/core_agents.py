"""
Demand Agent, Inventory Agent, Pricing Agent.
Each agent has ONE clear responsibility and calls into the ML model /
simulation engine — they don't call an LLM. The Decision Agent (in
decision_agent.py) is the one that reasons over their combined output.
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from backend.models.forecast_model import predict_next_days
from backend.simulation import simulation_engine as sim


class DemandAgent:
    """Predicts future demand using the trained ML forecasting model."""
    def forecast(self, store: str, item: str, days: int = 7):
        preds = predict_next_days(store, item, days=days)
        avg = sum(p["predicted_units"] for p in preds) / len(preds)
        return {"store": store, "item": item, "horizon_days": days, "forecast": preds, "avg_daily_demand": round(avg, 1)}


class InventoryAgent:
    """Projects stock coverage and stock-out risk given current stock."""
    def check_coverage(self, store: str, item: str, current_stock: int):
        return sim.simulate_inventory(store, item, current_stock)

    def supplier_delay_impact(self, store: str, item: str, current_stock: int, delay_days: int):
        return sim.simulate_supplier_delay(store, item, current_stock, delay_days)

    def transfer_recommendation(self, item: str, store_stock: dict, avg_daily_demand: dict):
        return sim.simulate_store_transfer(store_stock, item, avg_daily_demand)


class PricingAgent:
    """Compares price-change scenarios and their revenue/profit/margin impact."""
    def compare_scenarios(self, store: str, item: str, price_changes_pct: list):
        return sim.simulate_price_scenarios(store, item, price_changes_pct)
