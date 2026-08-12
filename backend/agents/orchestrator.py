"""
Orchestrator — the entry point every request goes through. Decides which
agents to call and assembles their outputs. This is the piece that turns
independent agents into one coherent "Agentic Retail Decision Intelligence
System" instead of a set of unrelated endpoints.
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import pandas as pd

from backend.agents.core_agents import DemandAgent, InventoryAgent, PricingAgent
from backend.agents.policy_agent import PolicyAgent
from backend.agents.decision_agent import DecisionAgent

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "retail_sales.csv")


class Orchestrator:
    def __init__(self):
        self.demand_agent = DemandAgent()
        self.inventory_agent = InventoryAgent()
        self.pricing_agent = PricingAgent()
        self.policy_agent = PolicyAgent()
        self.decision_agent = DecisionAgent(self.policy_agent)
        self._raw = pd.read_csv(DATA_PATH)

    def _current_stock_estimate(self, store: str, item: str) -> int:
        # No live stock feed in this dataset, so approximate current stock
        # as ~10 days of trailing average demand (used consistently across
        # all "next week plan" runs so results are comparable).
        avg = self._raw[(self._raw.store == store) & (self._raw.item == item)]["units_sold"].tail(14).mean()
        return int(round(avg * 10))

    def price_scenario(self, store: str, item: str, price_changes_pct: list):
        pricing_result = self.pricing_agent.compare_scenarios(store, item, price_changes_pct)
        return self.decision_agent.recommend_price(pricing_result)

    def inventory_scenario(self, store: str, item: str, current_stock: int = None):
        if current_stock is None:
            current_stock = self._current_stock_estimate(store, item)
        inv = self.inventory_agent.check_coverage(store, item, current_stock)
        return self.decision_agent.recommend_inventory_action(inv, item)

    def supplier_delay_scenario(self, store: str, item: str, delay_days: int, current_stock: int = None):
        if current_stock is None:
            current_stock = self._current_stock_estimate(store, item)
        delay = self.inventory_agent.supplier_delay_impact(store, item, current_stock, delay_days)
        return self.decision_agent.recommend_supplier_action(delay)

    def transfer_scenario(self, item: str, stores: list):
        store_stock, avg_demand = {}, {}
        for store in stores:
            stock = self._current_stock_estimate(store, item)
            demand = self.demand_agent.forecast(store, item, days=7)["avg_daily_demand"]
            store_stock[store], avg_demand[store] = stock, demand
        return self.inventory_agent.transfer_recommendation(item, store_stock, avg_demand)

    def policy_query(self, question: str):
        return {"question": question, "results": self.policy_agent.query(question)}

    def next_week_plan(self, store: str):
        """The 'Planner Agent' flow: fans out to Demand + Inventory + Pricing
        for every item in a store and rolls it into one action plan."""
        items = sorted(self._raw["item"].unique())
        reorder, monitor, pricing_notes = [], [], []

        for item in items:
            stock = self._current_stock_estimate(store, item)
            inv = self.inventory_agent.check_coverage(store, item, stock)
            decision = self.decision_agent.recommend_inventory_action(inv, item)
            if decision["urgency"] in ("critical", "reorder"):
                reorder.append({"item": item, "urgency": decision["urgency"], "reasoning": decision["reasoning"]})
            else:
                monitor.append({"item": item})

            price_rec = self.price_scenario(store, item, [-10, -5, 0, 5])
            best = price_rec["recommended_scenario"]
            if best["price_change_pct"] != 0:
                pricing_notes.append({
                    "item": item,
                    "suggested_change_pct": best["price_change_pct"],
                    "reasoning": price_rec["reasoning"],
                    "policy_note": price_rec["policy_note"],
                })

        transfer = None
        stores = sorted(self._raw["store"].unique())
        if store in stores and len(stores) > 1 and reorder:
            critical_item = reorder[0]["item"]
            transfer = self.transfer_scenario(critical_item, stores)

        return {
            "store": store,
            "reorder": reorder,
            "monitor": monitor,
            "pricing_recommendations": pricing_notes,
            "suggested_transfer": transfer,
        }
