"""
Decision Agent — combines outputs from Demand, Inventory, Pricing and
Policy agents into a single ranked recommendation with an explanation.

Reasoning is rule-based/deterministic by default (fully offline, no
external LLM call needed to demo the app). If GEMINI_API_KEY is set in
the environment, `explain_with_llm` upgrades the final explanation to a
natural-language one via the Gemini API — this is the "LLM explains the
result" layer from the architecture; the LLM never does the math itself.
"""
import os


class DecisionAgent:
    def __init__(self, policy_agent):
        self.policy_agent = policy_agent

    def recommend_price(self, pricing_result: dict, margin_floor_pct: float = 12.0):
        scenarios = pricing_result["scenarios"]
        viable = [s for s in scenarios if s["margin_pct"] >= margin_floor_pct]
        pool = viable if viable else scenarios
        best = max(pool, key=lambda s: s["expected_daily_profit"])

        policy_hits = self.policy_agent.query("price reduction approval limit margin floor")
        approval_note = None
        if best["price_change_pct"] <= -8:
            approval_note = "This exceeds the 8% manager auto-approval limit — regional sign-off required." \
                if best["price_change_pct"] > -15 else \
                "This exceeds 15% — category director approval and a margin-impact simulation are required."

        reasoning = (
            f"Scenario at {best['price_change_pct']:+.0f}% price change gives the highest expected daily profit "
            f"(₹{best['expected_daily_profit']:,.0f}) among options at or above the {margin_floor_pct:.0f}% margin floor, "
            f"projecting {best['expected_daily_sales']:.0f} units/day at ₹{best['price']:.2f}."
        )
        return {
            "recommended_scenario": best,
            "all_scenarios": scenarios,
            "reasoning": reasoning,
            "policy_note": approval_note,
            "relevant_policies": policy_hits,
        }

    def recommend_inventory_action(self, inventory_result: dict, item: str):
        base_case = inventory_result["scenarios"][0]
        worst_case = inventory_result["scenarios"][-1]
        policy_hits = self.policy_agent.query("reorder threshold stock coverage")

        stockout_days = base_case["projected_stockout_in_days"]
        urgency = "critical" if isinstance(stockout_days, int) and stockout_days <= 3 else \
                  "reorder" if isinstance(stockout_days, int) and stockout_days <= 5 else "monitor"

        reasoning = (
            f"At current demand, {item} is projected to stock out in {stockout_days} days; "
            f"under a 40% demand surge that drops to {worst_case['projected_stockout_in_days']} days."
        )
        return {"urgency": urgency, "reasoning": reasoning, "scenarios": inventory_result["scenarios"], "relevant_policies": policy_hits}

    def recommend_supplier_action(self, delay_result: dict):
        policy_hits = self.policy_agent.query("emergency order supplier delay escalation")
        risk = delay_result["stockout_risk"]
        if delay_result["delay_days"] >= 5:
            action = "Escalate to category director (delay ≥ 5 days) and evaluate emergency order."
        elif risk == "High":
            action = "Place emergency order with alternate supplier — projected stock-out within the delay window."
        elif risk == "Medium":
            action = "Monitor closely; consider a partial emergency order if forecast worsens."
        else:
            action = "No action needed — projected stock coverage absorbs the delay."
        return {"action": action, "risk": risk, "relevant_policies": policy_hits, "details": delay_result}

    def explain_with_llm(self, context: str) -> str:
        """Optional: upgrades a reasoning string to natural language via Gemini,
        if GEMINI_API_KEY is configured. Falls back to the rule-based text otherwise."""
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return context
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-2.0-flash")
            prompt = (
                "You are a retail decision-support assistant. Rewrite this analysis "
                "for a store manager in 2-3 clear sentences, keep all numbers exact, "
                "do not invent new facts:\n\n" + context
            )
            resp = model.generate_content(prompt)
            return resp.text.strip()
        except Exception:
            return context
