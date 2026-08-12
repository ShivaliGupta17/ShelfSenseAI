import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd

from backend.agents.orchestrator import Orchestrator

app = FastAPI(title="ShelfSenseAI", description="Agentic Retail Decision Intelligence System")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

orch = Orchestrator()
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "retail_sales.csv")


class PriceScenarioRequest(BaseModel):
    store: str
    item: str
    price_changes_pct: List[float] = [-10, -5, 0, 5]


class InventoryRequest(BaseModel):
    store: str
    item: str
    current_stock: Optional[int] = None


class SupplierDelayRequest(BaseModel):
    store: str
    item: str
    delay_days: int
    current_stock: Optional[int] = None


class TransferRequest(BaseModel):
    item: str
    stores: List[str]


class PolicyQuery(BaseModel):
    question: str


class NextWeekPlanRequest(BaseModel):
    store: str


@app.get("/api/meta")
def meta():
    df = pd.read_csv(DATA_PATH)
    return {"stores": sorted(df["store"].unique().tolist()), "items": sorted(df["item"].unique().tolist())}


@app.get("/api/kpis")
def kpis():
    df = pd.read_csv(DATA_PATH)
    recent = df[df["date"] >= df["date"].max()]
    revenue = float((df["price"] * df["units_sold"]).sum())
    profit = revenue * 0.28  # blended margin estimate for the dashboard header
    return {
        "total_revenue": round(revenue, 0),
        "estimated_profit": round(profit, 0),
        "avg_daily_units_today": round(float(recent["units_sold"].sum()), 0),
        "num_products": int(df["item"].nunique()),
        "num_stores": int(df["store"].nunique()),
    }


@app.post("/api/simulate/price")
def simulate_price(req: PriceScenarioRequest):
    try:
        return orch.price_scenario(req.store, req.item, req.price_changes_pct)
    except Exception as e:
        raise HTTPException(400, str(e))


@app.post("/api/simulate/inventory")
def simulate_inventory(req: InventoryRequest):
    try:
        return orch.inventory_scenario(req.store, req.item, req.current_stock)
    except Exception as e:
        raise HTTPException(400, str(e))


@app.post("/api/simulate/supplier-delay")
def simulate_supplier_delay(req: SupplierDelayRequest):
    try:
        return orch.supplier_delay_scenario(req.store, req.item, req.delay_days, req.current_stock)
    except Exception as e:
        raise HTTPException(400, str(e))


@app.post("/api/simulate/transfer")
def simulate_transfer(req: TransferRequest):
    try:
        return orch.transfer_scenario(req.item, req.stores)
    except Exception as e:
        raise HTTPException(400, str(e))


@app.post("/api/policy/query")
def policy_query(req: PolicyQuery):
    return orch.policy_query(req.question)


@app.post("/api/plan/next-week")
def next_week_plan(req: NextWeekPlanRequest):
    try:
        return orch.next_week_plan(req.store)
    except Exception as e:
        raise HTTPException(400, str(e))


@app.get("/api/demand/forecast")
def demand_forecast(store: str, item: str, days: int = 7):
    try:
        return orch.demand_agent.forecast(store, item, days)
    except Exception as e:
        raise HTTPException(400, str(e))


# Serve the dashboard
frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")


@app.get("/")
def dashboard():
    return FileResponse(os.path.join(frontend_dir, "index.html"))
