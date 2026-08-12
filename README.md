# ShelfSenseAI — Agentic Retail Decision Intelligence System

A working multi-agent digital twin for retail decision-making. It doesn't just predict —
it simulates "what happens if we do X?" and recommends the best action, with reasoning
grounded in real ML forecasts and company policy.

## Architecture

```
                    USER / MANAGER
                          |
                          v
                 ORCHESTRATOR (Planner)
                          |
        +-----------------+------------------+
        v                 v                  v
   Demand Agent      Inventory Agent     Pricing Agent        Policy Agent (RAG)
   (ML forecast)     (stock coverage,    (elasticity &        (TF-IDF retrieval
                       transfers,         profit scenarios)    over policy docs)
                       supplier delay)
        \                 |                  /                        |
         \                |                 /                         |
          +----------------+----------------+-------------------------+
                           v
                    Decision Agent
              (combines outputs, checks policy,
               ranks options, explains reasoning)
                           v
                  FINAL RECOMMENDATION
```

**Key separation of concerns (this is the interview talking point):**
- **Agent** → decides *what* to do (which simulations to run, how to combine them)
- **ML model** (scikit-learn RandomForestRegressor) → calculates/predicts demand
- **Simulation engine** → pure deterministic what-if math (price scenarios, stock depletion, delay impact)
- **RAG / Policy Agent** → retrieves relevant business rules (TF-IDF cosine similarity — swap in
  sentence-transformers + FAISS for production without changing the interface)
- **Decision Agent** → the only place that combines everything into a ranked, explained recommendation
- **LLM (optional, Gemini)** → only ever rewrites the final explanation in natural language;
  it never does the math

This mirrors what the JD is actually asking for: AI applications, ML workflows, data
preparation/evaluation, APIs, databases, and retail/supply-chain use cases — not "wrap an LLM
around a spreadsheet."

## What it actually does

| Feature | Endpoint | Powered by |
|---|---|---|
| 7-day demand forecast | `GET /api/demand/forecast` | RandomForestRegressor per store×item, trained on 2 years of daily data |
| Price scenario comparison | `POST /api/simulate/price` | Demand model re-run at each hypothetical price → revenue/profit/margin |
| Inventory stock-out projection | `POST /api/simulate/inventory` | Demand forecast rolled forward against current stock, at 0/20/40% demand-surge scenarios |
| Supplier delay impact | `POST /api/simulate/supplier-delay` | Projected demand during the delay window vs. current stock |
| Store-to-store transfer recommendation | `POST /api/simulate/transfer` | Coverage-day comparison across stores |
| Policy Q&A | `POST /api/policy/query` | TF-IDF RAG over `backend/policies/policies.json` |
| Full "next week" action plan | `POST /api/plan/next-week` | Orchestrator fans out to every agent for every product in a store |

## Dataset

`data/generate_data.py` produces `data/retail_sales.csv` — 2 years of daily sales across
3 stores × 5 products (10,950 rows), with the schema used by common public retail datasets
(date, store, item, price, promo, units_sold — same shape as Kaggle's "Store Item Demand
Forecasting" / Walmart sales data). Live dataset downloads aren't reachable from the sandbox
this was built in, so the generator stands in for one, with realistic weekly/yearly
seasonality, festive-season bumps, promo effects, and genuine price elasticity baked in so
the ML model has real signal to learn (trained models hit 9–15% MAPE on held-out data — check
by re-running `python backend/models/forecast_model.py`).

**To use a real dataset instead:** point `DATA_PATH` in `backend/models/forecast_model.py` and
`backend/simulation/simulation_engine.py` at a CSV with the same columns
(`date, store, item, price, base_price, promo, units_sold`) and retrain.

## Running it

```bash
cd shelfsense-ai
pip install -r backend/requirements.txt --break-system-packages

# (data + trained models are already included — only needed if you regenerate)
python data/generate_data.py
python backend/models/forecast_model.py

# start the app
python -m uvicorn backend.main:app --reload --port 8000
```

Open `http://localhost:8000` — the dashboard is served directly by FastAPI.

### Optional: LLM-polished explanations
Set `GEMINI_API_KEY` in your environment to have the Decision Agent's reasoning rewritten in
natural language via Gemini before being returned. Without a key, the app works fully offline
using the deterministic rule-based explanations.

## Project layout

```
shelfsense-ai/
├── data/
│   ├── generate_data.py       # dataset generator
│   └── retail_sales.csv       # generated dataset
├── backend/
│   ├── main.py                 # FastAPI app + endpoints
│   ├── models/
│   │   ├── forecast_model.py   # RandomForest training + prediction
│   │   └── trained/            # saved .joblib models (15 = 3 stores × 5 items)
│   ├── simulation/
│   │   └── simulation_engine.py  # price/inventory/supplier/transfer what-if math
│   ├── agents/
│   │   ├── core_agents.py      # Demand, Inventory, Pricing agents
│   │   ├── policy_agent.py     # RAG policy agent
│   │   ├── decision_agent.py   # combines + ranks + explains
│   │   └── orchestrator.py     # routes requests, assembles next-week plan
│   ├── policies/
│   │   └── policies.json       # business rules knowledge base
│   └── requirements.txt
├── frontend/
│   └── index.html               # dashboard (vanilla HTML/CSS/JS, no build step)
└── README.md
```

## Extending toward the full JD-scope vision

- Swap the TF-IDF policy retrieval for `sentence-transformers` + a real vector DB (Chroma/FAISS)
- Replace the rule-based Orchestrator with LangGraph's `StateGraph` for explicit agent-to-agent
  message passing and conditional routing (the interfaces here are already agent-shaped, so this
  is a refactor, not a rewrite)
- Add a persistence layer (Postgres) for actual live stock levels instead of the coverage-based
  estimate used here
- Add authenticated write-back so a manager's approved recommendation actually updates
  price/reorder systems
