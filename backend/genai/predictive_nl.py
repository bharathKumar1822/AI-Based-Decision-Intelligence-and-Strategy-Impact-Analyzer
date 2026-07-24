"""
backend/genai/predictive_nl.py

Natural Language Forecasting — parse NL queries and simulate business scenarios.
Examples:
  "What happens if sales increase by 15%?"
  "What if we reduce discounts to 20%?"
  "Predict profit if we add 500 new customers"

Routes (Blueprint prefix: /api/genai/forecast):
  POST /api/genai/forecast/query    — NL forecast query
  POST /api/genai/forecast/parse    — Parse NL intent (debug utility)
  GET  /api/genai/forecast/examples — Return example queries
"""

import re
import logging
import traceback
from datetime import datetime
from typing import Optional

from flask import Blueprint, request, jsonify

from .ollama_client import generate, OllamaUnavailableError, get_active_model, extract_json_from_response

logger = logging.getLogger(__name__)

forecast_bp = Blueprint("forecast", __name__)


# ── Pre-defined scenario templates ───────────────────────────────────

EXAMPLE_QUERIES = [
    "What happens if sales increase by 15%?",
    "What if we reduce discounts to 20%?",
    "Predict profit if sales drop by 10% next quarter",
    "How much profit would we make if we eliminated all loss-making products?",
    "What if we expand to 2 new regions?",
    "What happens if we improve profit margins by 5%?",
    "Predict the impact of a 25% increase in marketing spend",
    "What if customer retention increases by 10%?",
]

# ── Route ─────────────────────────────────────────────────────────────

@forecast_bp.route("/examples", methods=["GET"])
def get_examples():
    """Return example forecast queries for UI suggestion chips."""
    return jsonify({"examples": EXAMPLE_QUERIES})


@forecast_bp.route("/parse", methods=["POST"])
def parse_intent():
    """
    Debug endpoint: parse NL query intent without running simulation.
    Body: {"query": str}
    """
    data  = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"error": "query is required"}), 400
    parsed = _parse_query(query)
    return jsonify({"query": query, "parsed": parsed})


@forecast_bp.route("/query", methods=["POST"])
def forecast_query():
    """
    Natural language forecasting endpoint.

    Body (JSON):
      {
        "query":           str,    (required — NL forecast question)
        "dataset_summary": dict,   (required — from /api/summary/<name>)
        "ml_results":      dict,   (optional — from /api/predict/<name>)
        "model":           str,    (optional — Ollama model)
      }

    Returns:
      {
        "query":           str,
        "interpretation":  str,    (what the AI understood)
        "simulation":      dict,   (numeric simulation results)
        "narrative":       str,    (AI-narrated result in plain English)
        "recommendations": [str],  (follow-up actions)
        "confidence":      int,
      }
    """
    data            = request.get_json(silent=True) or {}
    query           = (data.get("query") or "").strip()
    dataset_summary = data.get("dataset_summary", {})
    ml_results      = data.get("ml_results", {})
    model           = data.get("model")

    if not query:
        return jsonify({"error": "query is required"}), 400

    # Extract key metrics from dataset summary
    total_sales    = float(dataset_summary.get("total_sales")  or 0)
    total_profit   = float(dataset_summary.get("total_profit") or 0)
    total_orders   = int(dataset_summary.get("total_orders")   or 0)
    dataset_name   = dataset_summary.get("name", "the dataset")
    best_model     = ml_results.get("best_model", dataset_summary.get("best_model", "Random Forest"))
    best_profit    = float(ml_results.get("best_predicted_profit") or total_profit or 0)

    # Parse the NL query to extract scenario parameters
    parsed = _parse_query(query)
    metric    = parsed.get("metric", "sales")
    direction = parsed.get("direction", "increase")
    pct       = parsed.get("percentage", 10.0)
    absolute  = parsed.get("absolute")

    # ── Run numeric simulation ─────────────────────────────────────
    sim = _run_simulation(
        metric=metric,
        direction=direction,
        pct=pct,
        absolute=absolute,
        total_sales=total_sales,
        total_profit=total_profit,
        total_orders=total_orders,
        best_profit=best_profit,
    )

    # ── Build AI narrative ────────────────────────────────────────
    context = (
        f"Dataset: {dataset_name}\n"
        f"Current Total Sales: ${total_sales:,.2f}\n"
        f"Current Total Profit: ${total_profit:,.2f}\n"
        f"Current Orders: {total_orders:,}\n"
        f"Best ML Model: {best_model}\n"
        f"Best Predicted Profit: ${best_profit:,.2f}\n\n"
        f"USER QUESTION: {query}\n\n"
        f"SIMULATION RESULTS:\n"
        f"- Scenario: {sim.get('scenario_description', '')}\n"
        f"- Projected Sales: ${sim.get('projected_sales', 0):,.2f} "
        f"({'+'if sim.get('sales_delta',0)>=0 else ''}{sim.get('sales_delta', 0):,.2f})\n"
        f"- Projected Profit: ${sim.get('projected_profit', 0):,.2f} "
        f"({'+'if sim.get('profit_delta',0)>=0 else ''}{sim.get('profit_delta', 0):,.2f})\n"
        f"- Percentage Change: {sim.get('pct_change', 0):+.1f}%\n"
    )

    prompt = (
        f"You are a business intelligence analyst. A user asked a what-if forecasting question. "
        f"Based on the data and simulation results below, provide:\n"
        f"1. A clear interpretation of what the user is asking\n"
        f"2. A business-friendly explanation of the simulated outcome\n"
        f"3. The key implications for the business\n"
        f"4. 3 specific action recommendations\n"
        f"5. Important caveats or assumptions\n\n"
        f"{context}\n"
        f"Be concise, specific, and use the actual numbers from the simulation."
    )

    narrative    = ""
    ollama_ok    = True
    try:
        narrative = generate(prompt=prompt, model=model, temperature=0.55, max_tokens=700)
    except OllamaUnavailableError as e:
        narrative = (
            f"📊 Simulation Results:\n"
            f"If {query.lower()}, projected profit would be ${sim.get('projected_profit', 0):,.2f} "
            f"(change: {'+'if sim.get('profit_delta',0)>=0 else ''}{sim.get('profit_delta', 0):,.2f}). "
            f"Note: AI narration unavailable — {str(e)}"
        )
        ollama_ok = False
    except Exception as e:
        logger.error(f"Forecast narrative error: {e}")
        narrative = f"Simulation complete. AI narrative generation failed: {str(e)}"
        ollama_ok = False

    return jsonify({
        "query":          query,
        "interpretation": f"Simulating: {sim.get('scenario_description', query)}",
        "simulation":     sim,
        "narrative":      narrative,
        "ollama_ok":      ollama_ok,
        "confidence":     sim.get("confidence", 70),
        "generated_at":   _now(),
        "dataset":        dataset_name,
    })


# ── NL Parser ─────────────────────────────────────────────────────────

def _parse_query(query: str) -> dict:
    """
    Rule-based NL parser to extract scenario parameters from a forecast query.
    Returns: {metric, direction, percentage, absolute}
    """
    q = query.lower()

    # Direction
    direction = "increase"
    if any(w in q for w in ["decrease", "drop", "fall", "reduce", "decline", "cut", "less"]):
        direction = "decrease"

    # Percentage
    pct_match = re.search(r"(\d+(?:\.\d+)?)\s*%", q)
    pct = float(pct_match.group(1)) if pct_match else 10.0

    # Absolute number
    abs_match = re.search(r"(\d{1,3}(?:,\d{3})*|\d+)\s+(?:customers?|orders?|units?|products?)", q)
    absolute = int(abs_match.group(1).replace(",", "")) if abs_match else None

    # Metric
    metric = "sales"
    if any(w in q for w in ["profit", "margin", "earnings"]):
        metric = "profit"
    elif any(w in q for w in ["customer", "retention", "churn"]):
        metric = "customers"
    elif any(w in q for w in ["discount", "price", "pricing"]):
        metric = "discount"
    elif any(w in q for w in ["order", "volume", "units"]):
        metric = "orders"
    elif any(w in q for w in ["market", "region", "expand"]):
        metric = "market"
    elif any(w in q for w in ["cost", "expense", "spend"]):
        metric = "cost"

    return {
        "metric":     metric,
        "direction":  direction,
        "percentage": pct,
        "absolute":   absolute,
    }


def _run_simulation(
    metric: str,
    direction: str,
    pct: float,
    absolute: Optional[int],
    total_sales: float,
    total_profit: float,
    total_orders: int,
    best_profit: float,
) -> dict:
    """
    Numeric simulation engine.
    Returns projected metrics and delta values.
    """
    multiplier = (1 + pct / 100) if direction == "increase" else (1 - pct / 100)
    multiplier = max(0.01, multiplier)

    proj_sales  = total_sales
    proj_profit = total_profit

    scenario_desc = f"{direction.capitalize()} {metric} by {pct:.0f}%"

    if metric == "sales":
        proj_sales  = total_sales  * multiplier
        # Profit scales roughly with sales (simplified)
        proj_profit = total_profit * multiplier * 0.9   # 90% coefficient (non-linear)

    elif metric == "profit":
        proj_profit = total_profit * multiplier
        # Sales stays roughly same
        proj_sales  = total_sales

    elif metric == "discount":
        # Reducing discounts → improve margins
        margin_effect = 0.3 if direction == "decrease" else -0.3
        proj_profit   = total_profit * (1 + (pct / 100) * margin_effect)
        proj_sales    = total_sales  * (1 - (pct / 200))  # slight sales impact

    elif metric == "customers":
        if absolute:
            # Fixed number of new customers
            avg_order_value = total_sales / max(total_orders, 1)
            extra_sales     = absolute * avg_order_value * 3   # avg 3 orders/customer
            proj_sales      = total_sales  + extra_sales
            proj_profit     = total_profit + extra_sales * (total_profit / max(total_sales, 1))
            scenario_desc   = f"Add {absolute:,} customers"
        else:
            proj_sales  = total_sales  * multiplier
            proj_profit = total_profit * multiplier * 0.85

    elif metric == "orders":
        if absolute:
            avg_order_value = total_sales / max(total_orders, 1)
            avg_order_profit = total_profit / max(total_orders, 1)
            proj_sales  = total_sales  + absolute * avg_order_value
            proj_profit = total_profit + absolute * avg_order_profit
            scenario_desc = f"Add {absolute:,} orders"
        else:
            proj_sales  = total_sales  * multiplier
            proj_profit = total_profit * multiplier

    elif metric == "market":
        proj_sales  = total_sales  * multiplier
        proj_profit = total_profit * multiplier * 0.7   # expansion has higher costs

    elif metric == "cost":
        # Reducing costs → improve profit without changing sales
        cost_savings = total_profit * (pct / 100) * (0.5 if direction == "decrease" else -0.5)
        proj_profit  = total_profit + cost_savings
        proj_sales   = total_sales

    # Derived metrics
    sales_delta  = proj_sales  - total_sales
    profit_delta = proj_profit - total_profit
    pct_change   = (profit_delta / abs(total_profit) * 100) if total_profit else 0
    profit_margin_proj = (proj_profit / proj_sales * 100) if proj_sales else 0
    confidence   = 85 if abs(pct) <= 20 else 70 if abs(pct) <= 40 else 55

    return {
        "scenario_description": scenario_desc,
        "metric":               metric,
        "direction":            direction,
        "change_pct":           pct,
        "projected_sales":      round(proj_sales,  2),
        "projected_profit":     round(proj_profit, 2),
        "current_sales":        round(total_sales, 2),
        "current_profit":       round(total_profit, 2),
        "sales_delta":          round(sales_delta,  2),
        "profit_delta":         round(profit_delta, 2),
        "pct_change":           round(pct_change,   2),
        "projected_margin":     round(profit_margin_proj, 2),
        "confidence":           confidence,
    }


def _now() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
