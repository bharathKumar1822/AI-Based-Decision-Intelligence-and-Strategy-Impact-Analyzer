"""
backend/genai/agents.py

Multi-Agent Analysis System for Decision Intelligence.
5 parallel specialist agents analyze the dataset from their domain perspective.
CEO Agent synthesizes all insights into a final unified strategy.

Agents:
  1. Sales Agent     — Revenue trends, top products, growth opportunities
  2. Finance Agent   — Profit margins, cost analysis, ROI, financial risks
  3. Marketing Agent — Customer segments, retention, campaigns, CLV
  4. Operations Agent— Efficiency, logistics, supply chain, process improvements
  5. CEO Agent       — Synthesizes all 4 agent outputs into final strategy

Routes (Blueprint prefix: /api/agents):
  POST /api/agents/analyze         — Run full multi-agent analysis (blocking)
  POST /api/agents/analyze/stream  — Run with SSE progress streaming
  GET  /api/agents/cache/<name>    — Retrieve cached results
"""

import json
import logging
import traceback
import threading
from datetime import datetime
from typing import Optional

from flask import Blueprint, request, jsonify, Response, stream_with_context

from .ollama_client import generate, OllamaUnavailableError, get_active_model
from backend.utils.cache import strategy_cache

logger = logging.getLogger(__name__)

agents_bp = Blueprint("agents", __name__)


# ── Agent Definitions ────────────────────────────────────────────────

AGENT_SPECS = {
    "sales": {
        "name":  "Sales Agent",
        "icon":  "📈",
        "color": "#6C63FF",
        "role":  "Senior Sales Analyst",
        "focus": (
            "Analyze sales performance, revenue trends, top-performing products, "
            "seasonal patterns, and growth opportunities. Focus on revenue maximization "
            "strategies, sales channel optimization, and pricing opportunities."
        ),
        "output_keys": [
            "revenue_trend", "top_products", "growth_opportunities",
            "pricing_strategy", "sales_risks", "quick_wins"
        ],
    },
    "finance": {
        "name":  "Finance Agent",
        "icon":  "💰",
        "color": "#4CAF50",
        "role":  "Chief Financial Officer",
        "focus": (
            "Analyze profit margins, cost structures, ROI of different segments, "
            "financial risks, cash flow patterns, and budget allocation recommendations. "
            "Identify margin erosion, over-discounting, and profitability leaks."
        ),
        "output_keys": [
            "profit_margins", "cost_analysis", "roi_assessment",
            "financial_risks", "budget_recommendations", "margin_recovery"
        ],
    },
    "marketing": {
        "name":  "Marketing Agent",
        "icon":  "🎯",
        "color": "#FF6584",
        "role":  "Chief Marketing Officer",
        "focus": (
            "Analyze customer segments, purchase behavior, retention rates, "
            "lifetime value (CLV), churn signals, and campaign effectiveness. "
            "Recommend targeted campaigns, loyalty programs, and customer acquisition strategies."
        ),
        "output_keys": [
            "customer_segments", "retention_strategy", "campaign_recommendations",
            "clv_improvement", "churn_prevention", "market_opportunities"
        ],
    },
    "operations": {
        "name":  "Operations Agent",
        "icon":  "⚙️",
        "color": "#F7971E",
        "role":  "Chief Operations Officer",
        "focus": (
            "Analyze operational efficiency, supply chain performance, logistics costs, "
            "shipping modes, regional distribution patterns, and process bottlenecks. "
            "Recommend process improvements, automation opportunities, and cost reductions."
        ),
        "output_keys": [
            "efficiency_gaps", "supply_chain_risks", "process_improvements",
            "logistics_optimization", "automation_opportunities", "cost_reductions"
        ],
    },
}

CEO_SPEC = {
    "name":  "CEO Agent",
    "icon":  "👔",
    "color": "#FFD700",
    "role":  "Chief Executive Officer",
    "focus": (
        "Synthesize the insights from all specialist agents into a unified, "
        "prioritized strategic plan. Resolve conflicts between agent recommendations, "
        "set priorities based on ROI and feasibility, define a 90-day action plan, "
        "and provide a confidence-weighted executive strategy."
    ),
}


# ── Dataset Context Builder ─────────────────────────────────────────

def _build_agent_context(dataset_summary: dict) -> str:
    """Build a compact dataset description for agent prompts."""
    lines = [
        "=== DATASET CONTEXT ===",
        f"Name: {dataset_summary.get('name', 'Unknown')}",
        f"Records: {dataset_summary.get('rows', 0):,}",
        f"Columns: {', '.join(dataset_summary.get('column_names', []))}",
    ]
    if dataset_summary.get("total_sales"):
        lines.append(f"Total Sales: ${dataset_summary['total_sales']:,.2f}")
    if dataset_summary.get("total_profit") is not None:
        lines.append(f"Total Profit: ${dataset_summary['total_profit']:,.2f}")
    if dataset_summary.get("total_orders"):
        lines.append(f"Total Orders: {dataset_summary['total_orders']:,}")
    if dataset_summary.get("profit_margin"):
        lines.append(f"Profit Margin: {dataset_summary['profit_margin']:.1f}%")
    if dataset_summary.get("best_model"):
        lines.append(f"Best ML Model: {dataset_summary['best_model']}")
    if dataset_summary.get("weaknesses"):
        w = dataset_summary["weaknesses"]
        if w.get("loss_making_products"):
            prods = ", ".join(list(w["loss_making_products"].keys())[:5])
            losses = sum(abs(v) for v in w["loss_making_products"].values())
            lines.append(f"Loss-Making Sub-Categories: {prods} (total loss: ${losses:,.2f})")
        if w.get("low_performing_regions"):
            regs = ", ".join(list(w["low_performing_regions"].keys()))
            lines.append(f"Underperforming Regions: {regs}")
        if w.get("poor_profit_margins"):
            cats = ", ".join(list(w["poor_profit_margins"].keys()))
            lines.append(f"Low-Margin Categories: {cats}")
    return "\n".join(lines)


# ── Single Agent Runner ──────────────────────────────────────────────

def _run_agent(
    agent_key: str,
    spec: dict,
    dataset_context: str,
    model: Optional[str] = None,
) -> dict:
    """
    Run a single specialist agent. Returns structured analysis dict.
    """
    prompt = (
        f"You are a {spec['role']} specializing in {spec['focus']}\n\n"
        f"{dataset_context}\n\n"
        f"Provide a detailed analysis from the perspective of your role. "
        f"Structure your response with the following sections:\n"
        f"1. Key Observations (3-5 bullet points)\n"
        f"2. Critical Issues (what needs immediate attention)\n"
        f"3. Strategic Recommendations (3 specific, actionable recommendations with expected impact)\n"
        f"4. Quick Wins (actions achievable in 30 days)\n"
        f"5. Risk Warning (1-2 risks to watch)\n\n"
        f"Be specific, quantitative where possible, and business-focused."
    )

    try:
        raw = generate(
            prompt=prompt,
            model=model,
            temperature=0.65,
            max_tokens=800,
        )
        return {
            "agent":      agent_key,
            "name":       spec["name"],
            "icon":       spec["icon"],
            "color":      spec["color"],
            "role":       spec["role"],
            "analysis":   raw,
            "status":     "success",
            "generated_at": _now(),
        }
    except OllamaUnavailableError as e:
        return {
            "agent":    agent_key,
            "name":     spec["name"],
            "icon":     spec["icon"],
            "color":    spec["color"],
            "role":     spec["role"],
            "analysis": f"⚠️ Agent unavailable: {str(e)}",
            "status":   "error",
            "error":    str(e),
        }
    except Exception as e:
        logger.error(f"Agent {agent_key} error: {e}\n{traceback.format_exc()}")
        return {
            "agent":    agent_key,
            "name":     spec["name"],
            "icon":     spec["icon"],
            "color":    spec["color"],
            "role":     spec["role"],
            "analysis": f"Agent encountered an error: {str(e)}",
            "status":   "error",
            "error":    str(e),
        }


def _run_ceo(agent_results: list[dict], dataset_context: str, model: Optional[str] = None) -> dict:
    """
    CEO Agent: synthesize all specialist agent outputs.
    """
    agent_summaries = "\n\n".join([
        f"=== {r['name']} ===\n{r['analysis']}"
        for r in agent_results
        if r["status"] == "success"
    ])

    prompt = (
        f"You are the CEO of a data-driven organization. Your specialist agents have provided "
        f"the following analyses:\n\n{agent_summaries}\n\n"
        f"{dataset_context}\n\n"
        f"As CEO, synthesize these insights into a UNIFIED STRATEGY with:\n"
        f"1. Executive Summary (2-3 sentences)\n"
        f"2. Top 3 Strategic Priorities (ranked by impact, with rationale)\n"
        f"3. 90-Day Action Plan (specific milestones per month)\n"
        f"4. Resource Allocation Recommendation\n"
        f"5. Expected ROI (quantified estimate)\n"
        f"6. Risk Mitigation Plan\n"
        f"7. Confidence Level: X/100 and why\n\n"
        f"Be decisive, strategic, and business-outcome focused."
    )

    try:
        raw = generate(
            prompt=prompt,
            model=model,
            temperature=0.5,
            max_tokens=1200,
        )
        return {
            "agent":      "ceo",
            "name":       CEO_SPEC["name"],
            "icon":       CEO_SPEC["icon"],
            "color":      CEO_SPEC["color"],
            "role":       CEO_SPEC["role"],
            "analysis":   raw,
            "status":     "success",
            "generated_at": _now(),
        }
    except OllamaUnavailableError as e:
        return {
            "agent":    "ceo",
            "name":     CEO_SPEC["name"],
            "icon":     CEO_SPEC["icon"],
            "color":    CEO_SPEC["color"],
            "role":     CEO_SPEC["role"],
            "analysis": f"⚠️ CEO synthesis unavailable: {str(e)}",
            "status":   "error",
            "error":    str(e),
        }


# ── Routes ────────────────────────────────────────────────────────────

@agents_bp.route("/analyze", methods=["POST"])
def analyze():
    """
    Run full multi-agent analysis (blocking, sequential).
    Body (JSON):
      {
        "dataset_summary": dict,   (required — dataset summary object)
        "model": str,              (optional — Ollama model)
        "use_cache": bool,         (optional, default true)
      }
    """
    data            = request.get_json(silent=True) or {}
    dataset_summary = data.get("dataset_summary", {})
    model           = data.get("model")
    use_cache       = data.get("use_cache", True)

    dataset_name = dataset_summary.get("name", "unknown")

    # Check cache
    cache_key = f"agents_{dataset_name}"
    if use_cache:
        cached = strategy_cache.get(cache_key)
        if cached:
            cached["from_cache"] = True
            return jsonify(cached)

    dataset_context = _build_agent_context(dataset_summary)

    # Run all 4 specialist agents in parallel
    agent_results = [None] * len(AGENT_SPECS)
    threads       = []

    def _run_and_store(idx, key, spec):
        agent_results[idx] = _run_agent(key, spec, dataset_context, model)

    for i, (key, spec) in enumerate(AGENT_SPECS.items()):
        t = threading.Thread(target=_run_and_store, args=(i, key, spec), daemon=True)
        threads.append(t)
        t.start()

    for t in threads:
        t.join(timeout=180)

    # Fill any None results (timed out threads)
    for i, (key, spec) in enumerate(AGENT_SPECS.items()):
        if agent_results[i] is None:
            agent_results[i] = {
                "agent": key, "name": spec["name"], "icon": spec["icon"],
                "color": spec["color"], "role": spec["role"],
                "analysis": "Agent timed out.", "status": "timeout",
            }

    # Run CEO agent
    ceo_result = _run_ceo(agent_results, dataset_context, model)

    result = {
        "agents":         agent_results,
        "ceo":            ceo_result,
        "dataset":        dataset_name,
        "model_used":     model or get_active_model(),
        "generated_at":   _now(),
        "from_cache":     False,
        "total_agents":   len(AGENT_SPECS) + 1,
    }

    # Cache the result
    strategy_cache.set(cache_key, result, ttl=600)

    return jsonify(result)


@agents_bp.route("/analyze/stream", methods=["POST"])
def analyze_stream():
    """
    Streaming multi-agent analysis with SSE progress updates.
    Same body as /analyze.
    SSE events:
      {"type": "progress", "agent": "sales", "status": "running"}
      {"type": "agent_done", "agent": "sales", "result": {...}}
      {"type": "ceo_start"}
      {"type": "ceo_done", "result": {...}}
      {"type": "done", "summary": {...}}
    """
    data            = request.get_json(silent=True) or {}
    dataset_summary = data.get("dataset_summary", {})
    model           = data.get("model")
    dataset_name    = dataset_summary.get("name", "unknown")
    dataset_context = _build_agent_context(dataset_summary)

    def sse_gen():
        agent_results = []

        # Run agents one by one (sequential for SSE feedback)
        for key, spec in AGENT_SPECS.items():
            yield f"data: {json.dumps({'type': 'progress', 'agent': key, 'name': spec['name'], 'status': 'running'})}\n\n"
            result = _run_agent(key, spec, dataset_context, model)
            agent_results.append(result)
            yield f"data: {json.dumps({'type': 'agent_done', 'agent': key, 'result': result})}\n\n"

        # CEO synthesis
        yield f"data: {json.dumps({'type': 'ceo_start'})}\n\n"
        ceo = _run_ceo(agent_results, dataset_context, model)
        yield f"data: {json.dumps({'type': 'ceo_done', 'result': ceo})}\n\n"

        # Final done
        summary = {
            "agents":       agent_results,
            "ceo":          ceo,
            "dataset":      dataset_name,
            "generated_at": _now(),
        }
        strategy_cache.set(f"agents_{dataset_name}", summary, ttl=600)
        yield f"data: {json.dumps({'type': 'done', 'summary': summary})}\n\n"

    return Response(
        stream_with_context(sse_gen()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@agents_bp.route("/cache/<dataset_name>", methods=["GET"])
def get_cached(dataset_name: str):
    """Return cached agent analysis for a dataset."""
    cached = strategy_cache.get(f"agents_{dataset_name}")
    if cached:
        return jsonify(cached)
    return jsonify({"error": "No cached analysis found"}), 404


# ── Helpers ───────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
