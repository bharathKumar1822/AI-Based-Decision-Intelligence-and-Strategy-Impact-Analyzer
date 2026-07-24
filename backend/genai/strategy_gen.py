"""
backend/genai/strategy_gen.py

AI Strategy Generator — produces structured business strategies from dataset analysis.
Output includes: strategies, root-cause analysis, risks, ROI, priority, confidence,
timeline, and a concrete action plan.

Routes (Blueprint prefix: /api/genai/strategy):
  POST /api/genai/strategy/generate   — Generate strategy from dataset
  POST /api/genai/strategy/refine     — Refine/customize existing strategy
"""

import json
import logging
import traceback
from datetime import datetime
from typing import Optional

from flask import Blueprint, request, jsonify

from .ollama_client import generate, OllamaUnavailableError, get_active_model, extract_json_from_response
from backend.utils.cache import strategy_cache

logger = logging.getLogger(__name__)

strategy_bp = Blueprint("strategy_gen", __name__)

FOCUS_AREAS = {"sales", "marketing", "finance", "operations", "overall"}


# ── Routes ────────────────────────────────────────────────────────────

@strategy_bp.route("/generate", methods=["POST"])
def generate_strategy():
    """
    Generate a comprehensive AI business strategy from dataset metrics.

    Body (JSON):
      {
        "dataset_summary":  dict,       (required)
        "focus_area":       str,        (optional: sales/marketing/finance/operations/overall)
        "model":            str,        (optional)
        "use_cache":        bool,       (default true)
      }

    Returns:
      {
        "strategies":          [{"title", "description", "expected_impact", "timeline", "priority"}, ...],
        "root_cause_analysis": str,
        "risks":               [{"risk", "likelihood", "mitigation"}, ...],
        "roi_estimate":        str,
        "priority":            "HIGH|MEDIUM|LOW",
        "confidence":          int (0-100),
        "timeline":            str,
        "action_plan":         [{"month": 1, "actions": [...]}, ...],
        "executive_summary":   str,
      }
    """
    data            = request.get_json(silent=True) or {}
    dataset_summary = data.get("dataset_summary", {})
    focus           = (data.get("focus_area") or "overall").lower()
    model           = data.get("model")
    use_cache       = data.get("use_cache", True)

    if focus not in FOCUS_AREAS:
        focus = "overall"

    dataset_name = dataset_summary.get("name", "unknown")
    cache_key    = f"strategy_{dataset_name}_{focus}"

    if use_cache:
        cached = strategy_cache.get(cache_key)
        if cached:
            cached["from_cache"] = True
            return jsonify(cached)

    # Build context
    ctx = _build_context(dataset_summary, focus)

    # Prompt for structured JSON strategy
    prompt = f"""You are an expert business strategist analyzing a company's data.

{ctx}

Generate a comprehensive {focus} strategy. Return your response as a JSON object with EXACTLY this structure:

{{
  "executive_summary": "2-3 sentence overview of the strategic situation and opportunity",
  "strategies": [
    {{
      "title": "Strategy name",
      "description": "What to do and how",
      "expected_impact": "Quantified impact (e.g., +15% profit)",
      "timeline": "30/60/90 days",
      "priority": "HIGH/MEDIUM/LOW"
    }}
  ],
  "root_cause_analysis": "Detailed explanation of the core business problems driving current performance",
  "risks": [
    {{
      "risk": "Risk description",
      "likelihood": "High/Medium/Low",
      "mitigation": "How to mitigate it"
    }}
  ],
  "roi_estimate": "Estimated ROI with timeframe (e.g., 180% ROI in 12 months)",
  "priority": "HIGH",
  "confidence": 85,
  "timeline": "Overall implementation timeline",
  "action_plan": [
    {{"month": 1, "theme": "Foundation", "actions": ["Action 1", "Action 2", "Action 3"]}},
    {{"month": 2, "theme": "Execution", "actions": ["Action 1", "Action 2", "Action 3"]}},
    {{"month": 3, "theme": "Optimization", "actions": ["Action 1", "Action 2", "Action 3"]}}
  ]
}}

Return ONLY valid JSON. No markdown, no extra text."""

    try:
        raw = generate(prompt=prompt, model=model, temperature=0.4, max_tokens=2000)

        # Try to parse structured JSON
        parsed = extract_json_from_response(raw)
        if parsed and _validate_strategy_json(parsed):
            result = {**parsed, "from_cache": False, "model_used": model or get_active_model(),
                      "generated_at": _now(), "dataset": dataset_name, "focus_area": focus}
        else:
            # Fallback: wrap raw text in structured format
            result = _fallback_strategy(raw, dataset_name, focus, model)

        strategy_cache.set(cache_key, result, ttl=600)
        return jsonify(result)

    except OllamaUnavailableError as e:
        return jsonify({
            "error": str(e),
            "ollama_ok": False,
            "hint": "Install Ollama from https://ollama.com, then run: ollama pull llama3",
        }), 503
    except Exception as e:
        logger.error(f"Strategy generation error: {e}\n{traceback.format_exc()}")
        return jsonify({"error": f"Strategy generation failed: {str(e)}"}), 500


@strategy_bp.route("/refine", methods=["POST"])
def refine_strategy():
    """
    Refine an existing strategy with user feedback or additional constraints.

    Body (JSON):
      {
        "existing_strategy": dict,   (required — previous generate output)
        "refinement":        str,    (required — user instructions)
        "model":             str,    (optional)
      }
    """
    data              = request.get_json(silent=True) or {}
    existing          = data.get("existing_strategy", {})
    refinement        = (data.get("refinement") or "").strip()
    model             = data.get("model")

    if not refinement:
        return jsonify({"error": "refinement instructions are required"}), 400

    existing_text = json.dumps(existing, indent=2) if existing else "No existing strategy provided."

    prompt = (
        f"You are refining a business strategy based on user feedback.\n\n"
        f"EXISTING STRATEGY:\n{existing_text}\n\n"
        f"USER REFINEMENT REQUEST:\n{refinement}\n\n"
        f"Update the strategy incorporating the user's feedback. "
        f"Return the complete refined strategy in the same JSON format as the original."
    )

    try:
        raw    = generate(prompt=prompt, model=model, temperature=0.4, max_tokens=2000)
        parsed = extract_json_from_response(raw)
        if parsed and _validate_strategy_json(parsed):
            return jsonify({**parsed, "refined": True, "refinement": refinement,
                           "generated_at": _now()})
        return jsonify({"refined_text": raw, "refined": True, "generated_at": _now()})
    except OllamaUnavailableError as e:
        return jsonify({"error": str(e), "ollama_ok": False}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Helpers ───────────────────────────────────────────────────────────

def _build_context(summary: dict, focus: str) -> str:
    lines = [f"=== BUSINESS DATA CONTEXT ===", f"Focus Area: {focus.upper()}",
             f"Dataset: {summary.get('name', 'Unknown')}",
             f"Records: {summary.get('rows', 0):,}"]

    if summary.get("total_sales"):
        lines.append(f"Total Sales: ${summary['total_sales']:,.2f}")
    if summary.get("total_profit") is not None:
        lines.append(f"Total Profit: ${summary['total_profit']:,.2f}")
    if summary.get("profit_margin"):
        lines.append(f"Profit Margin: {summary['profit_margin']:.1f}%")
    if summary.get("total_orders"):
        lines.append(f"Total Orders: {summary['total_orders']:,}")
    if summary.get("best_model"):
        lines.append(f"Best Predictive Model: {summary['best_model']}")

    w = summary.get("weaknesses", {})
    if w.get("loss_making_products"):
        prods  = list(w["loss_making_products"].keys())[:5]
        losses = sum(abs(v) for v in w["loss_making_products"].values())
        lines.append(f"Loss-Making Products: {', '.join(prods)} (total loss: ${losses:,.2f})")
    if w.get("low_performing_regions"):
        regs = list(w["low_performing_regions"].keys())
        lines.append(f"Underperforming Regions: {', '.join(regs)}")
    if w.get("poor_profit_margins"):
        cats = list(w["poor_profit_margins"].keys())
        lines.append(f"Low-Margin Categories (<10%): {', '.join(cats)}")

    return "\n".join(lines)


def _validate_strategy_json(obj: dict) -> bool:
    """Check that the parsed JSON has the minimum required keys."""
    required = {"executive_summary", "strategies", "root_cause_analysis", "action_plan"}
    return bool(obj) and required.issubset(obj.keys())


def _fallback_strategy(raw: str, dataset_name: str, focus: str, model: Optional[str]) -> dict:
    """Wrap a non-JSON Ollama response into a consistent structure."""
    return {
        "executive_summary":   "AI strategy generated (text format).",
        "strategies":          [{"title": "AI Generated Strategy", "description": raw,
                                 "expected_impact": "See analysis", "timeline": "90 days",
                                 "priority": "HIGH"}],
        "root_cause_analysis": "See AI analysis above.",
        "risks":               [],
        "roi_estimate":        "See analysis",
        "priority":            "HIGH",
        "confidence":          70,
        "timeline":            "90 days",
        "action_plan":         [{"month": 1, "theme": "Implementation", "actions": ["Review AI analysis and prioritize"]}],
        "raw_text":            raw,
        "from_cache":          False,
        "model_used":          model or get_active_model(),
        "generated_at":        _now(),
        "dataset":             dataset_name,
        "focus_area":          focus,
    }


def _now() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
