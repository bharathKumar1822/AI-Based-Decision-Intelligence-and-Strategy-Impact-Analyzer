"""
backend/genai/copilot.py

AI Business Copilot — natural language chat about uploaded datasets.
Features:
  - Conversation memory (per-session in-memory store)
  - Streaming SSE responses
  - Dataset context injection
  - ML result explanation
  - Chart explanation prompts
  - Business Q&A

Routes (Blueprint prefix: /api/genai):
  POST /api/genai/chat          — Main chat endpoint (streaming SSE)
  POST /api/genai/chat/sync     — Non-streaming variant
  GET  /api/genai/models        — List available Ollama models
  POST /api/genai/clear-session — Clear conversation history for a session
  GET  /api/genai/status        — Ollama health check
"""

import json
import uuid
import logging
import traceback
from datetime import datetime
from flask import Blueprint, request, jsonify, Response, stream_with_context

from .ollama_client import (
    chat,
    chat_stream,
    list_available_models,
    is_ollama_available,
    get_active_model,
    OllamaUnavailableError,
)

logger = logging.getLogger(__name__)

copilot_bp = Blueprint("copilot", __name__)

# ── In-memory conversation store ─────────────────────────────────────
# { session_id: [{"role": "...", "content": "...", "ts": "..."}, ...] }
_sessions: dict[str, list[dict]] = {}
MAX_HISTORY = 20    # Keep last 20 turns per session


# ── System Prompt Builder ────────────────────────────────────────────

def _build_system_prompt(dataset_context: dict | None = None) -> str:
    """
    Build the AI Copilot system prompt, optionally injecting dataset context.
    """
    base = (
        "You are an expert AI Business Copilot embedded in a Decision Intelligence "
        "Analytics platform. You help business analysts, executives, and data scientists "
        "understand their data, interpret ML predictions, and make strategic decisions.\n\n"
        "Your capabilities:\n"
        "- Analyze uploaded business datasets (sales, profit, orders, customers)\n"
        "- Explain machine learning model predictions in plain English\n"
        "- Interpret charts and visualizations\n"
        "- Answer business strategy questions\n"
        "- Suggest actionable improvements based on data insights\n"
        "- Identify risks, opportunities, and KPI trends\n\n"
        "Guidelines:\n"
        "- Always be concise, professional, and data-driven\n"
        "- Use numbers and specific examples from the dataset when available\n"
        "- Structure responses with clear sections when the answer is complex\n"
        "- If you're uncertain, say so — never fabricate data\n"
        "- Suggest follow-up questions to guide the user's analysis\n"
    )

    if dataset_context:
        ctx_lines = ["\n## Current Dataset Context:"]
        name = dataset_context.get("name", "Unknown")
        ctx_lines.append(f"- Dataset: **{name}**")
        if dataset_context.get("rows"):
            ctx_lines.append(f"- Records: {dataset_context['rows']:,}")
        if dataset_context.get("total_sales") is not None:
            ctx_lines.append(f"- Total Sales: ${dataset_context['total_sales']:,.2f}")
        if dataset_context.get("total_profit") is not None:
            ctx_lines.append(f"- Total Profit: ${dataset_context['total_profit']:,.2f}")
        if dataset_context.get("total_orders"):
            ctx_lines.append(f"- Total Orders: {dataset_context['total_orders']:,}")
        if dataset_context.get("column_names"):
            cols = ", ".join(dataset_context["column_names"][:15])
            ctx_lines.append(f"- Columns: {cols}")
        if dataset_context.get("best_model"):
            ctx_lines.append(f"- Best ML Model: {dataset_context['best_model']}")
        if dataset_context.get("weaknesses"):
            w = dataset_context["weaknesses"]
            if w.get("loss_making_products"):
                prods = ", ".join(list(w["loss_making_products"].keys())[:3])
                ctx_lines.append(f"- Loss-Making Sub-Categories: {prods}")
            if w.get("low_performing_regions"):
                regs = ", ".join(list(w["low_performing_regions"].keys()))
                ctx_lines.append(f"- Underperforming Regions: {regs}")
        base += "\n".join(ctx_lines)

    return base


# ── Routes ────────────────────────────────────────────────────────────

@copilot_bp.route("/status", methods=["GET"])
def genai_status():
    """Health check — returns Ollama availability and active model."""
    available = is_ollama_available()
    models    = list_available_models() if available else []
    return jsonify({
        "ollama_available": available,
        "active_model":     get_active_model() if available else None,
        "available_models": [m.get("name", "") for m in models],
        "ollama_url":       "http://localhost:11434",
        "setup_hint": (
            None if available else
            "Install Ollama from https://ollama.com, then run: ollama pull llama3"
        ),
    })


@copilot_bp.route("/models", methods=["GET"])
def list_models():
    """Return installed Ollama models."""
    if not is_ollama_available():
        return jsonify({"models": [], "error": "Ollama not running"}), 503
    models = list_available_models()
    return jsonify({
        "models":       [m.get("name", "") for m in models],
        "active_model": get_active_model(),
        "count":        len(models),
    })


@copilot_bp.route("/chat", methods=["POST"])
def chat_stream_endpoint():
    """
    Streaming SSE chat endpoint.
    Body (JSON):
      {
        "message":          str,           (required)
        "session_id":       str,           (optional — creates new if missing)
        "dataset_context":  dict,          (optional — dataset summary object)
        "model":            str,           (optional — override Ollama model)
      }
    Response: text/event-stream with SSE events:
      data: {"type": "token", "content": "..."}
      data: {"type": "done",  "session_id": "..."}
      data: {"type": "error", "content": "..."}
    """
    data           = request.get_json(silent=True) or {}
    user_message   = (data.get("message") or "").strip()
    session_id     = data.get("session_id") or str(uuid.uuid4())
    dataset_ctx    = data.get("dataset_context")
    model_override = data.get("model")

    if not user_message:
        return jsonify({"error": "message is required"}), 400

    # Retrieve or create conversation history
    if session_id not in _sessions:
        _sessions[session_id] = []
    history = _sessions[session_id]

    # Add user turn
    history.append({"role": "user", "content": user_message, "ts": _now()})

    # Build messages list for Ollama chat API
    system_prompt = _build_system_prompt(dataset_ctx)
    messages = [{"role": "system", "content": system_prompt}]

    # Add last N turns (exclude ts field for Ollama)
    for turn in history[-(MAX_HISTORY * 2):]:
        messages.append({"role": turn["role"], "content": turn["content"]})

    def sse_generator():
        full_response = []
        try:
            for token in chat_stream(
                messages=messages,
                model=model_override,
                temperature=0.7,
                max_tokens=1024,
            ):
                full_response.append(token)
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

            # Save assistant response to history
            assistant_reply = "".join(full_response)
            history.append({
                "role":    "assistant",
                "content": assistant_reply,
                "ts":      _now(),
            })

            # Trim history to avoid memory bloat
            if len(history) > MAX_HISTORY * 2:
                _sessions[session_id] = history[-(MAX_HISTORY * 2):]

            yield f"data: {json.dumps({'type': 'done', 'session_id': session_id})}\n\n"

        except OllamaUnavailableError as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
        except Exception as e:
            logger.error(f"Copilot stream error: {e}\n{traceback.format_exc()}")
            yield f"data: {json.dumps({'type': 'error', 'content': 'Internal error — check server logs'})}\n\n"

    return Response(
        stream_with_context(sse_generator()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control":  "no-cache",
            "X-Accel-Buffering": "no",     # Disable nginx buffering
        },
    )


@copilot_bp.route("/chat/sync", methods=["POST"])
def chat_sync_endpoint():
    """
    Non-streaming chat — returns complete response in one JSON.
    Useful for programmatic calls or when SSE is not supported.
    Body: same as /chat
    """
    data           = request.get_json(silent=True) or {}
    user_message   = (data.get("message") or "").strip()
    session_id     = data.get("session_id") or str(uuid.uuid4())
    dataset_ctx    = data.get("dataset_context")
    model_override = data.get("model")

    if not user_message:
        return jsonify({"error": "message is required"}), 400

    if session_id not in _sessions:
        _sessions[session_id] = []
    history = _sessions[session_id]
    history.append({"role": "user", "content": user_message, "ts": _now()})

    system_prompt = _build_system_prompt(dataset_ctx)
    messages = [{"role": "system", "content": system_prompt}]
    for turn in history[-(MAX_HISTORY * 2):]:
        messages.append({"role": turn["role"], "content": turn["content"]})

    try:
        reply = chat(
            messages=messages,
            model=model_override,
            temperature=0.7,
            max_tokens=1024,
        )
        history.append({"role": "assistant", "content": reply, "ts": _now()})
        if len(history) > MAX_HISTORY * 2:
            _sessions[session_id] = history[-(MAX_HISTORY * 2):]

        return jsonify({
            "reply":      reply,
            "session_id": session_id,
            "turn":       len(history) // 2,
        })

    except OllamaUnavailableError as e:
        return jsonify({
            "error":      str(e),
            "ollama_ok":  False,
            "session_id": session_id,
        }), 503
    except Exception as e:
        logger.error(f"Copilot sync error: {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


@copilot_bp.route("/clear-session", methods=["POST"])
def clear_session():
    """Clear conversation history for a given session."""
    data       = request.get_json(silent=True) or {}
    session_id = data.get("session_id")
    if session_id and session_id in _sessions:
        del _sessions[session_id]
    return jsonify({"cleared": True, "session_id": session_id})


@copilot_bp.route("/history/<session_id>", methods=["GET"])
def get_history(session_id: str):
    """Return conversation history for a session."""
    history = _sessions.get(session_id, [])
    return jsonify({
        "session_id": session_id,
        "turns":      len([h for h in history if h["role"] == "user"]),
        "history":    history,
    })


# ── Pre-built prompt endpoints ────────────────────────────────────────

@copilot_bp.route("/explain/dataset", methods=["POST"])
def explain_dataset():
    """
    Quick prompt: explain the loaded dataset in business terms.
    Body: {"dataset_context": dict, "session_id": str}
    """
    data    = request.get_json(silent=True) or {}
    ctx     = data.get("dataset_context", {})
    sid     = data.get("session_id") or str(uuid.uuid4())
    name    = ctx.get("name", "the dataset")
    message = (
        f"Please give me a comprehensive business overview of '{name}'. "
        "Include: what the data represents, key metrics (sales, profit, orders), "
        "notable patterns, and 3 immediate business insights I should know."
    )
    # Delegate to sync chat with dataset context
    if sid not in _sessions:
        _sessions[sid] = []
    _sessions[sid].append({"role": "user", "content": message, "ts": _now()})
    system_prompt = _build_system_prompt(ctx)
    messages = [{"role": "system", "content": system_prompt}] + [
        {"role": h["role"], "content": h["content"]}
        for h in _sessions[sid][-(MAX_HISTORY * 2):]
    ]
    try:
        reply = chat(messages=messages, temperature=0.6, max_tokens=800)
        _sessions[sid].append({"role": "assistant", "content": reply, "ts": _now()})
        return jsonify({"reply": reply, "session_id": sid})
    except OllamaUnavailableError as e:
        return jsonify({"error": str(e), "ollama_ok": False}), 503


@copilot_bp.route("/explain/ml", methods=["POST"])
def explain_ml():
    """
    Quick prompt: explain ML model results in plain English.
    Body: {"ml_results": dict, "dataset_context": dict, "session_id": str}
    """
    data    = request.get_json(silent=True) or {}
    ml      = data.get("ml_results", {})
    ctx     = data.get("dataset_context", {})
    sid     = data.get("session_id") or str(uuid.uuid4())

    best    = ml.get("best_model", "Random Forest")
    bp      = ml.get("best_predicted_profit", 0)
    message = (
        f"Our ML analysis selected '{best}' as the best profit prediction model "
        f"with a predicted average profit of ${bp:,.2f}. "
        "Explain in simple business terms why this model was chosen, "
        "what the predicted profit means for the business, "
        "and what actions the business should take based on this prediction."
    )
    if sid not in _sessions:
        _sessions[sid] = []
    _sessions[sid].append({"role": "user", "content": message, "ts": _now()})
    system_prompt = _build_system_prompt(ctx)
    messages = [{"role": "system", "content": system_prompt}] + [
        {"role": h["role"], "content": h["content"]}
        for h in _sessions[sid][-(MAX_HISTORY * 2):]
    ]
    try:
        reply = chat(messages=messages, temperature=0.65, max_tokens=700)
        _sessions[sid].append({"role": "assistant", "content": reply, "ts": _now()})
        return jsonify({"reply": reply, "session_id": sid})
    except OllamaUnavailableError as e:
        return jsonify({"error": str(e), "ollama_ok": False}), 503


# ── Helpers ───────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
