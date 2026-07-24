"""
backend/genai/ollama_client.py

Ollama HTTP client wrapper for Decision Intelligence Analyzer.
Supports: llama3, mistral, gemma, qwen, deepseek-r1
Auto-selects best available model.
Provides both blocking and streaming generation.
"""

import os
import json
import logging
import requests
from typing import Generator, Optional

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────
OLLAMA_URL   = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3")
TIMEOUT      = int(os.environ.get("OLLAMA_TIMEOUT", "120"))

# Ordered preference list — first available model wins during auto-select
PREFERRED_MODELS = [
    "llama3",
    "llama3.1",
    "mistral",
    "gemma",
    "gemma2",
    "qwen",
    "qwen2",
    "deepseek-r1",
    "phi3",
    "neural-chat",
]


# ── Health & Model Discovery ─────────────────────────────────────────

def is_ollama_available() -> bool:
    """Return True if Ollama is running and reachable."""
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def list_available_models() -> list[dict]:
    """
    Return list of models installed in Ollama.
    Each entry: {"name": str, "size": int, "modified_at": str}
    Returns [] if Ollama is not reachable.
    """
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if r.status_code == 200:
            return r.json().get("models", [])
    except Exception:
        pass
    return []


def get_best_model() -> Optional[str]:
    """
    Auto-select the best available model from PREFERRED_MODELS.
    Returns None if no preferred model is installed.
    """
    available = {m["name"].split(":")[0] for m in list_available_models()}
    for preferred in PREFERRED_MODELS:
        if preferred in available:
            return preferred
    # Fallback: return first installed model (if any)
    models = list_available_models()
    return models[0]["name"] if models else None


def get_active_model() -> str:
    """
    Return the model to use: env var > auto-selected best > default string.
    """
    if OLLAMA_MODEL and OLLAMA_MODEL != "auto":
        return OLLAMA_MODEL
    best = get_best_model()
    return best or "llama3"


# ── Generation ───────────────────────────────────────────────────────

def generate(
    prompt: str,
    system: str = "",
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> str:
    """
    Blocking generation. Returns full response string.
    Raises OllamaUnavailableError if Ollama is not running.
    """
    _model = model or get_active_model()

    payload = {
        "model":  _model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    if system:
        payload["system"] = system

    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json=payload,
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        raise OllamaUnavailableError(
            "Ollama is not running. Start it with: `ollama serve`"
        )
    except requests.exceptions.Timeout:
        raise OllamaUnavailableError(
            f"Ollama timed out after {TIMEOUT}s. Try a smaller model."
        )
    except requests.exceptions.HTTPError as e:
        raise OllamaUnavailableError(f"Ollama HTTP error: {e}")


def generate_stream(
    prompt: str,
    system: str = "",
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> Generator[str, None, None]:
    """
    Streaming generation. Yields text chunks as they arrive.
    Suitable for Server-Sent Events (SSE).
    """
    _model = model or get_active_model()

    payload = {
        "model":  _model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    if system:
        payload["system"] = system

    try:
        with requests.post(
            f"{OLLAMA_URL}/api/generate",
            json=payload,
            stream=True,
            timeout=TIMEOUT,
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        token = chunk.get("response", "")
                        if token:
                            yield token
                        if chunk.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue
    except requests.exceptions.ConnectionError:
        yield "[ERROR: Ollama is not running. Start it with: `ollama serve`]"
    except requests.exceptions.Timeout:
        yield f"[ERROR: Ollama timed out after {TIMEOUT}s]"


def chat(
    messages: list[dict],
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> str:
    """
    Chat-style generation with conversation history.
    messages: [{"role": "user"|"assistant"|"system", "content": str}, ...]
    """
    _model = model or get_active_model()

    payload = {
        "model":    _model,
        "messages": messages,
        "stream":   False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json=payload,
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        return r.json().get("message", {}).get("content", "").strip()
    except requests.exceptions.ConnectionError:
        raise OllamaUnavailableError(
            "Ollama is not running. Start it with: `ollama serve`"
        )
    except requests.exceptions.Timeout:
        raise OllamaUnavailableError(
            f"Ollama timed out after {TIMEOUT}s."
        )
    except requests.exceptions.HTTPError as e:
        raise OllamaUnavailableError(f"Ollama HTTP error: {e}")


def chat_stream(
    messages: list[dict],
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> Generator[str, None, None]:
    """
    Streaming chat. Yields text tokens as they arrive.
    """
    _model = model or get_active_model()

    payload = {
        "model":    _model,
        "messages": messages,
        "stream":   True,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    try:
        with requests.post(
            f"{OLLAMA_URL}/api/chat",
            json=payload,
            stream=True,
            timeout=TIMEOUT,
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        token = chunk.get("message", {}).get("content", "")
                        if token:
                            yield token
                        if chunk.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue
    except requests.exceptions.ConnectionError:
        yield "[ERROR: Ollama is not running. Start it with: `ollama serve`]"
    except requests.exceptions.Timeout:
        yield f"[ERROR: Ollama timed out after {TIMEOUT}s]"


# ── Prompt Builders ──────────────────────────────────────────────────

def build_dataset_context(df_summary: dict) -> str:
    """Build a compact dataset description for inclusion in AI prompts."""
    lines = [
        f"Dataset: {df_summary.get('name', 'Unknown')}",
        f"Rows: {df_summary.get('rows', 0):,}",
        f"Columns: {', '.join(df_summary.get('column_names', []))}",
    ]
    if df_summary.get("total_sales"):
        lines.append(f"Total Sales: ${df_summary['total_sales']:,.2f}")
    if df_summary.get("total_profit") is not None:
        lines.append(f"Total Profit: ${df_summary['total_profit']:,.2f}")
    if df_summary.get("total_orders"):
        lines.append(f"Total Orders: {df_summary['total_orders']:,}")
    return "\n".join(lines)


def extract_json_from_response(text: str) -> Optional[dict]:
    """
    Try to extract a JSON object from an Ollama response.
    Handles cases where the model wraps JSON in markdown code blocks.
    """
    import re
    # Try raw JSON first
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    # Try JSON inside ```json ... ``` block
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # Try first { ... } block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


# ── Custom Exceptions ────────────────────────────────────────────────

class OllamaUnavailableError(Exception):
    """Raised when Ollama is not reachable or returns an error."""
    pass
