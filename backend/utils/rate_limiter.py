"""
backend/utils/rate_limiter.py
Flask-Limiter configuration for protecting GenAI endpoints from abuse.
"""

from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# ── Limiter instance — attach to Flask app in app.py ────────────────
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["200 per minute", "2000 per hour"],
    storage_uri="memory://",     # In-memory (no Redis needed)
    strategy="fixed-window",
)

# ── Shared limit strings (import these in route decorators) ─────────

# Standard API routes: generous limits
STANDARD_LIMIT = "100 per minute"

# GenAI routes: more conservative (Ollama is CPU-heavy)
GENAI_LIMIT = "10 per minute"

# RAG upload: very conservative (file processing + embedding)
RAG_UPLOAD_LIMIT = "5 per minute"

# Agent analysis: expensive parallel workload
AGENT_LIMIT = "5 per minute"
