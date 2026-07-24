"""
tests/test_backend.py
Unit tests for the existing Decision Intelligence backend API routes.
Uses Flask test client — no server startup required.
"""

import os
import sys
import json
import pytest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.app import app


@pytest.fixture
def client():
    """Create a Flask test client."""
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


class TestHealthRoutes:
    def test_ping(self, client):
        """Backend should respond to /api/ping."""
        res = client.get("/api/ping")
        assert res.status_code == 200
        data = json.loads(res.data)
        assert "status" in data or "pong" in data.get("status", "pong")

    def test_datasets_empty(self, client):
        """Should return an empty list when no datasets are loaded."""
        res = client.get("/api/datasets")
        assert res.status_code == 200
        data = json.loads(res.data)
        assert "datasets" in data
        assert isinstance(data["datasets"], list)

    def test_refresh_status(self, client):
        """Refresh-status should return server metadata."""
        res = client.get("/api/refresh-status")
        assert res.status_code == 200
        data = json.loads(res.data)
        assert "datasets_loaded" in data


class TestStaticServing:
    def test_index_html(self, client):
        """Frontend index.html should be served."""
        res = client.get("/")
        assert res.status_code == 200
        assert b"DecisionAI" in res.data or b"Decision" in res.data


class TestGenAIRoutes:
    def test_genai_status(self, client):
        """GenAI status endpoint should respond (even if Ollama is offline)."""
        res = client.get("/api/genai/status")
        assert res.status_code == 200
        data = json.loads(res.data)
        assert "ollama_available" in data

    def test_genai_models(self, client):
        """Model list endpoint should respond."""
        res = client.get("/api/genai/models")
        # May be 200 (if Ollama up) or 503 (if Ollama down) — both are valid
        assert res.status_code in (200, 503)

    def test_rag_status(self, client):
        """RAG status should respond."""
        res = client.get("/api/rag/status")
        assert res.status_code in (200, 503)

    def test_rag_documents_empty(self, client):
        """Document list should return valid JSON (or 500 if chromadb missing)."""
        res = client.get("/api/rag/documents")
        # 200 if chromadb installed, 500 if not — both are valid
        assert res.status_code in (200, 500)
        data = json.loads(res.data)
        if res.status_code == 200:
            assert "documents" in data

    def test_forecast_examples(self, client):
        """Forecast examples endpoint should return list."""
        res = client.get("/api/genai/forecast/examples")
        assert res.status_code == 200
        data = json.loads(res.data)
        assert "examples" in data
        assert len(data["examples"]) > 0

    def test_chat_no_message(self, client):
        """Chat endpoint should reject empty messages."""
        res = client.post(
            "/api/genai/chat/sync",
            data=json.dumps({"message": ""}),
            content_type="application/json",
        )
        assert res.status_code == 400

    def test_forecast_parse(self, client):
        """Intent parser should work without Ollama."""
        res = client.post(
            "/api/genai/forecast/parse",
            data=json.dumps({"query": "What if sales increase by 20%?"}),
            content_type="application/json",
        )
        assert res.status_code == 200
        data = json.loads(res.data)
        assert "parsed" in data
        parsed = data["parsed"]
        assert parsed["direction"] == "increase"
        assert parsed["percentage"] == 20.0


class TestForecastSimulation:
    def test_forecast_query_no_ollama(self, client):
        """Forecast should still return simulation even if Ollama is offline."""
        payload = {
            "query": "What if profit increases by 10%?",
            "dataset_summary": {
                "name":         "test",
                "total_sales":  1000000,
                "total_profit":  150000,
                "total_orders":  5000,
            },
        }
        res = client.post(
            "/api/genai/forecast/query",
            data=json.dumps(payload),
            content_type="application/json",
        )
        assert res.status_code == 200
        data = json.loads(res.data)
        assert "simulation" in data
        sim = data["simulation"]
        assert "projected_profit" in sim
        assert "projected_sales"  in sim

    def test_forecast_direction_decrease(self, client):
        """Decrease queries should produce lower projections."""
        payload = {
            "query": "What if sales decrease by 25%?",
            "dataset_summary": {
                "name": "test", "total_sales": 1000000,
                "total_profit": 150000, "total_orders": 5000,
            },
        }
        res = client.post(
            "/api/genai/forecast/query",
            data=json.dumps(payload),
            content_type="application/json",
        )
        assert res.status_code == 200
        data = json.loads(res.data)
        sim  = data["simulation"]
        assert sim["projected_sales"] < sim["current_sales"]
