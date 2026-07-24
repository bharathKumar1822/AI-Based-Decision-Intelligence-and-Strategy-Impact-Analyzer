"""
backend/genai/__init__.py
GenAI module package for Decision Intelligence Analyzer.
Exports all Flask blueprints for registration in app.py.
"""
from .copilot import copilot_bp
from .rag import rag_bp
from .agents import agents_bp
from .strategy_gen import strategy_bp
from .predictive_nl import forecast_bp

__all__ = ["copilot_bp", "rag_bp", "agents_bp", "strategy_bp", "forecast_bp"]
