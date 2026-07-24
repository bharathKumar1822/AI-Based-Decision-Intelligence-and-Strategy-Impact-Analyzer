"""
run.py — Local development runner
Usage: python run.py
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
from backend.app import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Decision Intelligence Analyzer on http://localhost:{port}")
    app.run(debug=True, port=port, host="0.0.0.0")
