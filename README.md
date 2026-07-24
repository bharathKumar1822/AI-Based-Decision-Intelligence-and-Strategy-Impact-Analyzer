# AI-Based Decision Intelligence & Strategy Impact Analyzer

> **v5.0 — GenAI Edition** | Production-ready AI-powered business intelligence platform with GenAI capabilities.

---

## 🚀 Live Deployments

| Service | URL | Status |
|---------|-----|--------|
| 🖥️ **Frontend (Vercel)** | [decision-intelligence-project.vercel.app](https://decision-intelligence-project.vercel.app/) | ✅ Live |
| ⚙️ **Backend + Full App (Render)** | [ai-based-decision-intelligence-and-hh6v.onrender.com](https://ai-based-decision-intelligence-and-hh6v.onrender.com) | ✅ Live |
| 📁 **GitHub** | [AI-Based-Decision-Intelligence-and-Strategy-Impact-Analyzer](https://github.com/bharathKumar1822/AI-Based-Decision-Intelligence-and-Strategy-Impact-Analyzer) | ✅ Updated |

---

## ✨ Dashboard Features

### Classic Analytics
- 📊 **Overview** — KPI cards with region/category/segment filters
- 🧠 **AI Decision Engine** — Ranked strategies with ROI, risk & time-to-impact
- 🔍 **EDA Charts** — Profit by category, sales by region, monthly trends
- ⚠️ **Weakness Detection** — Root-cause analysis for critical business issues
- 🔮 **ML Predictions** — Multi-model comparison (Linear Regression, Decision Tree, Random Forest)
- 🔬 **Model Explainability** — Feature importance & SHAP-style profit driver analysis
- ⚙️ **Strategy Simulation** — What-if analysis with dynamic growth scenarios (1–100%)
- 🚨 **Risk & Anomaly Detection** — Z-score outliers, profit drop detection, discount risk
- 💡 **Recommendations** — AI-generated step-by-step action plans with do/don't guidance
- ⚖️ **Company Comparison** — Multi-dataset side-by-side analysis with charts
- 🏁 **Conclusion & Report** — AI-generated project summary with PDF print support

### GenAI Features (NEW in v5.0)
- 🤖 **AI Business Copilot** — Natural language chat with streaming responses, conversation memory, dataset-aware context
- 📚 **Knowledge Base (RAG)** — Upload PDF/DOCX/PPTX/TXT/CSV documents, query with AI-cited answers using ChromaDB
- 🕵️ **Multi-Agent Strategy** — 5 specialized AI agents (Sales, Finance, Marketing, Operations, CEO) analyze data in parallel
- 🔮 **NL Forecasting** — Ask what-if questions in plain English ("What if sales increase by 15%?")
- 🎯 **AI Strategy Generator** — Structured business strategies with ROI estimates, risk register, and 90-day action plans
- 🌙 **Dark/Light Mode** — Theme toggle with persistent preference

---

## 🏗️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | Vanilla HTML, CSS, JavaScript (Google Fonts: Inter, Space Grotesk) |
| **Backend** | Python 3.11+ · Flask · Gunicorn |
| **ML / Data** | scikit-learn · pandas · numpy · matplotlib · seaborn · scipy |
| **GenAI** | Ollama (Llama 3, Mistral, Gemma, Qwen, DeepSeek) |
| **RAG** | ChromaDB · Sentence-Transformers (all-MiniLM-L6-v2) |
| **Document Parsing** | PyMuPDF (PDF) · python-docx (DOCX) · python-pptx (PPTX) |
| **Hosting** | Vercel (frontend) · Render (backend + full app) · Docker |

---

## 📁 Project Structure

```
Decision-Intelligence-Project/
├── backend/
│   ├── app.py                      # Flask REST API (all dashboard endpoints)
│   ├── __init__.py
│   ├── genai/                      # GenAI feature modules
│   │   ├── __init__.py
│   │   ├── ollama_client.py        # Ollama HTTP client (sync/streaming)
│   │   ├── copilot.py              # AI Business Copilot (chat + SSE)
│   │   ├── rag.py                  # RAG pipeline (ChromaDB + document ingestion)
│   │   ├── agents.py               # Multi-agent analysis system
│   │   ├── strategy_gen.py         # AI strategy generator
│   │   └── predictive_nl.py        # NL forecasting engine
│   └── utils/
│       ├── __init__.py
│       ├── cache.py                # In-memory TTL cache
│       └── rate_limiter.py         # Flask-Limiter configuration
├── frontend/
│   ├── index.html                  # Dashboard SPA
│   ├── style.css                   # Dark/light theme styles
│   ├── app.js                      # Core frontend logic + API calls
│   └── genai.js                    # GenAI UI (chat, RAG, agents, forecast)
├── data/
│   ├── superstore_dataset1.csv
│   ├── superstore_dataset2.csv
│   └── global dataset 6.csv
├── tests/
│   ├── __init__.py
│   └── test_backend.py             # 13 unit tests (pytest)
├── docs/
│   └── api_docs.md                 # GenAI API reference
├── Dockerfile                      # Container build
├── docker-compose.yml              # App + Ollama orchestration
├── .env.example                    # Environment config template
├── render.yaml                     # Render deployment
├── vercel.json                     # Vercel deployment
├── Procfile                        # Gunicorn startup
├── requirements.txt                # Python dependencies
├── run.py                          # Local dev runner
└── README.md
```

---

## 🛠️ Local Development

### Prerequisites
- Python 3.11+
- pip
- **Ollama** (optional — for GenAI features)

### Setup

```bash
# 1. Clone the repo
git clone https://github.com/bharathKumar1822/AI-Based-Decision-Intelligence-and-Strategy-Impact-Analyzer.git
cd AI-Based-Decision-Intelligence-and-Strategy-Impact-Analyzer

# 2. Create and activate virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Set up Ollama for GenAI features
# Download from https://ollama.com
ollama pull llama3
ollama serve   # Runs on port 11434

# 5. Run the app
python run.py
```

Open **http://localhost:5000** in your browser.

The Flask server serves both the API (`/api/*`) and the static frontend at root (`/`).

### Docker Setup

```bash
# Start everything (app + Ollama)
docker-compose up -d

# Pull a model inside the Ollama container
docker exec -it decision-intelligence-ollama ollama pull llama3
```

---

## 🌐 Deployment

### Render (Backend + Full App)

The app is configured via `render.yaml`. On Render:

1. Connect the GitHub repo to Render
2. Render auto-detects `render.yaml` and creates a **Python Web Service**
3. **Build Command**: `pip install -r requirements.txt`
4. **Start Command**: `gunicorn backend.app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120`
5. **Environment Variables**:
   - `PYTHON_VERSION`: `3.11.9`

> **Note**: The free tier on Render spins down after 15 minutes of inactivity. First request may take ~30 seconds to wake up.
> GenAI features require Ollama running — they gracefully degrade when Ollama is unavailable.

### Vercel (Frontend Only)

The `vercel.json` at root configures Vercel to serve the `frontend/` directory and proxy `/api/*` requests to the Render backend.

1. Connect the GitHub repo to Vercel
2. **Framework Preset**: Other (static)
3. **Output Directory**: `frontend`
4. No environment variables needed (Render URL is baked into `app.js` and `vercel.json`)

---

## 🔑 Environment Variables

| Variable | Where | Default | Description |
|----------|-------|---------|-------------|
| `PORT` | Render (auto-set) | `5000` | Server port |
| `PYTHON_VERSION` | Render | `3.11.9` | Python version |
| `OLLAMA_URL` | `.env` | `http://localhost:11434` | Ollama API endpoint |
| `OLLAMA_MODEL` | `.env` | `llama3` | Default LLM model |
| `OLLAMA_TIMEOUT` | `.env` | `120` | LLM request timeout (seconds) |
| `CHROMA_PERSIST_DIR` | `.env` | `./chroma_db` | ChromaDB storage directory |

No database required. The app uses in-memory storage for datasets.

---

## 📡 API Endpoints

### Classic Analytics API

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/datasets` | List all loaded datasets |
| POST | `/api/upload` | Upload a CSV file |
| POST | `/api/load-defaults` | Load built-in demo datasets |
| DELETE | `/api/remove/<name>` | Remove a loaded dataset |
| GET | `/api/summary/<name>` | KPI summary + dataset info |
| GET | `/api/eda/<name>` | EDA charts (base64 PNG) |
| GET | `/api/weakness/<name>` | Weakness detection |
| GET | `/api/predict/<name>` | ML predictions & model comparison |
| GET | `/api/strategy/<name>` | Strategy simulation |
| GET | `/api/recommend/<name>` | AI recommendations |
| GET | `/api/anomaly/<name>` | Risk & anomaly detection |
| GET | `/api/explain/<name>` | Model explainability |
| GET | `/api/decision-engine/<name>` | AI decision engine |
| GET | `/api/compare` | Multi-dataset comparison |
| GET | `/api/conclusion` | Final AI-generated report |
| GET | `/api/refresh-status` | Server health check |

### GenAI API (22 routes)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/genai/status` | Ollama health check |
| GET | `/api/genai/models` | List installed LLM models |
| POST | `/api/genai/chat` | Streaming SSE chat |
| POST | `/api/genai/chat/sync` | Non-streaming chat |
| POST | `/api/genai/clear-session` | Clear chat history |
| GET | `/api/genai/history/<id>` | Get conversation history |
| POST | `/api/genai/explain/dataset` | AI dataset explanation |
| POST | `/api/genai/explain/ml` | AI ML results explanation |
| POST | `/api/rag/upload` | Upload document to knowledge base |
| GET | `/api/rag/documents` | List uploaded documents |
| DELETE | `/api/rag/documents/<id>` | Remove document |
| POST | `/api/rag/query` | Query documents with AI |
| GET | `/api/rag/status` | RAG system status |
| POST | `/api/agents/analyze` | Multi-agent analysis |
| POST | `/api/agents/analyze/stream` | Streaming multi-agent analysis |
| POST | `/api/genai/strategy/generate` | Generate AI strategy |
| POST | `/api/genai/strategy/refine` | Refine existing strategy |
| GET | `/api/genai/forecast/examples` | Forecast example queries |
| POST | `/api/genai/forecast/query` | NL forecast simulation |
| POST | `/api/genai/forecast/parse` | Parse NL intent (debug) |

> Full API documentation: [`docs/api_docs.md`](docs/api_docs.md)

---

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage (install pytest-cov first)
python -m pytest tests/ -v --cov=backend
```

13 tests covering:
- Health routes (ping, datasets, refresh)
- Static file serving (index.html)
- GenAI endpoints (status, models, chat validation, RAG)
- NL forecasting (parser, simulation, direction handling)

---

## 📋 Dataset Format

The app is optimized for **Superstore-style CSV datasets** with these columns:

`Order ID`, `Order Date`, `Ship Date`, `Ship Mode`, `Customer ID`, `Customer Name`, `Segment`, `Country`, `City`, `State`, `Postal Code`, `Region`, `Product ID`, `Category`, `Sub-Category`, `Product Name`, `Sales`, `Quantity`, `Discount`, `Profit`

You can also upload any CSV with at least `Sales`, `Quantity`, and `Profit` columns for ML analysis.

---

## 🤖 GenAI Setup (Optional)

GenAI features require **Ollama** running locally. Without Ollama, the classic analytics features work perfectly — GenAI tabs will show a setup guide.

```bash
# Install Ollama (https://ollama.com)
# Then pull a model:
ollama pull llama3          # Default, 4.7GB
ollama pull mistral         # Alternative, 4.1GB
ollama pull gemma:7b        # Google Gemma
ollama pull qwen2:7b        # Alibaba Qwen
ollama pull deepseek-coder  # Coding-focused

# Start the server:
ollama serve
```

The platform auto-detects available models and selects the best one. You can switch models from the AI Copilot sidebar.