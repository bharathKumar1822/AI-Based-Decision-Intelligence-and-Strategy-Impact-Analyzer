# AI-Based Decision Intelligence & Strategy Impact Analyzer

> **v4.0** | AI-powered business intelligence dashboard for data-driven decision making.

---

## 🚀 Live Deployments

| Service | URL | Status |
|---------|-----|--------|
| 🖥️ **Frontend (Vercel)** | [decision-intelligence-frontend-iota.vercel.app](https://decision-intelligence-frontend-iota.vercel.app) | ✅ Live |
| ⚙️ **Backend + Full App (Render)** | [ai-based-decision-intelligence-and-hh6v.onrender.com](https://ai-based-decision-intelligence-and-hh6v.onrender.com) | ✅ Live |
| 📁 **GitHub** | [AI-Based-Decision-Intelligence-and-Strategy-Impact-Analyzer](https://github.com/bharathKumar1822/AI-Based-Decision-Intelligence-and-Strategy-Impact-Analyzer) | ✅ Updated |

---

## ✨ Dashboard Features

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

---

## 🏗️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | Vanilla HTML, CSS, JavaScript (Google Fonts: Inter, Space Grotesk) |
| **Backend** | Python 3.11 · Flask · Gunicorn |
| **ML / Data** | scikit-learn · pandas · numpy · matplotlib · seaborn · scipy |
| **Hosting** | Vercel (frontend) · Render (backend + full app) |

---

## 📁 Project Structure

```
Decision-Intelligence-Project/
├── backend/
│   └── app.py              # Flask REST API (all dashboard endpoints)
├── frontend/
│   ├── index.html          # Dashboard SPA
│   ├── style.css           # Dark-theme styles
│   └── app.js              # Frontend logic + API calls
├── data/
│   ├── superstore_dataset1.csv   # Dataset A (default)
│   ├── superstore_dataset2.csv   # Dataset B (default)
│   └── global dataset 6.csv     # Global Dataset 6 (default)
├── render.yaml             # Render deployment config
├── vercel.json             # Vercel deployment config
├── Procfile                # Gunicorn startup (Render)
├── requirements.txt        # Python dependencies
├── run.py                  # Local development runner
└── README.md
```

---

## 🛠️ Local Development

### Prerequisites
- Python 3.11+
- pip

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

# 4. Run the app
python run.py
```

Open **http://localhost:5000** in your browser.

The Flask server serves both the API (`/api/*`) and the static frontend at root (`/`).

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

### Vercel (Frontend Only)

The `vercel.json` at root configures Vercel to serve the `frontend/` directory and proxy `/api/*` requests to the Render backend.

1. Connect the GitHub repo to Vercel
2. **Framework Preset**: Other (static)
3. **Output Directory**: `frontend`
4. No environment variables needed (Render URL is baked into `app.js` and `vercel.json`)

---

## 🔑 Environment Variables

| Variable | Where | Value |
|----------|-------|-------|
| `PORT` | Render (auto-set) | Set by Render automatically |
| `PYTHON_VERSION` | Render | `3.11.9` |

No other environment variables are required. The app uses in-memory storage for datasets (no database).

---

## 📡 API Endpoints

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

---

## 📋 Dataset Format

The app is optimized for **Superstore-style CSV datasets** with these columns:

`Order ID`, `Order Date`, `Ship Date`, `Ship Mode`, `Customer ID`, `Customer Name`, `Segment`, `Country`, `City`, `State`, `Postal Code`, `Region`, `Product ID`, `Category`, `Sub-Category`, `Product Name`, `Sales`, `Quantity`, `Discount`, `Profit`

You can also upload any CSV with at least `Sales`, `Quantity`, and `Profit` columns for ML analysis.