# AI Decision Copilot (v5.0 GenAI Edition)
## Master Technical Interview & System Architecture Guide

---

# 1. Executive Summary

### Project Overview
**AI Decision Copilot** (officially titled *AI-Based Decision Intelligence & Strategy Impact Analyzer*) is a production-ready, full-stack GenAI analytics and decision intelligence platform. It bridges classic statistical data analysis, machine learning forecasting, multi-agent LLM strategy synthesis, and Retrieval-Augmented Generation (RAG) over corporate knowledge bases into a unified single-page web dashboard.

### Objective
To empower business analysts, executives, and decision-makers to transform raw tabular business data (sales, profit, discount, regions, categories) and unstructured corporate documents (PDF, DOCX, PPTX, TXT, CSV) into actionable strategic decisions, predictive financial forecasts, risk notifications, and automated executive reports without requiring data science expertise.

### Business Value
- **Reduces Analysis Time from Days to Seconds:** Automated Exploratory Data Analysis (EDA), weakness detection, and multi-model machine learning eliminate manual spreadsheet slicing.
- **Data-Driven Strategy Selection:** Ranks business interventions by expected ROI, implementation time, and risk level.
- **On-Premise / Private GenAI Privacy:** Integrates local LLM execution via Ollama (Llama 3, Mistral, Gemma), ensuring enterprise data privacy without sending sensitive financial records to external cloud API vendors.
- **Cross-Enterprise Knowledge Querying:** RAG enables semantic natural language Q&A over internal policy documents and financial statements with source citation.

### Real-World Use Case
A retail conglomerate operating across multiple geographic regions and product categories notices declining profit margins despite high total revenue. Using **AI Decision Copilot**:
1. The CFO uploads regional sales CSVs into the dashboard.
2. The ML engine trains and identifies that heavy discounting (greater than 20%) in specific sub-categories (e.g., Tables, Supplies) is eroding profit.
3. The Multi-Agent system (Sales, Finance, CMO, COO, CEO agents) analyzes the data in parallel to generate a 90-day turnaround strategy.
4. The CFO asks the NL Forecasting engine: *"What if we cap discounts at 15% across all regions?"*, instantly receiving a numerical simulation showing a +$82,000 net profit increase.

---

# 2. Problem Statement

### Existing Business Problem
Enterprise business decision-making is severely bottlenecked by data fragmentation, manual spreadsheet modeling, slow reporting cycles, and non-explainable black-box analytical tools.

### Key Challenges & Limitations of Legacy Systems
1. **Siloed & Static Dashboards:** Traditional BI tools (Tableau, PowerBI) show *what happened* (descriptive analytics), but fail to answer *why it happened* or *what will happen if we take action X* (prescriptive analytics).
2. **Spreadsheet Error Fragility:** Manual scenario modeling in Excel is prone to formula errors, lacks scalable machine learning predictions, and cannot process unstructured PDF reports.
3. **Black-Box AI Lack of Trust:** Standard ML models output prediction numbers without plain-English explanations or feature impact breakdowns, creating resistance among non-technical executives.
4. **Data Privacy Constraints in Cloud LLMs:** Sending proprietary corporate sales figures or merger strategies to third-party public API endpoints (e.g., OpenAI API) violates strict enterprise compliance (GDPR, HIPAA, SOC2).

### Why This Solution Is Needed
Modern enterprise management requires an integrated **Decision Intelligence Platform** that combines classical data validation/ML with private local GenAI, offering conversation-driven exploration, vector search over internal reports, and interactive what-if simulations.

---

# 3. Proposed Solution

### How the Platform Solves the Problem
The **AI Decision Copilot** combines statistical computing (`pandas`, `scipy`), machine learning (`scikit-learn`), local LLM orchestration (`Ollama`), vector search (`ChromaDB`), and real-time streaming web frontends (`HTML5/CSS3/ES6 JS`) into a cohesive decision engine.

```
+-----------------------------------------------------------------------+
|                         PROPOSED SOLUTION                             |
+-----------------------------------------------------------------------+
| 1. Automated Data Ingestion & Weakness Auto-Detection                 |
| 2. Multi-Model ML Profit Prediction & Plain-English Explainability    |
| 3. What-If Interactive Simulation Engine                              |
| 4. Private Local GenAI Copilot (Ollama + Llama 3)                     |
| 5. Document RAG Pipeline with ChromaDB & Source Citations             |
| 6. Parallel 5-Agent Executive Panel Analysis (Sales/Fin/Mkt/Ops/CEO)   |
+-----------------------------------------------------------------------+
```

### Business Impact & Benefits
- **Automated Root-Cause Discovery:** Instantly highlights unprofitable product sub-categories, loss-making regions, and high-discount risk profiles.
- **Accurate Financial Modeling:** Auto-selects the top-performing ML model (evaluating $R^2$, RMSE, MAE) and provides feature importance breakdowns.
- **Zero Third-Party Token Costs:** Runs completely locally or self-hosted via Docker containerization with local LLM models.

---

# 4. Complete Workflow

```
[ User Action ]
       │
       ├──────────────────────────────────────────────────────┐
       │ (Tabular CSV Upload)                                  │ (Unstructured Document)
       ▼                                                      ▼
[ /api/upload ]                                      [ /api/rag/upload ]
       │                                                      │
       ▼                                                      ▼
[ Validation & Cleaning ]                            [ PyMuPDF / docx / pptx ]
- Column Normalization                               - Text Extraction & Normalization
- Missing Value Imputation                           - Recursive Character Chunking
- Duplicate Removal                                           │
       │                                                      ▼
       ▼                                            [ Sentence Transformers ]
[ Feature Engineering & EDA ]                        - all-MiniLM-L6-v2 (384-dim)
- Profit Margin, Discount Binning                             │
- Z-Score Anomaly Detection                                   ▼
       │                                            [ ChromaDB Vector Store ]
       ├──────────────────────────┐                  - Persistent HNSW Indexing
       ▼                          ▼                           │
[ ML Model Training ]     [ Statistical Rules ]               │
- Linear Regression       - Loss Sub-category                 │
- Decision Tree           - Discount Risk                     │
- Random Forest Regressor         │                           │
       │                          │                           │
       ▼                          ▼                           ▼
[ Model Evaluation ]      [ Simulation Engine ]     [ RAG Query Engine ]
- Picks Best R^2          - What-If Calculations     - Similarity Search (Top-K)
       │                          │                  - Context Assembly
       └──────────────────────────┼───────────────────────────┘
                                  │
                                  ▼
                     [ Ollama LLM Orchestrator ]
                     - Llama 3 / Mistral Inference
                     - System Prompt Injection
                     - SSE Streaming Generator
                                  │
                                  ├───────────────────────────┐
                                  ▼                           ▼
                     [ Multi-Agent Panel ]           [ Copilot Chat ]
                     - Sales, Finance, Marketing,    - Real-Time Token Stream
                       Operations, CEO Synthesis              │
                                  │                           │
                                  └─────────────┬─────────────┘
                                                │
                                                ▼
                                   [ Glassmorphism SPA ]
                                   - KPI Cards & EDA Charts
                                   - Interactive Simulation
                                   - PDF Executive Export
```

---

# 5. System Architecture

```
+-----------------------------------------------------------------------+
|                          SYSTEM ARCHITECTURE                          |
+-----------------------------------------------------------------------+
|                                                                       |
|  [ Vercel CDN / Static Host ]  <--->  [ Vanilla HTML5 / CSS3 / JS SPA ]|
|                                                     │ (REST & SSE)    |
|                                                     ▼                 |
|  [ Render / Docker Container ] ─────────────────────────────────────  |
|  │  Flask Application Server (WSGI / Gunicorn)                       │  |
|  │  ├── Rate Limiter & In-Memory TTL Cache                           │  |
|  │                                                                   │  |
|  │  ├── [/api/upload, /api/predict, /api/eda, /api/recommend]       │  |
|  │  │   ├── Pandas / NumPy Data Processing                           │  |
|  │  │   ├── Scikit-Learn Regression Pipeline                         │  |
|  │  │   └── Matplotlib Non-Interactive Rendering Engine              │  |
|  │  │                                                                │  |
|  │  ├── [/api/genai/*] GenAI & Copilot Blueprints                    │  |
|  │  │   ├── Ollama Client (Sync & SSE Async Streamer)                │  |
|  │  │   ├── Predictive NL Scenario Simulation                        │  |
|  │  │   └── Strategy Generator                                       │  |
|  │  │                                                                │  |
|  │  ├── [/api/rag/*] Knowledge Base Blueprint                        │  |
|  │  │   ├── Document Loaders (PyMuPDF, docx, pptx)                   │  |
|  │  │   ├── SentenceTransformers Embedding Pipeline                  │  |
|  │  │   └── ChromaDB Persistent Vector Database Engine               │  |
|  │  │                                                                │  |
|  │  └── [/api/agents/*] Multi-Agent Blueprint                        │  |
|  │      └── Multi-Threaded Parallel Specialist Execution               │  |
|  └───────────────────────────────────────────────────────────────────┘  |
|                                                     │                 |
|                                                     ▼                 |
|  [ Ollama Server (Port 11434) ] <───────────────────┘                 |
|  └── Local Model Storage (Llama 3 / Mistral / Gemma)                  |
+-----------------------------------------------------------------------+
```

---

# 6. Folder Structure

```
Decision-Intelligence-Project/
├── backend/
│   ├── app.py                      # Core Flask API server & classic analytics routes
│   ├── __init__.py                 # Package initializer
│   ├── genai/                      # GenAI feature sub-package
│   │   ├── __init__.py
│   │   ├── ollama_client.py        # Ollama HTTP REST client (sync & SSE streaming)
│   │   ├── copilot.py              # AI Business Copilot logic & session manager
│   │   ├── rag.py                  # RAG ingestion, chunking, ChromaDB vector search
│   │   ├── agents.py               # Parallel 5-agent strategic analysis engine
│   │   ├── strategy_gen.py         # Structured AI strategy & 90-day action planner
│   │   └── predictive_nl.py        # NL scenario parsing & simulation engine
│   └── utils/
│       ├── __init__.py
│       ├── cache.py                # Thread-safe in-memory TTL cache
│       └── rate_limiter.py         # Flask-Limiter configuration module
├── frontend/
│   ├── index.html                  # Single Page Application structure & layout
│   ├── style.css                   # Glassmorphism dark/light theme stylesheet
│   ├── app.js                      # Core SPA tab navigation & classic API calls
│   └── genai.js                    # GenAI UI handlers (chat, RAG, agents, forecast)
├── data/                           # Default CSV datasets (Superstore 1, 2, Global 6)
├── chroma_db/                      # ChromaDB vector database persistence directory
├── docs/
│   ├── api_docs.md                 # Complete REST API reference documentation
│   └── Decision_Intelligence_Interview_Guide.md # Technical Master Guide
├── tests/
│   ├── __init__.py
│   └── test_backend.py             # 13 backend unit & integration tests (pytest)
├── Dockerfile                      # Production Docker container configuration
├── docker-compose.yml              # Multi-container orchestration (App + Ollama)
├── .env.example                    # Environment variables template
├── render.yaml                     # Render cloud deployment specification
├── vercel.json                     # Vercel static hosting configuration
├── Procfile                        # Production Gunicorn process launcher
├── requirements.txt                # Python project dependencies
├── run.py                          # Local development launcher
└── README.md                       # Comprehensive project documentation
```

---

# 7. Technology Stack Analysis

### 1. Python (v3.11+)
- **Why Chosen:** De-facto industry standard for data science, machine learning, data engineering, and GenAI pipelines.
- **Where Used:** Complete backend server, ML training, RAG embeddings, and LLM orchestration.
- **Alternatives Considered:** Node.js, Go. Node.js lacks native mature machine learning ecosystems like `scikit-learn` and `pandas`.

### 2. Flask (Python Backend Framework)
- **Why Chosen:** Lightweight, highly flexible, WSGI-compliant micro-framework. Easily supports custom Server-Sent Events (SSE) streaming responses via `stream_with_context`.
- **Where Used:** Main web application server (`backend/app.py`) and Blueprint routing.
- **Alternatives Considered:** FastAPI, Django. Flask was selected to maintain zero-overhead compatibility with Gunicorn deployments on Render while keeping simple Blueprint modularity.

### 3. Vanilla JavaScript (ES6+), HTML5, Vanilla CSS3
- **Why Chosen:** Eliminates heavy JavaScript framework build steps, node_modules bloat, and bundling overhead. Offers sub-millisecond initial page rendering and instant DOM manipulation.
- **Where Used:** Frontend SPA (`index.html`, `style.css`, `app.js`, `genai.js`).
- **Alternatives Considered:** React, Vue, Next.js. Vanilla JS ensures high performance, zero build dependencies, and complete control over glassmorphic CSS styling.

### 4. Pandas & NumPy
- **Why Chosen:** High-performance tabular data manipulation, vectorized mathematical matrix computations, missing value handling, and grouping aggregation.
- **Where Used:** All data ingestion, EDA generation, weakness detection, and simulation calculations.

### 5. Scikit-Learn
- **Why Chosen:** Standard, robust Python machine learning library providing efficient, standardized APIs for regression algorithms, preprocessing, and metrics evaluation.
- **Where Used:** ML model training, cross-validation, and feature importance calculations (`LinearRegression`, `DecisionTreeRegressor`, `RandomForestRegressor`).

### 6. Matplotlib & Seaborn
- **Why Chosen:** Produces publication-quality statistical charts.
- **Where Used:** Backend headless chart generation (`matplotlib.use("Agg")`) converted to Base64 PNG images sent to the frontend.

### 7. Ollama & Llama 3
- **Why Chosen:** Enables offline, local execution of open-source LLMs without external API costs or data privacy leaks.
- **Where Used:** AI Business Copilot chat, RAG query synthesis, Multi-Agent strategy analysis, and NL forecasting narratives.

### 8. ChromaDB & Sentence-Transformers (`all-MiniLM-L6-v2`)
- **Why Chosen:** Native, lightweight vector database running in-process with persistent storage. Sentence-transformers generates 384-dimensional semantic embeddings rapidly on standard CPUs.
- **Where Used:** Knowledge Base (RAG) vector indexing and semantic similarity retrieval.

### 9. Render & Vercel
- **Why Chosen:** Vercel provides global CDN hosting for static frontends. Render provides Docker containerized backend execution with automatic Git deployment triggers.

---

# 8. Machine Learning Pipeline

### Algorithms Implemented
The platform executes three distinct regression algorithms on tabular data to predict continuous variables (specifically `Profit` based on `Sales`, `Quantity`, and `Discount`):

1. **Linear Regression:** Standard ordinary least squares regression model establishing linear relationships.
2. **Decision Tree Regressor:** Non-linear decision tree splitting data based on feature thresholds (`max_depth=5` to prevent overfitting).
3. **Random Forest Regressor:** Ensemble of 50 decision trees (`n_estimators=50, random_state=42`) using bagging to reduce variance.

```
                  ┌─────────────────────────────────────┐
                  │    Tabular Dataset (CSV Input)      │
                  └──────────────────┬──────────────────┘
                                     │
                                     ▼
                  ┌─────────────────────────────────────┐
                  │  Feature Selection & Preprocessing  │
                  │  X = [Sales, Quantity, Discount]    │
                  │  y = Profit                         │
                  └──────────────────┬──────────────────┘
                                     │
                        ┌────────────┴────────────┐
                        ▼                         ▼
             ┌─────────────────────┐   ┌─────────────────────┐
             │ Train Set (80%)     │   │ Test Set (20%)      │
             └──────────┬──────────┘   └──────────┬──────────┘
                        │                         │
                        ▼                         │
   ┌───────────────────────────────────────────┐  │
   │           Model Training Loop             │  │
   │  - Linear Regression                      │  │
   │  - Decision Tree Regressor (max_depth=5)  │  │
   │  - Random Forest Regressor (trees=50)     │  │
   └────────────────────┬──────────────────────┘  │
                        │                         │
                        └────────────┬────────────┘
                                     │
                                     ▼
                  ┌─────────────────────────────────────┐
                  │          Evaluation Metric          │
                  │  Calculate R^2, RMSE, MAE           │
                  └──────────────────┬──────────────────┘
                                     │
                                     ▼
                  ┌─────────────────────────────────────┐
                  │   Auto-Select Best Model (Max R^2)  │
                  └─────────────────────────────────────┘
```

### Feature Importance Calculation
For tree models, feature importances are extracted directly (`model.feature_importances_`). For Linear Regression, normalized absolute coefficients are calculated to provide SHAP-style visual breakdowns of top profit drivers.

---

# 9. GenAI & RAG Architecture

### Local LLM Orchestration (Ollama)
The application communicates with Ollama via HTTP REST APIs (`http://localhost:11434/api/generate` and `/api/chat`).

### RAG (Retrieval-Augmented Generation) Architecture
```
[ Document Input: PDF / DOCX / PPTX / TXT / CSV ]
                       │
                       ▼
[ PyMuPDF / python-docx / python-pptx Extraction ]
                       │
                       ▼
[ Recursive Character Text Splitter ]
  - Chunk Size: 500 characters
  - Chunk Overlap: 50 characters
                       │
                       ▼
[ SentenceTransformers Embedding ]
  - Model: all-MiniLM-L6-v2 (384-dimensional dense vectors)
                       │
                       ▼
[ ChromaDB Storage & Indexing ]
  - Collection: "document_knowledge_base"
  - Distance Metric: Cosine Similarity / L2
                       │
                       ▼  (User Question Received)
[ Semantic Similarity Search ]
  - Vectorize Question -> Query ChromaDB for Top-K (K=5) Chunks
                       │
                       ▼
[ Context-Augmented Prompt Construction ]
  - System Prompt + Document Context Chunks + User Question
                       │
                       ▼
[ Ollama Llama 3 LLM Generation ] -> [ Output with Source Citations ]
```

### Multi-Agent Specialist Strategy Architecture
Five distinct AI agents execute in parallel using Python threading:
1. **Sales Agent:** Analyzes product revenue performance and customer segments.
2. **Finance Agent:** Evaluates profit margins, overhead costs, and discount erosion.
3. **Marketing Agent:** Focuses on customer acquisition, retention, and campaign targets.
4. **Operations Agent:** Identifies shipping efficiency, regional fulfillment, and logistics.
5. **CEO Synthesis Agent:** Consolidates reports from all four specialists into an executive strategic master plan.

---

# 10. Data Analytics & Quality Assessment

### Data Cleaning & Validation Rules
- **Missing Value Handling:** Numeric columns are imputed using median values; categorical columns are imputed using mode or `"Unknown"`.
- **Normalization:** Column names are stripped of whitespace, converted to standard casing, and mapped to expected canonical names (`Sales`, `Profit`, `Quantity`, `Discount`, `Category`).
- **Outlier Detection:** Uses $Z$-score statistical thresholding ($|Z| > 3.0$) to flag extreme outliers in profit or sales.
- **Discount Risk Assessment:** Flags orders where `Discount >= 0.20` and `Profit < 0` as high-risk discount erosion.

---

# 11. REST API Reference Overview

| Endpoint | Method | Purpose | Key Input | Key Output |
|---|---|---|---|---|
| `/api/upload` | `POST` | Upload CSV dataset | Multipart File | Dataset Summary & Status |
| `/api/summary/<name>` | `GET` | Retrieve dataset KPIs | Dataset Name | Total Sales, Profit, Records, Columns |
| `/api/predict/<name>` | `GET` | Train & evaluate ML models | Dataset Name | $R^2$, RMSE, Best Model, Feature Importance |
| `/api/strategy/<name>` | `GET` | Simulate what-if scenarios | `growth_pct` Query Param | Simulated Sales, Profit, Growth Delta |
| `/api/genai/chat` | `POST` | Streaming AI Copilot chat | Prompt & Context JSON | Server-Sent Events (SSE Token Stream) |
| `/api/rag/upload` | `POST` | Ingest document to RAG | PDF/DOCX/PPTX File | File ID, Total Chunks Index |
| `/api/rag/query` | `POST` | Semantic search Q&A | Question JSON | Answer Text & Citation Sources |
| `/api/agents/analyze/stream` | `POST` | Run 5-Agent Analysis | Dataset Summary JSON | SSE Stream of Agent Reports |
| `/api/genai/strategy/generate` | `POST` | Generate AI Strategy | Focus Area | Structured Action Plan & Risk Register |
| `/api/genai/forecast/query` | `POST` | NL Scenario Forecasting | Natural Language Query | Numeric Simulation + AI Narrative |

---

# 12. Feature Explanation Details

1. **AI Business Copilot:** Interactive, dataset-aware streaming chat assistant providing real-time data insights.
2. **Knowledge Base (RAG):** Upload corporate documents and perform semantic vector searches with interactive citation source chips.
3. **Multi-Agent Executive Panel:** Parallel strategic analysis from 5 domain-specific AI agents.
4. **Natural Language Forecasting:** Enables asking questions like *"What if sales increase by 15%?"* to compute numerical projections combined with AI explanations.
5. **AI Strategy Generator:** Generates structured initiatives, 90-day action plans, and risk registers categorized by impact and ROI.
6. **Dark/Light Glassmorphism Theme Engine:** Dynamically switches UI themes with persistent local storage saving.

---

# 13. End-to-End User Journey Example

```
Step 1: User uploads `superstore_dataset1.csv` via drag-and-drop.
Step 2: Backend validates 9,994 rows, cleans missing fields, and computes KPIs ($2.29M Sales, $286K Profit).
Step 3: User clicks ML Predictions tab -> Backend trains Linear Regression, Decision Tree, and Random Forest.
Step 4: Random Forest achieves R^2 = 0.86 (selected as best model). Explains Discount as top negative driver.
Step 5: User opens RAG tab and uploads `Q4_Strategy_Report.pdf`. PyMuPDF parses text -> SentenceTransformers generates embeddings -> Saved to ChromaDB.
Step 6: User opens Copilot Chat and asks "Why is Technology sub-category profit dropping?".
Step 7: Copilot queries dataset context + ChromaDB vector chunks -> streams answer in real time.
Step 8: User switches to NL Forecast tab and types "What if we reduce discounts by 10%?".
Step 9: Simulation engine computes +$42,000 profit growth -> Llama 3 generates executive summary narrative.
Step 10: User clicks "Generate Conclusion" -> Generates full executive report and prints PDF artifact.
```

---

# 14. Technical Interview Questions & Model Answers

### HR & Behavioral Questions
**Q: Tell me about yourself and your role in this project.**
*Model Answer:* "I am a Full-Stack AI Engineer and Data Architect. In this project, I designed and implemented the end-to-end architecture of **AI Decision Copilot**—ranging from Flask REST APIs, scikit-learn machine learning pipelines, and ChromaDB vector search to local Ollama LLM orchestration and a high-performance Vanilla JS glassmorphic SPA frontend."

### Data Analytics & Data Engineering Questions
**Q: How do you handle missing values and outliers in your data pipeline?**
*Model Answer:* "In `backend/app.py`, missing numerical values are median-imputed to avoid skewness from extreme values, while categorical fields use mode imputation. Outliers are detected using $Z$-score calculation ($|Z| > 3.0$), identifying anomalies in profit and sales distributions."

### Machine Learning Questions
**Q: Why did you train multiple regression models instead of choosing just one?**
*Model Answer:* "Different datasets exhibit different structural patterns (linear vs non-linear). By evaluating Linear Regression, Decision Tree, and Random Forest simultaneously on an 80/20 train-test split, the system dynamically selects the model with the highest $R^2$ score and lowest RMSE, guaranteeing optimal prediction accuracy for any uploaded CSV."

### GenAI & RAG Questions
**Q: How does your RAG implementation avoid hallucination and ensure source accuracy?**
*Model Answer:* "Our RAG pipeline in `backend/genai/rag.py` splits ingested documents into 500-character chunks with 50-character overlap, vectorizes them using `sentence-transformers/all-MiniLM-L6-v2`, and stores them in ChromaDB. During user queries, similarity search retrieves top-$K$ matching chunks, which are injected directly into the LLM system prompt with strict instructions: *'Base your answer strictly on the provided document excerpts. Cite chunk numbers.'* This grounds the model and eliminates hallucinations."

---

# 15. Technical Skills Demonstrated

- **Languages & Core:** Python 3.11+, JavaScript (ES6+), HTML5, Vanilla CSS3.
- **Data Engineering & Analysis:** Pandas, NumPy, Scipy, Data Cleaning, Outlier Detection, Feature Scaling.
- **Machine Learning:** Scikit-Learn (Regression, Model Selection, Metrics, Feature Importance).
- **Generative AI & LLMs:** Ollama, Llama 3, System Prompt Engineering, Conversation Memory Management.
- **RAG & Vector Search:** ChromaDB, Sentence-Transformers, PyMuPDF, Document Chunking, Semantic Search.
- **Backend Architecture:** Flask, REST APIs, Blueprints, SSE Streaming (`stream_with_context`), Flask-Limiter, Threading.
- **DevOps & Cloud:** Docker, Docker Compose, Git, GitHub, Render, Vercel.

---

# 16. Challenges Faced & Solutions

1. **Challenge: Vercel 10-Second Gateway Timeout on Long LLM Generation**
   - *Root Cause:* Serverless functions on Vercel hobby tier abort connections lasting longer than 10 seconds.
   - *Solution:* Implemented Server-Sent Events (SSE) streaming (`mimetype='text/event-stream'`) in Flask and updated `app.js` to point directly to Render backend endpoints for heavy operations, keeping connections alive via token streaming.

2. **Challenge: Windows OpenBLAS Memory Allocation Crashes**
   - *Root Cause:* Concurrent CPU thread contention during matrix math operations in `numpy`/`scipy` on multi-core Windows processors.
   - *Solution:* Programmatically set environment variables at the very top of `run.py` and `backend/app.py`: `OPENBLAS_NUM_THREADS="1"`, `OMP_NUM_THREADS="1"`, and `MKL_NUM_THREADS="1"`.

---

# 17. Future Enhancements

1. **Autonomous Tool-Calling Agents:** Upgrade Multi-Agent system to support ReAct loop tools (e.g., executing code in sandbox containers to plot live dynamic charts).
2. **Fine-Tuned Domain LLMs:** Fine-tune open LLMs (e.g., Llama 3 8B) on corporate financial terminology for specialized domain precision.
3. **Role-Based Access Control (RBAC):** Add JWT authentication and column-level data masking for multi-tenant enterprise deployments.

---

# 18. Resume Descriptions

### 2-Line Version
> Engineered an enterprise GenAI analytics platform (Flask, Vanilla JS, Scikit-Learn, Ollama Llama 3, ChromaDB) integrating ML profit forecasting, RAG document search, and 5-agent parallel strategic analysis. Reduced analysis cycles from days to seconds with zero third-party token costs.

### 5-Line Version
> • Built **AI Decision Copilot**, a full-stack decision intelligence dashboard using Python Flask, Vanilla JS (Glassmorphic SPA), and Scikit-Learn.  
> • Implemented multi-model machine learning regression pipelines (Linear Regression, Decision Tree, Random Forest) with automated best-model selection ($R^2$, RMSE) and feature importance breakdowns.  
> • Designed an on-premise RAG pipeline using ChromaDB and Sentence-Transformers (`all-MiniLM-L6-v2`) for semantic search over corporate PDF/DOCX/PPTX documents.  
> • Orchestrated private local GenAI inference via Ollama (Llama 3) featuring real-time SSE streaming chat, multi-agent parallel synthesis, and natural language scenario forecasting.  
> • Containerized with Docker and deployed web services on Render and Vercel with automated CI/CD workflows.

### ATS-Optimized Version
> **AI & Full-Stack Software Architect | AI Decision Copilot**  
> *Technologies:* Python, Flask, JavaScript, HTML5, CSS3, Pandas, NumPy, Scikit-Learn, Matplotlib, Ollama, Llama 3, RAG, ChromaDB, Sentence-Transformers, Docker, Git, Render, Vercel.  
> - Designed and deployed a production-ready decision intelligence web platform capable of processing 10,000+ record tabular business datasets and unstructured enterprise reports.  
> - Developed automated machine learning regression pipelines evaluating models via $R^2$ and RMSE metrics to predict financial profit and quantify discount erosion risks.  
> - Engineered an enterprise RAG knowledge base leveraging vector embeddings (ChromaDB) and local LLMs (Ollama Llama 3) for private, hallucination-free document Q&A with source citations.  
> - Created a 5-agent parallel processing framework using Python multi-threading to generate cross-functional executive strategies (Sales, Finance, Marketing, Operations, CEO).  
> - Architected responsive glassmorphic SPA frontend supporting Server-Sent Events (SSE) token streaming and PDF executive report generation.

---

# 19. Technical Viva & Elevator Pitch Prep

### 30-Second Version
"I built **AI Decision Copilot**, a full-stack decision intelligence platform that turns business CSVs and PDF documents into actionable executive strategies. It combines Scikit-Learn machine learning to predict profit, a local Ollama LLM with ChromaDB vector search for private document Q&A, and a 5-agent AI panel that synthesizes cross-departmental recommendations—all delivered via a real-time glassmorphic dashboard."

### 2-Minute Version
"My project, **AI Decision Copilot**, addresses a key business challenge: traditional dashboards show what happened, but cannot explain why or predict future scenario impacts. 

I built the backend using Python Flask with modular Blueprints. For tabular data, it performs automated cleaning, $Z$-score outlier detection, and trains Linear Regression, Decision Tree, and Random Forest models—automatically selecting the top model based on $R^2$ scores.

For unstructured data, I built a RAG pipeline using PyMuPDF, `sentence-transformers`, and ChromaDB vector database. Users can upload reports and ask natural language questions with full source citations.

Using Ollama with Llama 3, the system runs local LLM inference for privacy. It features an AI Copilot chat streaming via SSE, a 5-agent executive strategy panel (Sales, CFO, CMO, COO, CEO), and a What-If forecasting simulator. The frontend is a responsive Vanilla JS SPA hosted on Vercel, connected to a Render backend."

---

# 20. Technical Deep Dive (Code Walkthrough)

### 1. Ollama Streaming SSE Handler (`backend/genai/copilot.py`)
```python
@copilot_bp.route("/chat", methods=["POST"])
def chat_stream_endpoint():
    data = request.get_json(silent=True) or {}
    user_message = data.get("message", "").strip()
    session_id = data.get("session_id") or str(uuid.uuid4())
    
    # Context injection & conversation memory
    history = _sessions.setdefault(session_id, [])
    history.append({"role": "user", "content": user_message})
    
    def sse_generator():
        for token in chat_stream(messages=messages, model=model_override):
            yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'session_id': session_id})}\n\n"

    return Response(stream_with_context(sse_generator()), mimetype="text/event-stream")
```
*How it works:* The endpoint receives a user query, retrieves session history, constructs a context-aware system prompt, and calls `chat_stream()` which yields tokens as they arrive from Ollama's HTTP stream. `stream_with_context` keeps the Flask WSGI thread open while streaming SSE packets to the browser without loading the entire response into memory.

### 2. Semantic Document Search (`backend/genai/rag.py`)
```python
def query_rag(question: str, top_k: int = 5) -> dict:
    # 1. Generate query embedding
    model = _get_embedding_model()
    query_vector = model.encode([question]).tolist()
    
    # 2. Query ChromaDB collection
    collection = _get_chroma_collection()
    results = collection.query(query_embeddings=query_vector, n_results=top_k)
    
    # 3. Assemble context & prompt LLM
    context_str = "\n---\n".join(results['documents'][0])
    prompt = f"Context:\n{context_str}\n\nQuestion: {question}\nAnswer:"
    answer = generate(prompt=prompt)
    return {"answer": answer, "citations": results['metadatas'][0]}
```
*How it works:* The user's string question is encoded into a 384-dimensional dense vector via `sentence-transformers`. ChromaDB performs an HNSW vector index search to retrieve the top 5 nearest text chunks by cosine similarity. These chunks are formatted into a context block and fed to Ollama to generate an exact, grounded answer with chunk metadata citations.
