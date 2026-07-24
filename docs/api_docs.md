# GenAI API Documentation

> Decision Intelligence Analyzer — GenAI REST API Reference

## Base URL

| Environment | URL |
|---|---|
| Local | `http://localhost:5000` |
| Production | `https://ai-based-decision-intelligence-and-hh6v.onrender.com` |

---

## System Status

### `GET /api/genai/status`

Health check — returns Ollama availability and active model.

**Response:**
```json
{
  "ollama_available": true,
  "active_model": "llama3",
  "available_models": ["llama3", "mistral", "gemma"],
  "ollama_url": "http://localhost:11434",
  "setup_hint": null
}
```

### `GET /api/genai/models`

List installed Ollama models.

**Response (200):**
```json
{
  "models": ["llama3", "mistral"],
  "active_model": "llama3",
  "count": 2
}
```

---

## AI Business Copilot

### `POST /api/genai/chat` (Streaming SSE)

Main chat endpoint with streaming Server-Sent Events.

**Request Body:**
```json
{
  "message": "What are the key weaknesses in my data?",
  "session_id": "optional-uuid",
  "dataset_context": {
    "name": "Superstore",
    "rows": 9994,
    "total_sales": 2297201,
    "total_profit": 286397,
    "column_names": ["Sales", "Profit", "Region", "Category"]
  },
  "model": "llama3"
}
```

**Response:** `text/event-stream`
```
data: {"type": "token", "content": "Based on"}
data: {"type": "token", "content": " your data"}
data: {"type": "done", "session_id": "abc-123"}
```

### `POST /api/genai/chat/sync`

Non-streaming variant. Same request body as `/chat`.

**Response (200):**
```json
{
  "reply": "Based on your dataset...",
  "session_id": "abc-123",
  "turn": 3
}
```

### `POST /api/genai/clear-session`

Clear conversation history for a session.

**Request:** `{"session_id": "abc-123"}`
**Response:** `{"cleared": true, "session_id": "abc-123"}`

### `GET /api/genai/history/<session_id>`

Retrieve conversation history.

---

## RAG Knowledge Base

### `POST /api/rag/upload`

Upload a document to the knowledge base.

**Request:** `multipart/form-data` with field `file`
- Supported: PDF, DOCX, PPTX, TXT, CSV

**Response (200):**
```json
{
  "message": "Document 'report.pdf' processed successfully",
  "doc_id": "uuid",
  "title": "report.pdf",
  "chunks": 24,
  "filename": "report.pdf"
}
```

### `GET /api/rag/documents`

List all uploaded documents.

**Response:**
```json
{
  "documents": [
    {
      "doc_id": "uuid",
      "title": "report.pdf",
      "filename": "report.pdf",
      "chunks": 24,
      "uploaded_at": "2026-07-24T10:00:00Z"
    }
  ],
  "total": 1
}
```

### `DELETE /api/rag/documents/<doc_id>`

Remove a document from the knowledge base.

### `POST /api/rag/query`

Query uploaded documents with AI-generated answers and citations.

**Request:**
```json
{
  "question": "What are the key financial risks?",
  "top_k": 5
}
```

**Response:**
```json
{
  "question": "What are the key financial risks?",
  "answer": "Based on the uploaded documents...",
  "chunks_used": 5,
  "citations": [
    {
      "source_num": 1,
      "title": "Q4 Report",
      "chunk_idx": 3,
      "snippet": "Revenue declined by 15%...",
      "relevance": 0.92
    }
  ]
}
```

---

## Multi-Agent Strategy Analysis

### `POST /api/agents/analyze`

Run multi-agent analysis (non-streaming).

**Request:**
```json
{
  "dataset_summary": {
    "name": "Superstore",
    "rows": 9994,
    "total_sales": 2297201,
    "total_profit": 286397
  }
}
```

### `POST /api/agents/analyze/stream`

Streaming multi-agent analysis with real-time progress.

**Response:** `text/event-stream`
```
data: {"type": "progress", "agent": "sales"}
data: {"type": "agent_done", "agent": "sales", "result": {...}}
data: {"type": "ceo_start"}
data: {"type": "ceo_done", "result": {...}}
data: {"type": "done"}
```

Agent result objects:
```json
{
  "name": "Sales Agent",
  "role": "Sales Analyst",
  "icon": "📈",
  "analysis": "Based on the sales data...",
  "status": "success"
}
```

---

## AI Strategy Generator

### `POST /api/genai/strategy/generate`

Generate structured business strategies with ROI estimates.

**Request:**
```json
{
  "focus_area": "sales",
  "dataset_summary": {
    "name": "Superstore",
    "total_sales": 2297201,
    "total_profit": 286397,
    "weaknesses": {}
  },
  "use_cache": true
}
```

**Response:**
```json
{
  "executive_summary": "...",
  "strategies": [
    {
      "title": "Focus on Technology Sales",
      "description": "Expand high-margin tech products...",
      "priority": "HIGH",
      "timeline": "30 days",
      "expected_impact": "+15% profit growth"
    }
  ],
  "risks": [
    {
      "risk": "Supply chain disruption",
      "likelihood": "Medium",
      "mitigation": "Diversify suppliers..."
    }
  ],
  "action_plan": [
    {
      "month": 1,
      "theme": "Foundation",
      "actions": ["Audit product portfolio", "Identify loss-makers"]
    }
  ],
  "roi_estimate": "15-25% profit improvement",
  "confidence": 78,
  "priority": "HIGH",
  "timeline": "90 days"
}
```

---

## Natural Language Forecasting

### `GET /api/genai/forecast/examples`

Returns example forecast queries for UI suggestion chips.

**Response:**
```json
{
  "examples": [
    "What happens if sales increase by 15%?",
    "What if we reduce discounts to 20%?"
  ]
}
```

### `POST /api/genai/forecast/query`

Run an NL forecast simulation.

**Request:**
```json
{
  "query": "What if sales increase by 15%?",
  "dataset_summary": {
    "name": "Superstore",
    "total_sales": 2297201,
    "total_profit": 286397,
    "total_orders": 5009
  }
}
```

**Response:**
```json
{
  "query": "What if sales increase by 15%?",
  "interpretation": "Simulating: Increase sales by 15%",
  "simulation": {
    "scenario_description": "Increase sales by 15%",
    "metric": "sales",
    "direction": "increase",
    "projected_sales": 2641781.15,
    "projected_profit": 294446.85,
    "current_sales": 2297201.0,
    "current_profit": 286397.0,
    "sales_delta": 344580.15,
    "profit_delta": 8049.85,
    "pct_change": 2.81,
    "confidence": 85
  },
  "narrative": "Based on the analysis...",
  "ollama_ok": true
}
```

### `POST /api/genai/forecast/parse`

Debug endpoint — parse NL query intent without simulation.

**Request:** `{"query": "What if sales increase by 20%?"}`
**Response:**
```json
{
  "query": "What if sales increase by 20%?",
  "parsed": {
    "metric": "sales",
    "direction": "increase",
    "percentage": 20.0,
    "absolute": null
  }
}
```

---

## Error Handling

All endpoints return structured error JSON:
```json
{
  "error": "Description of what went wrong"
}
```

| Status Code | Meaning |
|---|---|
| 200 | Success |
| 400 | Bad request (missing required fields) |
| 500 | Internal server error |
| 503 | Ollama not available |
