/* ============================================================
   genai.js  —  GenAI Features Frontend
   AI Business Copilot, RAG, Multi-Agent, NL Forecast, AI Strategy
   Loaded after app.js — shares API, activeDataset, fmt, showToast
   ============================================================ */

// ── GenAI API base ────────────────────────────────────────────
const GENAI_API = (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1")
  ? ""   // Use relative paths since API is already set
  : "https://ai-based-decision-intelligence-and-hh6v.onrender.com";

async function genAIFetch(path, opts = {}) {
  const res = await fetch(GENAI_API + path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }));
    throw new Error(err.error || res.statusText);
  }
  return res.json();
}

// ── Shared state ──────────────────────────────────────────────
let _copilotSessionId = null;
let _copilotModel     = null;
let _ollamaStatus     = null;   // cached status

// ── Ollama status check ───────────────────────────────────────
async function checkOllama() {
  if (_ollamaStatus) return _ollamaStatus;
  try {
    _ollamaStatus = await genAIFetch("/api/genai/status");
    return _ollamaStatus;
  } catch (e) {
    return { ollama_available: false, setup_hint: "Backend not reachable." };
  }
}

function ollamaOfflineBanner() {
  return `
    <div class="ollama-offline-banner">
      <div class="ollama-offline-icon">🔌</div>
      <div>
        <div class="ollama-offline-title">Ollama is not running</div>
        <div class="ollama-offline-body">
          GenAI features require Ollama running locally. It's free and takes ~2 minutes to set up.
          <div class="ollama-setup-steps">
            <ol>
              <li>Download from <strong>ollama.com</strong></li>
              <li>Open a terminal and run: <code>ollama pull llama3</code></li>
              <li>Start: <code>ollama serve</code></li>
              <li>Refresh this page</li>
            </ol>
          </div>
        </div>
      </div>
    </div>`;
}

// ═══════════════════════════════════════════════════════════════
// AI BUSINESS COPILOT
// ═══════════════════════════════════════════════════════════════

async function loadCopilot() {
  const container = document.getElementById("copilot-content");
  container.innerHTML = `<div style="text-align:center;padding:40px;color:var(--text-secondary)">⚡ Initializing AI Copilot…</div>`;

  const status = await checkOllama();

  const offlineBanner = status.ollama_available ? "" : ollamaOfflineBanner();
  const modelOptions  = (status.available_models || []).map(m =>
    `<option value="${m}" ${m === status.active_model ? "selected" : ""}>${m}</option>`
  ).join("") || `<option value="llama3">llama3 (default)</option>`;

  _copilotModel = status.active_model || "llama3";

  container.innerHTML = `
    ${offlineBanner}
    <div class="ai-copilot-layout">
      <!-- Chat Panel -->
      <div class="chat-panel">
        <div class="chat-header">
          <div class="chat-header-title">
            <div class="chat-status-dot ${status.ollama_available ? "" : "offline"}"></div>
            🤖 AI Business Copilot
          </div>
          <button class="btn btn-secondary btn-sm" id="copilot-clear-btn" onclick="clearCopilotChat()">🗑 Clear</button>
        </div>

        <div class="chat-messages" id="chat-messages">
          <div class="chat-message assistant">
            <div class="chat-avatar">🤖</div>
            <div class="chat-bubble">
              Hi! I'm your AI Business Copilot. ${status.ollama_available
                ? `I'm running on <strong>${_copilotModel}</strong>. Ask me anything about your data!`
                : "Ollama is not running — please set it up to enable AI responses."
              }
              <br><br>Try asking me:
              <br>• <em>"What are the top weaknesses in my data?"</em>
              <br>• <em>"Explain the ML prediction results"</em>
              <br>• <em>"What should I do to improve profit?"</em>
            </div>
          </div>
        </div>

        <!-- Quick Prompt Chips -->
        <div class="quick-prompts">
          <button class="quick-prompt-chip" onclick="copilotQuickPrompt('Give me a business overview of the current dataset')">📊 Dataset Overview</button>
          <button class="quick-prompt-chip" onclick="copilotQuickPrompt('What are the key weaknesses and risks I should address?')">⚠️ Key Risks</button>
          <button class="quick-prompt-chip" onclick="copilotQuickPrompt('Explain the ML model predictions in simple terms')">🔮 Explain ML</button>
          <button class="quick-prompt-chip" onclick="copilotQuickPrompt('What are the top 3 quick wins to improve profit this quarter?')">💡 Quick Wins</button>
          <button class="quick-prompt-chip" onclick="copilotQuickPrompt('Compare sales trends across different product categories')">📈 Sales Trends</button>
        </div>

        <!-- Input Area -->
        <div class="chat-input-area">
          <textarea
            class="ai-chat-input"
            id="copilot-input"
            placeholder="Ask anything about your business data…"
            rows="1"
          ></textarea>
          <button class="chat-send-btn" id="copilot-send-btn" onclick="sendCopilotMessage()">➤</button>
        </div>
      </div>

      <!-- Sidebar -->
      <div class="copilot-sidebar">
        <div class="copilot-model-select">
          <label>🤖 AI Model</label>
          <select class="select-box" id="copilot-model-select" style="width:100%">
            ${modelOptions}
          </select>
          <div style="font-size:0.75rem;color:var(--text-secondary);margin-top:6px">
            ${status.ollama_available ? `✅ Ollama Online` : `❌ Ollama Offline`}
          </div>
        </div>

        <div class="ollama-status-box">
          <div class="ollama-status-label">🔌 System Status</div>
          <div class="${status.ollama_available ? "ollama-online" : "ollama-offline"}">
            ${status.ollama_available ? `✅ Online — ${_copilotModel}` : "❌ Offline"}
          </div>
          <div style="font-size:0.75rem;color:var(--text-secondary);margin-top:8px">
            Sessions preserved across tab switches
          </div>
        </div>

        <div class="card" style="padding:16px">
          <div class="ollama-status-label">📊 Context</div>
          <div style="font-size:0.82rem;color:var(--text-secondary)">
            ${activeDataset
              ? `Active dataset: <strong style="color:var(--accent2)">${activeDataset}</strong><br>AI has full context of your data.`
              : "No dataset selected. Load a dataset for data-aware responses."
            }
          </div>
        </div>

        <div class="card" style="padding:16px">
          <div class="ollama-status-label">💬 Conversation</div>
          <div style="font-size:0.82rem;color:var(--text-secondary)" id="copilot-turn-count">Session active</div>
          <button class="btn btn-secondary btn-sm" style="margin-top:8px;width:100%" onclick="explainDatasetWithAI()">
            🔍 Explain My Dataset
          </button>
        </div>
      </div>
    </div>`;

  // Auto-resize textarea
  const ta = document.getElementById("copilot-input");
  if (ta) {
    ta.addEventListener("keydown", e => {
      if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendCopilotMessage(); }
    });
    ta.addEventListener("input", () => {
      ta.style.height = "auto";
      ta.style.height = Math.min(ta.scrollHeight, 120) + "px";
    });
  }
}

function clearCopilotChat() {
  if (_copilotSessionId) {
    genAIFetch("/api/genai/clear-session", {
      method: "POST",
      body: JSON.stringify({ session_id: _copilotSessionId }),
    }).catch(() => {});
    _copilotSessionId = null;
  }
  const msgs = document.getElementById("chat-messages");
  if (msgs) msgs.innerHTML = `
    <div class="chat-message assistant">
      <div class="chat-avatar">🤖</div>
      <div class="chat-bubble">Chat cleared. How can I help you?</div>
    </div>`;
}

function copilotQuickPrompt(msg) {
  const ta = document.getElementById("copilot-input");
  if (ta) { ta.value = msg; sendCopilotMessage(); }
}

async function getDatasetContext() {
  if (!activeDataset) return null;
  try {
    const summary = await genAIFetch(`/api/summary/${encodeURIComponent(activeDataset)}`);
    return {
      name:         activeDataset,
      rows:         summary.total_records,
      total_sales:  summary.total_sales,
      total_profit: summary.total_profit,
      total_orders: summary.total_orders,
      column_names: summary.column_names || [],
      best_model:   summary.best_model,
      weaknesses:   summary.weaknesses || {},
    };
  } catch (e) {
    return { name: activeDataset };
  }
}

async function sendCopilotMessage() {
  const ta     = document.getElementById("copilot-input");
  const sendBtn = document.getElementById("copilot-send-btn");
  const msgs   = document.getElementById("chat-messages");
  if (!ta || !msgs) return;

  const message = ta.value.trim();
  if (!message) return;

  // Get selected model
  const modelSel = document.getElementById("copilot-model-select");
  _copilotModel  = modelSel ? modelSel.value : _copilotModel;

  // Add user message
  ta.value = "";
  ta.style.height = "auto";
  if (sendBtn) sendBtn.disabled = true;

  msgs.innerHTML += `
    <div class="chat-message user">
      <div class="chat-avatar">👤</div>
      <div class="chat-bubble">${escapeHtml(message)}</div>
    </div>`;

  // Typing indicator
  const typingId = "typing-" + Date.now();
  msgs.innerHTML += `
    <div class="chat-message assistant" id="${typingId}">
      <div class="chat-avatar">🤖</div>
      <div class="chat-bubble">
        <div class="typing-indicator"><span></span><span></span><span></span></div>
      </div>
    </div>`;
  msgs.scrollTop = msgs.scrollHeight;

  const datasetCtx = await getDatasetContext();

  try {
    const response = await fetch(GENAI_API + "/api/genai/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message,
        session_id:      _copilotSessionId,
        dataset_context: datasetCtx,
        model:           _copilotModel,
      }),
    });

    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    if (!response.body) throw new Error("No streaming response body");

    // Remove typing indicator, add streaming bubble
    const typingEl = document.getElementById(typingId);
    if (typingEl) typingEl.remove();

    const streamId = "stream-" + Date.now();
    msgs.innerHTML += `
      <div class="chat-message assistant" id="${streamId}">
        <div class="chat-avatar">🤖</div>
        <div class="chat-bubble" id="${streamId}-bubble"><span class="chat-cursor"></span></div>
      </div>`;
    msgs.scrollTop = msgs.scrollHeight;

    const bubble = document.getElementById(`${streamId}-bubble`);
    let fullText = "";
    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const chunk = decoder.decode(value, { stream: true });
      const lines = chunk.split("\n");
      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        try {
          const ev = JSON.parse(line.slice(6));
          if (ev.type === "token") {
            fullText += ev.content;
            if (bubble) bubble.innerHTML = escapeHtml(fullText) + `<span class="chat-cursor"></span>`;
            msgs.scrollTop = msgs.scrollHeight;
          } else if (ev.type === "done") {
            _copilotSessionId = ev.session_id;
            if (bubble) bubble.innerHTML = escapeHtml(fullText);
          } else if (ev.type === "error") {
            if (bubble) bubble.innerHTML = `<span style="color:var(--danger)">❌ ${escapeHtml(ev.content)}</span>`;
          }
        } catch (e) { /* malformed SSE line, skip */ }
      }
    }
    if (bubble) bubble.innerHTML = escapeHtml(fullText);

  } catch (err) {
    const typingEl = document.getElementById(typingId);
    if (typingEl) typingEl.remove();
    msgs.innerHTML += `
      <div class="chat-message assistant">
        <div class="chat-avatar">🤖</div>
        <div class="chat-bubble" style="color:var(--danger)">❌ ${escapeHtml(err.message)}</div>
      </div>`;
  } finally {
    if (sendBtn) sendBtn.disabled = false;
    msgs.scrollTop = msgs.scrollHeight;
  }
}

async function explainDatasetWithAI() {
  const datasetCtx = await getDatasetContext();
  if (!datasetCtx) { showToast("Load a dataset first", "info"); return; }
  const ta = document.getElementById("copilot-input");
  if (ta) { ta.value = "Give me a comprehensive business overview of this dataset including key insights and opportunities."; }
  await sendCopilotMessage();
}

// ═══════════════════════════════════════════════════════════════
// RAG KNOWLEDGE BASE
// ═══════════════════════════════════════════════════════════════

let _ragDocuments = [];

async function loadRAG() {
  const container = document.getElementById("rag-content");
  container.innerHTML = `<div style="text-align:center;padding:40px;color:var(--text-secondary)">⚡ Loading RAG system…</div>`;

  await refreshRagDocuments();

  renderRAGUI(container);
}

async function refreshRagDocuments() {
  try {
    const data = await genAIFetch("/api/rag/documents");
    _ragDocuments = data.documents || [];
  } catch (e) {
    _ragDocuments = [];
  }
}

function renderRAGUI(container) {
  const docListHtml = _ragDocuments.length
    ? _ragDocuments.map(doc => `
        <div class="rag-doc-item" id="rag-doc-${doc.doc_id}">
          <div>
            <div class="rag-doc-name">📄 ${escapeHtml(doc.title)}</div>
            <div class="rag-doc-meta">${doc.chunks} chunks · ${doc.filename} · ${doc.uploaded_at ? doc.uploaded_at.slice(0,10) : ''}</div>
          </div>
          <button class="rag-doc-delete" onclick="deleteRagDoc('${doc.doc_id}', '${escapeHtml(doc.title)}')" title="Remove document">🗑</button>
        </div>`).join("")
    : `<p style="color:var(--text-secondary);font-size:0.85rem;text-align:center;padding:20px">No documents uploaded yet.</p>`;

  container.innerHTML = `
    <div class="rag-layout">
      <!-- Upload & Documents Column -->
      <div>
        <div class="card" style="margin-bottom:20px">
          <div class="card-title">📤 Upload Document</div>
          <div class="rag-upload-zone" id="rag-drop-zone" onclick="document.getElementById('rag-file-input').click()">
            <div class="rag-upload-icon">📂</div>
            <div class="rag-upload-text"><strong>Click to upload</strong> or drag & drop</div>
            <div class="rag-file-types">PDF · DOCX · PPTX · TXT · CSV</div>
          </div>
          <input type="file" id="rag-file-input" accept=".pdf,.docx,.pptx,.txt,.csv" style="display:none" onchange="uploadRagFile(this)" />
          <div id="rag-upload-status" style="margin-top:10px;font-size:0.82rem;color:var(--text-secondary)"></div>
        </div>

        <div class="card">
          <div class="card-title" style="display:flex;justify-content:space-between;align-items:center">
            📚 Knowledge Base
            <span style="font-size:0.75rem;color:var(--text-secondary)">${_ragDocuments.length} document(s)</span>
          </div>
          <div class="rag-doc-list" id="rag-doc-list">${docListHtml}</div>
        </div>
      </div>

      <!-- Query Column -->
      <div>
        <div class="card">
          <div class="card-title">🔍 Query Your Documents</div>
          <div class="rag-query-area">
            <textarea
              class="rag-query-input"
              id="rag-query-input"
              placeholder="Ask a question about your uploaded documents…&#10;e.g. 'What are the key financial risks mentioned?'"
            ></textarea>
            <button class="btn btn-primary" onclick="queryRAG()">🔍 Search &amp; Answer</button>
          </div>

          <div id="rag-answer-area" style="margin-top:16px"></div>
        </div>

        <div class="card mt-4" style="background:rgba(108,99,255,0.04);border-color:rgba(108,99,255,0.15)">
          <div class="card-title">💡 Example Questions</div>
          <div style="display:flex;flex-wrap:wrap;gap:8px">
            ${[
              "What are the main risk factors?",
              "Summarize the key financial metrics",
              "What strategies are recommended?",
              "List the main conclusions",
              "What are the growth opportunities?",
            ].map(q => `<button class="nl-suggestion-chip" onclick="ragQuickQuery('${q}')">${q}</button>`).join("")}
          </div>
        </div>
      </div>
    </div>`;

  // Drag & drop
  const dropZone = document.getElementById("rag-drop-zone");
  if (dropZone) {
    dropZone.addEventListener("dragover", e => { e.preventDefault(); dropZone.classList.add("drag-over"); });
    dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
    dropZone.addEventListener("drop", e => {
      e.preventDefault();
      dropZone.classList.remove("drag-over");
      const files = e.dataTransfer.files;
      if (files.length) uploadRagFileObj(files[0]);
    });
  }
}

async function uploadRagFile(input) {
  if (!input.files.length) return;
  await uploadRagFileObj(input.files[0]);
  input.value = "";
}

async function uploadRagFileObj(file) {
  const status = document.getElementById("rag-upload-status");
  if (status) status.innerHTML = `<span style="color:var(--accent)">⏳ Uploading and processing <strong>${escapeHtml(file.name)}</strong>…</span>`;

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch(GENAI_API + "/api/rag/upload", { method: "POST", body: formData });
    if (!res.ok) { const e = await res.json(); throw new Error(e.error || res.statusText); }
    const data = await res.json();
    if (status) status.innerHTML = `<span style="color:var(--success)">✅ ${escapeHtml(data.message)}</span>`;
    showToast(`✅ ${data.title} uploaded (${data.chunks} chunks)`, "success");
    await refreshRagDocuments();
    const listEl = document.getElementById("rag-doc-list");
    if (listEl) {
      const docHtml = _ragDocuments.map(doc => `
        <div class="rag-doc-item" id="rag-doc-${doc.doc_id}">
          <div>
            <div class="rag-doc-name">📄 ${escapeHtml(doc.title)}</div>
            <div class="rag-doc-meta">${doc.chunks} chunks · ${doc.filename}</div>
          </div>
          <button class="rag-doc-delete" onclick="deleteRagDoc('${doc.doc_id}', '${escapeHtml(doc.title)}')" title="Remove">🗑</button>
        </div>`).join("") || `<p style="color:var(--text-secondary);font-size:0.85rem;text-align:center;padding:20px">No documents.</p>`;
      listEl.innerHTML = docHtml;
    }
  } catch (err) {
    if (status) status.innerHTML = `<span style="color:var(--danger)">❌ ${escapeHtml(err.message)}</span>`;
    showToast("Upload failed: " + err.message, "error");
  }
}

async function deleteRagDoc(docId, title) {
  if (!confirm(`Remove "${title}" from the knowledge base?`)) return;
  try {
    await genAIFetch(`/api/rag/documents/${docId}`, { method: "DELETE" });
    showToast(`🗑 "${title}" removed`, "info");
    const el = document.getElementById(`rag-doc-${docId}`);
    if (el) el.remove();
    _ragDocuments = _ragDocuments.filter(d => d.doc_id !== docId);
  } catch (err) {
    showToast("Delete failed: " + err.message, "error");
  }
}

function ragQuickQuery(q) {
  const inp = document.getElementById("rag-query-input");
  if (inp) { inp.value = q; queryRAG(); }
}

async function queryRAG() {
  const inp     = document.getElementById("rag-query-input");
  const ansArea = document.getElementById("rag-answer-area");
  if (!inp || !ansArea) return;

  const question = inp.value.trim();
  if (!question) { showToast("Please enter a question", "info"); return; }

  ansArea.innerHTML = `<div style="color:var(--text-secondary);font-size:0.85rem">⏳ Searching knowledge base and generating answer…</div>`;

  try {
    const data = await genAIFetch("/api/rag/query", {
      method: "POST",
      body: JSON.stringify({ question, top_k: 5 }),
    });

    const citationsHtml = (data.citations || []).map(c => `
      <div class="rag-citation">
        <div class="rag-citation-num">${c.source_num}</div>
        <div>
          <div class="rag-citation-title">${escapeHtml(c.title)} — Chunk ${c.chunk_idx + 1}
            <span style="font-size:0.7rem;color:var(--text-secondary);margin-left:6px">relevance: ${(c.relevance * 100).toFixed(0)}%</span>
          </div>
          <div class="rag-citation-snippet">${escapeHtml(c.snippet)}</div>
        </div>
      </div>`).join("");

    ansArea.innerHTML = `
      <div class="rag-answer-box">${escapeHtml(data.answer)}</div>
      ${citationsHtml ? `
        <div style="margin-top:14px">
          <div style="font-size:0.75rem;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;color:var(--accent2);margin-bottom:8px">
            📎 Sources (${data.chunks_used} chunks used)
          </div>
          <div class="rag-citations">${citationsHtml}</div>
        </div>` : ""}`;
  } catch (err) {
    ansArea.innerHTML = `<div style="color:var(--danger);font-size:0.85rem">❌ ${escapeHtml(err.message)}</div>`;
  }
}

// ═══════════════════════════════════════════════════════════════
// MULTI-AGENT STRATEGY
// ═══════════════════════════════════════════════════════════════

let _agentDatasetSummary = null;

async function loadAgents() {
  const container = document.getElementById("agents-content");

  if (!activeDataset) {
    container.innerHTML = renderAgentsEmpty();
    return;
  }

  try {
    _agentDatasetSummary = await genAIFetch(`/api/summary/${encodeURIComponent(activeDataset)}`);
  } catch (e) {
    _agentDatasetSummary = { name: activeDataset };
  }

  container.innerHTML = renderAgentsReady();
}

function renderAgentsEmpty() {
  return `
    <div class="empty-state">
      <div class="empty-icon">🕵️</div>
      <h3>Load a Dataset First</h3>
      <p>Select a dataset from the dropdown, then run the multi-agent analysis.</p>
    </div>`;
}

function renderAgentsReady() {
  const agents = [
    { key: "sales",      icon: "📈", name: "Sales Agent",      role: "Sales Analyst" },
    { key: "finance",    icon: "💰", name: "Finance Agent",    role: "CFO" },
    { key: "marketing",  icon: "🎯", name: "Marketing Agent",  role: "CMO" },
    { key: "operations", icon: "⚙️", name: "Operations Agent", role: "COO" },
    { key: "ceo",        icon: "👔", name: "CEO Agent",        role: "Chief Executive" },
  ];

  const progressItems = agents.map(a => `
    <div class="agent-progress-item" id="agent-progress-${a.key}">
      <div class="agent-progress-icon">${a.icon}</div>
      <div class="agent-progress-name">${a.name}</div>
      <div class="agent-progress-status" id="agent-status-${a.key}">Standby</div>
    </div>`).join("");

  return `
    <div class="card" style="margin-bottom:22px">
      <div class="card-title">🕵️ Multi-Agent Analysis — ${escapeHtml(_agentDatasetSummary?.name || activeDataset)}</div>
      <p style="color:var(--text-secondary);font-size:0.85rem;margin-bottom:16px">
        5 AI agents will analyze your dataset from their specialized perspective.
        The CEO agent synthesizes all findings into a unified strategy.
      </p>
      <div class="agent-progress-list">${progressItems}</div>
      <button class="btn btn-primary" id="btn-run-agents" onclick="runAgentAnalysis()">
        🚀 Run Multi-Agent Analysis
      </button>
    </div>
    <div id="agents-result-area"></div>`;
}

async function runAgentAnalysis() {
  const btn      = document.getElementById("btn-run-agents");
  const resultArea = document.getElementById("agents-result-area");
  if (!btn || !resultArea) return;

  btn.disabled   = true;
  btn.textContent = "⏳ Running agents…";
  resultArea.innerHTML = "";

  const agentKeys = ["sales", "finance", "marketing", "operations"];
  agentKeys.forEach(k => setAgentStatus(k, "running"));

  try {
    // Use streaming endpoint
    const response = await fetch(GENAI_API + "/api/agents/analyze/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        dataset_summary: {
          name:         _agentDatasetSummary?.name || activeDataset,
          rows:         _agentDatasetSummary?.total_records,
          total_sales:  _agentDatasetSummary?.total_sales,
          total_profit: _agentDatasetSummary?.total_profit,
          total_orders: _agentDatasetSummary?.total_orders,
          column_names: _agentDatasetSummary?.column_names || [],
          profit_margin: _agentDatasetSummary?.profit_margin,
          best_model:   _agentDatasetSummary?.best_model,
          weaknesses:   _agentDatasetSummary?.weaknesses || {},
        },
      }),
    });

    if (!response.ok) throw new Error(`HTTP ${response.status}`);

    const reader  = response.body.getReader();
    const decoder = new TextDecoder();
    let   agentResults = [];
    let   ceoResult    = null;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const chunk = decoder.decode(value, { stream: true });
      for (const line of chunk.split("\n")) {
        if (!line.startsWith("data: ")) continue;
        try {
          const ev = JSON.parse(line.slice(6));
          if (ev.type === "progress") {
            setAgentStatus(ev.agent, "running");
          } else if (ev.type === "agent_done") {
            setAgentStatus(ev.agent, ev.result?.status === "error" ? "error" : "done");
            agentResults.push(ev.result);
            // Render card immediately
            renderSingleAgentCard(ev.result, resultArea);
          } else if (ev.type === "ceo_start") {
            setAgentStatus("ceo", "running");
          } else if (ev.type === "ceo_done") {
            setAgentStatus("ceo", "done");
            ceoResult = ev.result;
          } else if (ev.type === "done") {
            // Render CEO card
            if (ceoResult) renderCEOCard(ceoResult, resultArea);
          }
        } catch (e) { /* skip */ }
      }
    }

  } catch (err) {
    resultArea.innerHTML = `<div class="empty-state"><div class="empty-icon">❌</div><h3>Analysis Failed</h3><p>${escapeHtml(err.message)}</p></div>`;
    ["sales","finance","marketing","operations","ceo"].forEach(k => setAgentStatus(k, "error"));
  } finally {
    btn.disabled = false;
    btn.textContent = "🔄 Re-run Analysis";
  }
}

function setAgentStatus(key, status) {
  const el  = document.getElementById(`agent-progress-${key}`);
  const txt = document.getElementById(`agent-status-${key}`);
  if (!el || !txt) return;
  el.className  = `agent-progress-item ${status}`;
  const labels  = { running: "⚡ Running…", done: "✅ Complete", error: "❌ Error", standby: "Standby" };
  txt.textContent = labels[status] || status;
}

function renderSingleAgentCard(result, container) {
  const el = document.getElementById("agents-result-area");
  if (!el) return;
  if (!el.querySelector(".agent-results-grid")) {
    el.innerHTML = `<div class="agent-results-grid" id="agent-cards-grid"></div>`;
  }
  const grid = document.getElementById("agent-cards-grid");
  if (!grid) return;
  const card = document.createElement("div");
  card.className = "agent-card";
  card.innerHTML = `
    <div class="agent-card-header">
      <div class="agent-card-icon">${result.icon}</div>
      <div>
        <div class="agent-card-name">${escapeHtml(result.name)}</div>
        <div class="agent-card-role">${escapeHtml(result.role)}</div>
      </div>
    </div>
    <div class="agent-card-body">${escapeHtml(result.analysis)}</div>`;
  grid.appendChild(card);
}

function renderCEOCard(ceo, container) {
  const el = document.getElementById("agents-result-area");
  if (!el) return;
  const ceoEl = document.createElement("div");
  ceoEl.className = "ceo-card mt-4";
  ceoEl.innerHTML = `
    <div class="ceo-card-header">
      <div class="ceo-icon">${ceo.icon}</div>
      <div>
        <div class="ceo-title">👔 ${escapeHtml(ceo.name)} — Unified Strategy</div>
        <div class="ceo-subtitle">Synthesis of all specialist agent analyses</div>
      </div>
    </div>
    <div class="ceo-body">${escapeHtml(ceo.analysis)}</div>`;
  el.appendChild(ceoEl);
}

// ═══════════════════════════════════════════════════════════════
// NL FORECASTING
// ═══════════════════════════════════════════════════════════════

let _forecastExamples = [];
let _forecastSummary  = null;

async function loadAIForecast() {
  const container = document.getElementById("aiforecast-content");
  container.innerHTML = `<div style="text-align:center;padding:40px;color:var(--text-secondary)">⚡ Loading forecasting engine…</div>`;

  // Load examples
  try {
    const ex = await genAIFetch("/api/genai/forecast/examples");
    _forecastExamples = ex.examples || [];
  } catch (e) {
    _forecastExamples = ["What if sales increase by 15%?", "What if we reduce discounts by 20%?"];
  }

  // Load dataset summary if dataset is selected
  if (activeDataset) {
    try {
      _forecastSummary = await genAIFetch(`/api/summary/${encodeURIComponent(activeDataset)}`);
    } catch (e) {
      _forecastSummary = null;
    }
  }

  const chipHtml = _forecastExamples.slice(0, 6).map(q =>
    `<button class="nl-suggestion-chip" onclick="forecastSuggestion('${escapeJs(q)}')">${escapeHtml(q)}</button>`
  ).join("");

  container.innerHTML = `
    <div class="nl-forecast-layout">
      <div class="nl-input-area">
        <div class="card-title" style="margin-bottom:14px">🔮 What-If Question</div>
        <textarea
          class="nl-query-input"
          id="nl-query-input"
          placeholder="Ask a what-if question in plain English…&#10;e.g. 'What happens if sales increase by 15%?'"
        ></textarea>
        <button class="btn btn-primary" style="width:100%" onclick="runForecast()">🚀 Run Forecast</button>

        <div style="margin-top:18px">
          <div style="font-size:0.75rem;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;color:var(--text-secondary);margin-bottom:10px">💡 Try These</div>
          <div class="nl-suggestions">${chipHtml}</div>
        </div>

        ${!activeDataset ? `
          <div style="margin-top:14px;padding:12px 14px;background:rgba(247,151,30,0.08);border:1px solid rgba(247,151,30,0.25);border-radius:8px;font-size:0.83rem;color:var(--warning)">
            ⚠️ Load a dataset for accurate forecasts using your actual data.
          </div>` : `
          <div style="margin-top:14px;padding:12px 14px;background:rgba(74,222,128,0.06);border:1px solid rgba(74,222,128,0.2);border-radius:8px;font-size:0.83rem;color:var(--success)">
            ✅ Using data from: <strong>${escapeHtml(activeDataset)}</strong>
          </div>`}
      </div>

      <div class="forecast-result-card" id="forecast-result">
        <div class="empty-state" style="padding:40px 20px">
          <div class="empty-icon">🔮</div>
          <h3>Enter a What-If Question</h3>
          <p>Type your forecast question or pick a suggestion on the left.</p>
        </div>
      </div>
    </div>`;
}

function forecastSuggestion(q) {
  const inp = document.getElementById("nl-query-input");
  if (inp) inp.value = q;
}

async function runForecast() {
  const inp    = document.getElementById("nl-query-input");
  const result = document.getElementById("forecast-result");
  if (!inp || !result) return;

  const query = inp.value.trim();
  if (!query) { showToast("Please enter a forecast question", "info"); return; }

  result.innerHTML = `<div style="text-align:center;padding:40px;color:var(--text-secondary)">⏳ Running simulation…</div>`;

  try {
    const data = await genAIFetch("/api/genai/forecast/query", {
      method: "POST",
      body: JSON.stringify({
        query,
        dataset_summary: _forecastSummary ? {
          name:         _forecastSummary.name || activeDataset,
          total_sales:  _forecastSummary.total_sales,
          total_profit: _forecastSummary.total_profit,
          total_orders: _forecastSummary.total_orders,
        } : { name: "Sample Business", total_sales: 1000000, total_profit: 150000, total_orders: 5000 },
      }),
    });

    const sim = data.simulation || {};
    const profDelta = sim.profit_delta || 0;
    const saleDelta = sim.sales_delta  || 0;
    const isProfit  = profDelta >= 0;
    const isSales   = saleDelta >= 0;

    result.innerHTML = `
      <div style="margin-bottom:16px">
        <div style="font-size:0.75rem;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;color:var(--accent2);margin-bottom:8px">📊 Scenario</div>
        <div style="font-weight:600;font-size:0.95rem">${escapeHtml(sim.scenario_description || query)}</div>
      </div>

      <div class="forecast-sim-grid">
        <div class="forecast-sim-item">
          <div class="forecast-sim-label">Projected Sales</div>
          <div class="forecast-sim-value">${fmt(sim.projected_sales)}</div>
          <div class="forecast-sim-delta ${isSales ? "positive" : "negative"}">
            ${isSales ? "▲" : "▼"} ${fmt(Math.abs(saleDelta))} vs current
          </div>
        </div>
        <div class="forecast-sim-item">
          <div class="forecast-sim-label">Projected Profit</div>
          <div class="forecast-sim-value" style="color:${isProfit ? "var(--success)" : "var(--danger)"}">${fmt(sim.projected_profit)}</div>
          <div class="forecast-sim-delta ${isProfit ? "positive" : "negative"}">
            ${isProfit ? "▲" : "▼"} ${fmt(Math.abs(profDelta))} (${Math.abs(sim.pct_change || 0).toFixed(1)}%)
          </div>
        </div>
        <div class="forecast-sim-item">
          <div class="forecast-sim-label">Current Sales</div>
          <div class="forecast-sim-value">${fmt(sim.current_sales)}</div>
        </div>
        <div class="forecast-sim-item">
          <div class="forecast-sim-label">Current Profit</div>
          <div class="forecast-sim-value">${fmt(sim.current_profit)}</div>
        </div>
      </div>

      ${data.narrative ? `
        <div class="forecast-narrative">${escapeHtml(data.narrative)}</div>` : ""}

      <div style="margin-top:12px;display:flex;gap:10px;align-items:center;flex-wrap:wrap">
        <span class="forecast-confidence">🎯 Confidence: ${sim.confidence || 70}%</span>
        ${!data.ollama_ok ? `<span style="font-size:0.75rem;color:var(--warning)">⚠️ AI narration unavailable (Ollama offline)</span>` : ""}
      </div>`;

  } catch (err) {
    result.innerHTML = `<div class="empty-state"><div class="empty-icon">❌</div><h3>Forecast Failed</h3><p>${escapeHtml(err.message)}</p></div>`;
  }
}

// ═══════════════════════════════════════════════════════════════
// AI STRATEGY GENERATOR
// ═══════════════════════════════════════════════════════════════

let _strategyFocus   = "overall";
let _strategySummary = null;

async function loadAIStrategy() {
  const container = document.getElementById("aistrategy-content");

  const status = await checkOllama();

  if (activeDataset) {
    try {
      _strategySummary = await genAIFetch(`/api/summary/${encodeURIComponent(activeDataset)}`);
    } catch (e) {
      _strategySummary = null;
    }
  }

  const focuses = [
    { key: "overall",    icon: "🌐", label: "Overall Business" },
    { key: "sales",      icon: "📈", label: "Sales Growth" },
    { key: "finance",    icon: "💰", label: "Financial Optimization" },
    { key: "marketing",  icon: "🎯", label: "Marketing Strategy" },
    { key: "operations", icon: "⚙️", label: "Operational Efficiency" },
  ];

  const focusBtns = focuses.map(f =>
    `<button class="strategy-focus-btn ${f.key === _strategyFocus ? "active" : ""}"
       id="sfocus-${f.key}"
       onclick="setStrategyFocus('${f.key}')">${f.icon} ${f.label}</button>`
  ).join("");

  container.innerHTML = `
    ${!status.ollama_available ? ollamaOfflineBanner() : ""}
    <div class="strategy-gen-layout">
      <div class="strategy-controls">
        <div style="font-size:0.75rem;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;color:var(--text-secondary);margin-bottom:12px">🎯 Focus Area</div>
        ${focusBtns}
        <button class="btn btn-primary" style="width:100%;margin-top:14px" onclick="generateStrategy()">
          🚀 Generate Strategy
        </button>
        ${!activeDataset ? `
          <div style="margin-top:12px;font-size:0.78rem;color:var(--warning);text-align:center">
            ⚠️ No dataset loaded<br>Using generic strategy
          </div>` : `
          <div style="margin-top:12px;font-size:0.78rem;color:var(--success);text-align:center">
            ✅ Dataset: ${escapeHtml(activeDataset)}
          </div>`}
      </div>

      <div id="strategy-output">
        <div class="strategy-output-card">
          <div class="empty-state" style="padding:40px 20px">
            <div class="empty-icon">🎯</div>
            <h3>Ready to Generate</h3>
            <p>Select a focus area and click <strong>Generate Strategy</strong>.</p>
          </div>
        </div>
      </div>
    </div>`;
}

function setStrategyFocus(key) {
  _strategyFocus = key;
  document.querySelectorAll(".strategy-focus-btn").forEach(b => b.classList.remove("active"));
  const btn = document.getElementById(`sfocus-${key}`);
  if (btn) btn.classList.add("active");
}

async function generateStrategy() {
  const output = document.getElementById("strategy-output");
  if (!output) return;
  output.innerHTML = `<div class="strategy-output-card"><div style="text-align:center;padding:40px;color:var(--text-secondary)">⏳ AI is generating your strategy…<br><small>This may take 30-60 seconds</small></div></div>`;

  try {
    const data = await genAIFetch("/api/genai/strategy/generate", {
      method: "POST",
      body: JSON.stringify({
        focus_area: _strategyFocus,
        use_cache:  false,
        dataset_summary: _strategySummary ? {
          name:         _strategySummary.name || activeDataset,
          rows:         _strategySummary.total_records,
          total_sales:  _strategySummary.total_sales,
          total_profit: _strategySummary.total_profit,
          total_orders: _strategySummary.total_orders,
          profit_margin: _strategySummary.profit_margin,
          best_model:   _strategySummary.best_model,
          weaknesses:   _strategySummary.weaknesses || {},
        } : { name: "Business Analysis" },
      }),
    });

    // Handle both structured JSON and raw text fallback
    if (data.raw_text && !data.strategies?.length) {
      output.innerHTML = `
        <div class="strategy-output-card">
          <div class="strategy-section">
            <div class="strategy-section-title">🤖 AI Generated Strategy</div>
            <div style="font-size:0.86rem;line-height:1.8;white-space:pre-wrap">${escapeHtml(data.raw_text)}</div>
          </div>
        </div>`;
      return;
    }

    const strategiesHtml = (data.strategies || []).map(s => `
      <div class="strategy-item">
        <div class="strategy-item-title">${escapeHtml(s.title)}</div>
        <div class="strategy-item-desc">${escapeHtml(s.description)}</div>
        <div class="strategy-item-meta">
          <span class="strategy-tag ${(s.priority || "medium").toLowerCase()}">${s.priority || "MEDIUM"}</span>
          <span class="strategy-tag time">⏱ ${escapeHtml(s.timeline || "90 days")}</span>
          ${s.expected_impact ? `<span style="font-size:0.75rem;color:var(--success)">📈 ${escapeHtml(s.expected_impact)}</span>` : ""}
        </div>
      </div>`).join("");

    const risksHtml = (data.risks || []).map(r => `
      <div style="background:rgba(255,101,132,0.06);border:1px solid rgba(255,101,132,0.2);border-radius:8px;padding:12px 14px;margin-bottom:8px">
        <div style="font-weight:600;font-size:0.85rem;color:var(--danger)">⚠️ ${escapeHtml(r.risk)}</div>
        <div style="font-size:0.78rem;color:var(--text-secondary);margin-top:4px">
          Likelihood: <strong>${escapeHtml(r.likelihood)}</strong> · Mitigation: ${escapeHtml(r.mitigation)}
        </div>
      </div>`).join("");

    const actionPlanHtml = (data.action_plan || []).map(month => `
      <div class="action-plan-month">
        <div class="action-plan-month-header">
          <span class="action-plan-month-badge">Month ${month.month}</span>
          <span style="font-size:0.82rem;font-weight:600">${escapeHtml(month.theme || "")}</span>
        </div>
        ${(month.actions || []).map(a => `<div class="action-plan-item">${escapeHtml(a)}</div>`).join("")}
      </div>`).join("");

    output.innerHTML = `
      <div class="strategy-output-card">
        ${data.executive_summary ? `
          <div class="strategy-section">
            <div class="strategy-section-title">📋 Executive Summary</div>
            <div style="font-size:0.88rem;line-height:1.8;color:var(--text-secondary)">${escapeHtml(data.executive_summary)}</div>
          </div>` : ""}

        <div class="strategy-section">
          <div class="strategy-section-title">🎯 Strategic Initiatives (${(data.strategies || []).length})</div>
          ${strategiesHtml || "<p style='color:var(--text-secondary)'>No strategies generated</p>"}
        </div>

        ${data.root_cause_analysis ? `
          <div class="strategy-section">
            <div class="strategy-section-title">🔍 Root Cause Analysis</div>
            <div style="font-size:0.86rem;line-height:1.8;color:var(--text-secondary)">${escapeHtml(data.root_cause_analysis)}</div>
          </div>` : ""}

        ${risksHtml ? `
          <div class="strategy-section">
            <div class="strategy-section-title">⚠️ Risk Register (${(data.risks || []).length})</div>
            ${risksHtml}
          </div>` : ""}

        <div class="strategy-section">
          <div class="strategy-section-title">📅 90-Day Action Plan</div>
          ${actionPlanHtml || "<p style='color:var(--text-secondary)'>No action plan generated</p>"}
        </div>

        <div style="display:flex;gap:14px;flex-wrap:wrap;margin-top:8px;align-items:center">
          ${data.roi_estimate ? `<div style="font-size:0.85rem"><strong style="color:var(--success)">💰 ROI:</strong> ${escapeHtml(data.roi_estimate)}</div>` : ""}
          ${data.timeline     ? `<div style="font-size:0.85rem"><strong style="color:var(--accent2)">⏱ Timeline:</strong> ${escapeHtml(data.timeline)}</div>` : ""}
          ${data.confidence != null ? `<span class="confidence-ring"><span class="confidence-value">${data.confidence}%</span> Confidence</span>` : ""}
          ${data.priority     ? `<span class="strategy-tag ${(data.priority || "medium").toLowerCase()}">${data.priority} Priority</span>` : ""}
        </div>
      </div>`;

  } catch (err) {
    output.innerHTML = `<div class="strategy-output-card"><div class="empty-state"><div class="empty-icon">❌</div><h3>Strategy generation failed</h3><p>${escapeHtml(err.message)}</p></div></div>`;
  }
}

// ── Utility ───────────────────────────────────────────────────

function escapeHtml(text) {
  if (!text) return "";
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function escapeJs(text) {
  return String(text || "").replace(/'/g, "\\'").replace(/\n/g, " ");
}
