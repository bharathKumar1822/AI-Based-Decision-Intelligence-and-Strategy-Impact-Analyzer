/* ============================================================
   app.js  —  Decision Intelligence Analyzer frontend logic
   ============================================================ */

const API = "http://127.0.0.1:5000/api";

// ── State ──────────────────────────────────────────────────
let activeDataset = "";
let activeTab     = "overview";

// ── DOM refs ───────────────────────────────────────────────
const datasetSelect   = document.getElementById("dataset-select");
const loadDefaultsBtn = document.getElementById("btn-load-defaults");
const csvUpload       = document.getElementById("csv-upload");
const loadingOverlay  = document.getElementById("loading-overlay");
const loadingMsg      = document.getElementById("loading-msg");
const toast           = document.getElementById("toast");
const hamburger       = document.getElementById("hamburger");
const sidebar         = document.getElementById("sidebar");

// ── Helpers ────────────────────────────────────────────────

function showLoading(msg = "Analyzing…") {
  loadingMsg.textContent = msg;
  loadingOverlay.classList.remove("hidden");
}
function hideLoading() { loadingOverlay.classList.add("hidden"); }

function showToast(msg, type = "info") {
  toast.textContent = msg;
  toast.className = `toast ${type}`;
  clearTimeout(toast._timer);
  toast._timer = setTimeout(() => toast.classList.add("hidden"), 3500);
}

function fmt(n) {
  if (n == null) return "—";
  if (Math.abs(n) >= 1_000_000) return "$" + (n / 1_000_000).toFixed(2) + "M";
  if (Math.abs(n) >= 1_000)     return "$" + (n / 1_000).toFixed(1)     + "K";
  return "$" + n.toFixed(2);
}
function fmtN(n) {
  if (n == null) return "—";
  return n.toLocaleString();
}

async function apiFetch(path, opts = {}) {
  const res = await fetch(API + path, opts);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }));
    throw new Error(err.error || res.statusText);
  }
  return res.json();
}

function emptyState(icon, title, sub) {
  return `<div class="empty-state">
    <div class="empty-icon">${icon}</div>
    <h3>${title}</h3>
    <p>${sub}</p>
  </div>`;
}

// ── Sidebar & Navigation ───────────────────────────────────

hamburger.addEventListener("click", () => sidebar.classList.toggle("open"));

document.querySelectorAll(".nav-item").forEach(item => {
  item.addEventListener("click", e => {
    e.preventDefault();
    const tab = item.dataset.tab;
    switchTab(tab);
    if (window.innerWidth < 900) sidebar.classList.remove("open");
  });
});

function switchTab(tab) {
  document.querySelectorAll(".nav-item").forEach(n => n.classList.remove("active"));
  document.querySelectorAll(".tab-section").forEach(s => s.classList.remove("active"));
  document.getElementById(`nav-${tab}`).classList.add("active");
  document.getElementById(`tab-${tab}`).classList.add("active");
  activeTab = tab;
  loadTab(tab);
}

function loadTab(tab) {
  if (!activeDataset && tab !== "compare" && tab !== "overview") {
    showToast("Please load and select a dataset first.", "info");
    return;
  }
  switch (tab) {
    case "overview":  loadOverview();   break;
    case "eda":       loadEDA();        break;
    case "weakness":  loadWeakness();   break;
    case "predict":   loadPredict();    break;
    case "strategy":  loadStrategy();   break;
    case "recommend": loadRecommend();  break;
    case "compare":   loadCompare();    break;
  }
}

// ── Dataset Management ─────────────────────────────────────

async function refreshDatasetList() {
  const data = await apiFetch("/datasets");
  const names = data.datasets;
  datasetSelect.innerHTML = `<option value="">— select dataset —</option>`;
  names.forEach(n => {
    const opt = document.createElement("option");
    opt.value = n; opt.textContent = n;
    if (n === activeDataset) opt.selected = true;
    datasetSelect.appendChild(opt);
  });
  return names;
}

datasetSelect.addEventListener("change", () => {
  activeDataset = datasetSelect.value;
  if (activeDataset) {
    loadTab(activeTab);
  }
});

loadDefaultsBtn.addEventListener("click", async () => {
  showLoading("Loading default datasets…");
  try {
    const res = await apiFetch("/load-defaults", { method: "POST" });
    const names = await refreshDatasetList();
    if (names.length) {
      activeDataset = names[0];
      datasetSelect.value = activeDataset;
    }
    showToast(`✅ Loaded: ${res.loaded.join(", ")}`, "success");
    loadTab(activeTab);
  } catch(e) {
    showToast("❌ " + e.message, "error");
  } finally {
    hideLoading();
  }
});

csvUpload.addEventListener("change", async () => {
  const file = csvUpload.files[0];
  if (!file) return;
  const name = file.name.replace(".csv", "");
  const fd = new FormData();
  fd.append("file", file);
  fd.append("name", name);
  showLoading(`Uploading ${file.name}…`);
  try {
    const res = await apiFetch("/upload", { method: "POST", body: fd });
    await refreshDatasetList();
    activeDataset = name;
    datasetSelect.value = name;
    showToast(`✅ ${res.message} (${res.rows} rows)`, "success");
    loadTab(activeTab);
  } catch(e) {
    showToast("❌ " + e.message, "error");
  } finally {
    hideLoading();
    csvUpload.value = "";
  }
});

// ── OVERVIEW ───────────────────────────────────────────────

async function loadOverview() {
  const grid = document.getElementById("kpi-cards");
  if (!activeDataset) {
    grid.innerHTML = `
      <div class="kpi-card"><div class="kpi-inner"><span class="kpi-label">Total Sales</span><span class="kpi-value">—</span></div></div>
      <div class="kpi-card"><div class="kpi-inner"><span class="kpi-label">Total Profit</span><span class="kpi-value">—</span></div></div>
      <div class="kpi-card"><div class="kpi-inner"><span class="kpi-label">Total Orders</span><span class="kpi-value">—</span></div></div>
      <div class="kpi-card"><div class="kpi-inner"><span class="kpi-label">Customers</span><span class="kpi-value">—</span></div></div>`;
    return;
  }
  try {
    const data = await apiFetch(`/summary/${encodeURIComponent(activeDataset)}`);
    grid.innerHTML = kpiCard("💰", "Total Sales",   fmt(data.total_sales),   "#6C63FF")
                   + kpiCard("📈", "Total Profit",  fmt(data.total_profit),  data.total_profit >= 0 ? "#4ade80" : "#FF6584")
                   + kpiCard("📦", "Total Orders",  fmtN(data.total_orders), "#43CBFF")
                   + kpiCard("👥", "Customers",     fmtN(data.total_customers), "#F7971E");
  } catch(e) {
    showToast("❌ Overview error: " + e.message, "error");
  }
}

function kpiCard(icon, label, value, color) {
  return `<div class="kpi-card">
    <div class="kpi-inner">
      <span class="kpi-label">${label}</span>
      <span class="kpi-value" style="color:${color}">${value}</span>
      <span class="kpi-icon">${icon}</span>
    </div>
  </div>`;
}

// ── EDA ────────────────────────────────────────────────────

async function loadEDA() {
  const container = document.getElementById("eda-charts");
  if (!activeDataset) { container.innerHTML = emptyState("🔍", "No dataset selected", "Load and select a dataset first."); return; }
  container.innerHTML = `<div style="color:var(--text-secondary);padding:20px">Loading charts…</div>`;
  showLoading("Generating EDA charts…");
  try {
    const data = await apiFetch(`/eda/${encodeURIComponent(activeDataset)}`);
    const charts = data.charts;
    const labels = {
      profit_by_category: "Profit by Category",
      sales_by_region:    "Sales by Region",
      monthly_sales:      "Monthly Sales Trend",
      top_subcategories:  "Top 10 Sub-Categories by Profit",
      discount_vs_profit: "Discount vs Profit",
    };
    container.innerHTML = "";
    let count = 0;
    for (const [key, b64] of Object.entries(charts)) {
      container.innerHTML += `<div class="chart-card">
        <h4>${labels[key] || key}</h4>
        <img src="data:image/png;base64,${b64}" alt="${key}" loading="lazy" />
      </div>`;
      count++;
    }
    if (!count) container.innerHTML = emptyState("📊", "No charts available", "Dataset may be missing required columns.");
  } catch(e) {
    container.innerHTML = emptyState("❌", "Failed to load charts", e.message);
  } finally {
    hideLoading();
  }
}

// ── WEAKNESS ───────────────────────────────────────────────

async function loadWeakness() {
  const container = document.getElementById("weakness-content");
  if (!activeDataset) { container.innerHTML = emptyState("⚠️", "No dataset selected", "Load and select a dataset first."); return; }
  showLoading("Detecting weaknesses…");
  try {
    const data = await apiFetch(`/weakness/${encodeURIComponent(activeDataset)}`);
    const w = data.weaknesses;
    const titles = {
      loss_making_products:   { title: "Loss-Making Sub-Categories", icon: "📉" },
      low_performing_regions: { title: "Low-Performing Regions",     icon: "🗺️" },
      poor_profit_margins:    { title: "Poor Profit Margin Categories", icon: "📊" },
    };
    let html = `<div class="weakness-grid">`;
    let found = false;
    for (const [key, items] of Object.entries(w)) {
      if (!Object.keys(items).length) continue;
      found = true;
      const { title, icon } = titles[key] || { title: key, icon: "⚠️" };
      let rows = Object.entries(items).map(([k, v]) =>
        `<div class="weakness-item"><span>${k}</span><span class="val">${typeof v === "number" ? fmt(v) : v}</span></div>`
      ).join("");
      html += `<div class="weakness-card"><h4>${icon} ${title}</h4>${rows}</div>`;
    }
    html += `</div>`;
    container.innerHTML = found ? html : emptyState("✅", "No weaknesses detected!", "Your dataset looks healthy.");
  } catch(e) {
    container.innerHTML = emptyState("❌", "Error", e.message);
  } finally {
    hideLoading();
  }
}

// ── PREDICT ────────────────────────────────────────────────

async function loadPredict() {
  const container = document.getElementById("predict-content");
  if (!activeDataset) { container.innerHTML = emptyState("🔮", "No dataset selected", "Load and select a dataset first."); return; }
  showLoading("Running ML models…");
  try {
    const data = await apiFetch(`/predict/${encodeURIComponent(activeDataset)}`);
    const pred = data.predictions;
    const models = [
      { key: "linear_regression", label: "Linear Regression",  icon: "📐" },
      { key: "decision_tree",     label: "Decision Tree",       icon: "🌲" },
      { key: "random_forest",     label: "Random Forest",       icon: "🌳" },
    ];
    let cards = models.map(m => {
      const p = pred[m.key];
      if (!p) return "";
      const isBest = pred.best_model_key === m.key;
      const r2Pct  = Math.max(0, Math.min(100, (p.r2 * 100))).toFixed(0);
      return `<div class="model-card${isBest ? " best" : ""}">
        <div class="model-name">${m.icon} ${m.label}</div>
        <div class="model-stat"><span class="label">MSE</span><span class="val">${p.mse.toLocaleString()}</span></div>
        <div class="model-stat"><span class="label">R² Score</span><span class="val">${p.r2}</span></div>
        <div class="model-stat"><span class="label">Avg Predicted Profit</span><span class="val">${fmt(p.avg_predicted_profit)}</span></div>
        <div class="r2-bar-wrap"><div class="r2-bar" style="width:${r2Pct}%"></div></div>
      </div>`;
    }).join("");
    container.innerHTML = `
      <div class="predict-grid">${cards}</div>
      <div class="card mt-4">
        <div class="card-title">🏆 Best Model: ${pred.best_model}</div>
        <p style="color:var(--text-secondary);font-size:0.87rem">
          Selected based on lowest Mean Squared Error (MSE). 
          Predicted average profit: <strong style="color:var(--success)">${fmt(pred.best_predicted_profit)}</strong>
        </p>
      </div>`;
  } catch(e) {
    container.innerHTML = emptyState("❌", "Prediction failed", e.message);
  } finally {
    hideLoading();
  }
}

// ── STRATEGY ───────────────────────────────────────────────

async function loadStrategy() {
  const container = document.getElementById("strategy-content");
  if (!activeDataset) { container.innerHTML = emptyState("⚙️", "No dataset selected", "Load and select a dataset first."); return; }
  showLoading("Evaluating strategies…");
  try {
    const data = await apiFetch(`/strategy/${encodeURIComponent(activeDataset)}`);
    const sim = data.simulation;
    container.innerHTML = `
      <div class="strategy-grid">
        <div class="strategy-card">
          <div class="strategy-icon">💰</div>
          <div class="strategy-label">Profit Maximization</div>
          <div class="strategy-value ${data.profit_maximization >= 0 ? "green" : "red"}">${fmt(data.profit_maximization)}</div>
        </div>
        <div class="strategy-card">
          <div class="strategy-icon">📈</div>
          <div class="strategy-label">Total Sales</div>
          <div class="strategy-value blue">${fmt(data.sales_growth)}</div>
        </div>
        <div class="strategy-card">
          <div class="strategy-icon">📉</div>
          <div class="strategy-label">Loss Reduction Target</div>
          <div class="strategy-value red">${fmt(data.loss_reduction)}</div>
        </div>
        <div class="strategy-card">
          <div class="strategy-icon">🏆</div>
          <div class="strategy-label">Best ML Model</div>
          <div class="strategy-value" style="font-size:1.1rem;color:var(--accent)">${data.best_model}</div>
        </div>
      </div>

      <div class="simulation-card">
        <div class="sim-title">🔥 Strategy Simulation — "What if we apply the best strategy?"</div>
        <div class="sim-compare">
          <div class="sim-box">
            <div class="sim-box-label">Current Profit</div>
            <div class="sim-box-val" style="color:var(--text-secondary)">${fmt(sim.current_profit)}</div>
          </div>
          <div class="sim-arrow">→</div>
          <div class="sim-box">
            <div class="sim-box-label">After Strategy (10% Growth)</div>
            <div class="sim-box-val" style="color:var(--success)">${fmt(sim.after_strategy)}</div>
          </div>
        </div>
        <div style="margin-top:18px">
          <div class="sim-growth">📈 +${sim.growth_percent}% Simulated Growth</div>
        </div>
      </div>`;
  } catch(e) {
    container.innerHTML = emptyState("❌", "Strategy error", e.message);
  } finally {
    hideLoading();
  }
}

// ── RECOMMEND ──────────────────────────────────────────────

async function loadRecommend() {
  const container = document.getElementById("recommend-content");
  if (!activeDataset) { container.innerHTML = emptyState("💡", "No dataset selected", "Load and select a dataset first."); return; }
  showLoading("Generating recommendations…");
  try {
    const data = await apiFetch(`/recommend/${encodeURIComponent(activeDataset)}`);
    const rec = data.recommendations;
    const doItems   = rec.DOs.map(d   => `<div class="rec-item"><span class="rec-bullet">✅</span><span>${d}</span></div>`).join("");
    const dontItems = rec.DONTs.map(d => `<div class="rec-item"><span class="rec-bullet">❌</span><span>${d}</span></div>`).join("");
    container.innerHTML = `<div class="rec-grid">
      <div class="rec-card rec-do">
        <div class="rec-card-title">✅ DO — Strategic Actions</div>
        ${doItems || "<div class='rec-item'>No specific actions required.</div>"}
      </div>
      <div class="rec-card rec-dont">
        <div class="rec-card-title">❌ DON'T — Things to Avoid</div>
        ${dontItems || "<div class='rec-item'>No issues detected.</div>"}
      </div>
    </div>`;
  } catch(e) {
    container.innerHTML = emptyState("❌", "Recommendation error", e.message);
  } finally {
    hideLoading();
  }
}

// ── COMPARE ────────────────────────────────────────────────

async function loadCompare() {
  const container = document.getElementById("compare-content");
  showLoading("Comparing datasets…");
  try {
    const data = await apiFetch("/compare");
    const { comparison, best_company, chart, cross_suggestion } = data;
    const rows = Object.entries(comparison).map(([name, c]) =>
      `<tr>
        <td><strong>${name}</strong>${name === best_company ? `<span class="badge-best">🏆 BEST</span>` : ""}</td>
        <td>${fmt(c.total_sales)}</td>
        <td>${fmt(c.total_profit)}</td>
        <td>${fmt(c.avg_profit)}</td>
        <td>${c.best_model}</td>
        <td>${fmt(c.predicted_profit)}</td>
      </tr>`
    ).join("");

    container.innerHTML = `
      <div class="card">
        <div class="compare-table-wrap">
          <table class="compare-table">
            <thead><tr>
              <th>Dataset</th><th>Total Sales</th><th>Total Profit</th>
              <th>Avg Profit</th><th>Best Model</th><th>Predicted Profit</th>
            </tr></thead>
            <tbody>${rows}</tbody>
          </table>
        </div>
      </div>

      <div class="chart-card mt-4">
        <h4>📊 Visual Comparison</h4>
        <img src="data:image/png;base64,${chart}" alt="Company Comparison Chart" />
      </div>

      <div class="cross-suggestion-box">
        <h4>🔥 Cross-Company Suggestion</h4>
        <p>${cross_suggestion}</p>
      </div>`;
  } catch(e) {
    container.innerHTML = emptyState("⚖️", "Cannot compare", e.message + " — Load at least 2 datasets.");
  } finally {
    hideLoading();
  }
}

// ── Bootstrap ──────────────────────────────────────────────

(async function init() {
  try {
    const names = await refreshDatasetList();
    if (names.length) {
      activeDataset = names[0];
      datasetSelect.value = activeDataset;
      loadOverview();
    }
  } catch(e) {
    showToast("⚠️ Backend not reachable. Click 'Load Default Datasets'.", "error");
  }
})();
