/* ============================================================
   app.js  —  Decision Intelligence Analyzer frontend logic
   ============================================================ */

// Dynamic API URL:
// - Use relative "/api" locally so it maps perfectly to Flask's local runner.
// - Use the direct Render backend URL in production to bypass Vercel Hobby's strict 10s gateway timeout
//   and completely avoid Vercel proxy/rewrite routing bugs.
const API = (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1")
  ? "/api"
  : "https://ai-based-decision-intelligence-and-hh6v.onrender.com/api";

// ── State ──────────────────────────────────────────────────
let activeDataset = "";
let activeTab     = "overview";
let _wasOffline   = false;   // tracks live-status for wake-up banner

// ── DOM refs ───────────────────────────────────────────────
const datasetSelect   = document.getElementById("dataset-select");
const loadDefaultsBtn = document.getElementById("btn-load-defaults");
const csvUpload       = document.getElementById("csv-upload");
const loadingOverlay  = document.getElementById("loading-overlay");
const loadingMsg      = document.getElementById("loading-msg");
const toast           = document.getElementById("toast");
const hamburger       = document.getElementById("hamburger");
const sidebar         = document.getElementById("sidebar");
const liveDot         = document.getElementById("live-indicator");

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

// Retry fetching with exponential backoff — handles Render cold-start (30-60s spin-up)
async function apiFetchWithRetry(path, opts = {}, maxRetries = 5, baseDelayMs = 8000) {
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await apiFetch(path, opts);
    } catch (e) {
      const isNetworkError = e.message === "Failed to fetch" || e.message.includes("NetworkError") || e.message.includes("502") || e.message.includes("503") || e.message.includes("504");
      if (!isNetworkError || attempt === maxRetries) throw e;
      const delay = baseDelayMs * Math.pow(1.5, attempt);
      const secondsLeft = Math.round((baseDelayMs * (Math.pow(1.5, maxRetries + 1) - 1) / (1.5 - 1) - delay * attempt) / 1000);
      showToast(`⏳ Server is waking up… retrying (attempt ${attempt + 1}/${maxRetries}). Please wait ~${Math.round(delay / 1000)}s`, "info");
      await new Promise(r => setTimeout(r, delay));
    }
  }
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
  // conclusion and compare work across all loaded datasets — no single activeDataset needed
  const needsDataset = !["compare", "overview", "conclusion", "engine", "anomaly", "explainability"].includes(tab);
  if (needsDataset && !activeDataset) {
    showToast("Please load and select a dataset first.", "info");
    return;
  }
  switch (tab) {
    case "overview":        loadOverview();        break;
    case "eda":             loadEDA();             break;
    case "weakness":        loadWeakness();        break;
    case "predict":         loadPredict();         break;
    case "strategy":        loadStrategy();        break;
    case "recommend":       loadRecommend();       break;
    case "compare":         loadCompare();         break;
    case "conclusion":      /* manual button only */ break;
    case "engine":          loadEngine();          break;
    case "anomaly":         loadAnomaly();         break;
    case "explainability":  loadExplainability();  break;

  }
}

// ── Dataset Management ─────────────────────────────────────

// ── Dropdown lock/unlock helpers ──────────────────────────
function lockDropdown() {
  datasetSelect.innerHTML = `<option value="">— load datasets first —</option>`;
  datasetSelect.disabled      = true;
  datasetSelect.style.opacity = "0.5";
  datasetSelect.style.cursor  = "not-allowed";
  datasetSelect.title         = "Click 'Load Default Datasets' first";
}

function unlockDropdown(names) {
  datasetSelect.innerHTML = `<option value="">— select a dataset —</option>`;
  names.forEach(n => {
    const opt = document.createElement("option");
    opt.value = n; opt.textContent = n;
    datasetSelect.appendChild(opt);
  });
  datasetSelect.disabled      = false;
  datasetSelect.style.opacity = "1";
  datasetSelect.style.cursor  = "pointer";
  datasetSelect.title         = "Select active dataset";
}

async function refreshDatasetList(enableDropdown = false) {
  const data  = await apiFetch("/datasets");
  const names = data.datasets;

  if (enableDropdown && names.length) {
    unlockDropdown(names);
  }
  // Never auto-enable the dropdown on a plain refresh — keep it locked
  // until the user explicitly clicks Load Default Datasets.

  renderDatasetChips(names);
  return names;
}

/** Show a pill/chip for each loaded dataset with a ✕ remove button */
function renderDatasetChips(names) {
  const bar    = document.getElementById("datasets-bar");
  const chips  = document.getElementById("dataset-chips");
  const cmpBtn = document.getElementById("btn-compare-now");

  if (!names.length) { bar.style.display = "none"; return; }
  bar.style.display = "flex";

  cmpBtn.style.display = names.length >= 2 ? "" : "none";

  chips.innerHTML = names.map(n => `
    <div class="dataset-chip ${n === activeDataset ? 'active' : ''}" data-name="${n}">
      <span class="chip-name" data-name="${n}">${n}</span>
      <button class="chip-remove" data-name="${n}" title="Remove dataset">✕</button>
    </div>`).join("");

  chips.querySelectorAll(".chip-name").forEach(el => {
    el.addEventListener("click", () => {
      activeDataset = el.dataset.name;
      datasetSelect.value = activeDataset;
      renderDatasetChips(names);
      loadTab(activeTab);
    });
  });

  chips.querySelectorAll(".chip-remove").forEach(el => {
    el.addEventListener("click", async () => {
      const name = el.dataset.name;
      try {
        await apiFetch(`/remove/${encodeURIComponent(name)}`, { method: "DELETE" });
        const remaining = names.filter(n => n !== name);
        if (activeDataset === name) {
          activeDataset = remaining.length ? remaining[0] : null;
        }
        if (remaining.length) {
          unlockDropdown(remaining);
          datasetSelect.value = activeDataset || "";
          renderDatasetChips(remaining);
          if (activeDataset) loadTab(activeTab);
        } else {
          lockDropdown();
          activeDataset = null;
        }
        showToast(`🗑️ Removed: ${name}`, "info");
      } catch(e) {
        showToast("❌ " + e.message, "error");
      }
    });
  });
}

datasetSelect.addEventListener("change", () => {
  activeDataset = datasetSelect.value || null;
  if (activeDataset) {
    loadTab(activeTab);
  }
});

loadDefaultsBtn.addEventListener("click", async () => {
  showLoading("Loading default datasets…");
  try {
    const res         = await apiFetch("/load-defaults", { method: "POST" });
    const loadedNames = res.loaded || [];
    const errNames    = res.errors || [];

    if (loadedNames.length) {
      // Explicitly unlock the dropdown with the newly loaded datasets
      unlockDropdown(loadedNames);
      activeDataset = null;
      datasetSelect.value = "";
      // Also sync the chips bar
      renderDatasetChips(loadedNames);
      showToast(`✅ Loaded: ${loadedNames.join(", ")} — now select a dataset`, "success");
    } else {
      showToast("⚠️ No datasets were loaded", "error");
    }
    if (errNames.length) {
      showToast(`⚠️ Errors: ${errNames.join(" | ")}`, "error");
    }
  } catch(e) {
    showToast("❌ " + e.message, "error");
  } finally {
    hideLoading();
  }
});

csvUpload.addEventListener("change", async () => {
  const files = Array.from(csvUpload.files);
  if (!files.length) return;

  // Keep track of ALL datasets already in the dropdown + newly uploaded
  const existingNames = Array.from(datasetSelect.options)
    .map(o => o.value).filter(v => v);
  const uploadedNames = [...existingNames];   // start from what's already shown

  showLoading(`Uploading ${files.length} file(s)…`);

  for (const file of files) {
    const name = file.name.replace(/\.csv$/i, "");
    const fd   = new FormData();
    fd.append("file", file);
    fd.append("name", name);
    try {
      loadingMsg.textContent = `Uploading ${file.name}…`;
      const res = await apiFetch("/upload", { method: "POST", body: fd });
      if (!uploadedNames.includes(name)) uploadedNames.push(name);
      showToast(`✅ ${res.message} (${res.rows} rows)`, "success");
    } catch(e) {
      showToast(`❌ ${file.name}: ${e.message}`, "error");
    }
  }

  csvUpload.value = "";   // reset so same file can be re-selected

  if (uploadedNames.length) {
    // Unlock dropdown with ALL uploaded datasets
    unlockDropdown(uploadedNames);
    renderDatasetChips(uploadedNames);

    // Auto-select the last newly uploaded file
    const lastFile = files[files.length - 1].name.replace(/\.csv$/i, "");
    activeDataset = uploadedNames.includes(lastFile) ? lastFile : uploadedNames[uploadedNames.length - 1];
    datasetSelect.value = activeDataset;

    showToast(
      `📂 ${uploadedNames.length} dataset(s) loaded — "${activeDataset}" selected`,
      "success"
    );
    loadTab(activeTab);
  }

  hideLoading();
});

// ── OVERVIEW ───────────────────────────────────────────────

async function loadOverview() {
  const grid    = document.getElementById("kpi-cards");
  const infoBox = document.getElementById("dataset-info-section");
  if (!activeDataset) {
    grid.innerHTML = `
      <div class="kpi-card"><div class="kpi-inner"><span class="kpi-label">Total Sales</span><span class="kpi-value">—</span></div></div>
      <div class="kpi-card"><div class="kpi-inner"><span class="kpi-label">Total Profit</span><span class="kpi-value">—</span></div></div>
      <div class="kpi-card"><div class="kpi-inner"><span class="kpi-label">Total Orders</span><span class="kpi-value">—</span></div></div>
      <div class="kpi-card"><div class="kpi-inner"><span class="kpi-label">Customers</span><span class="kpi-value">—</span></div></div>`;
    if (infoBox) infoBox.innerHTML = "";
    const filterBar = document.getElementById("kpi-filter-bar");
    if (filterBar) filterBar.style.display = "none";
    return;
  }
  try {
    const data = await apiFetch(`/summary/${encodeURIComponent(activeDataset)}`);
    grid.innerHTML = kpiCard("💰", "Total Sales",   fmt(data.total_sales),   "#6C63FF")
                   + kpiCard("📈", "Total Profit",  fmt(data.total_profit),  data.total_profit >= 0 ? "#4ade80" : "#FF6584")
                   + kpiCard("📦", "Total Orders",  fmtN(data.total_orders), "#43CBFF")
                   + kpiCard("👥", "Customers",     fmtN(data.total_customers), "#F7971E");
    if (infoBox && data.dataset_info) renderDatasetInfo(infoBox, data.dataset_info);
    // Load KPI filters inline — no function override needed
    loadKpiFilters();
  } catch(e) {
    showToast("❌ Overview error: " + e.message, "error");
  }
}

function renderDatasetInfo(container, info) {
  const dr = info.date_range
    ? `${info.date_range.from} → ${info.date_range.to}`
    : "N/A";
  const catBadges  = (info.categories || []).map(c => `<span class="ds-badge">${c}</span>`).join("");
  const regBadges  = (info.regions   || []).map(r => `<span class="ds-badge">${r}</span>`).join("");
  const segBadges  = (info.segments  || []).map(s => `<span class="ds-badge">${s}</span>`).join("");
  const colRows    = (info.columns   || []).map(c =>
    `<tr><td class="ds-col-name">${c.name}</td><td class="ds-col-desc">${c.description}</td></tr>`
  ).join("");
  container.innerHTML = `
  <div class="card mt-4 ds-info-card">
    <div class="ds-info-header">
      <span class="ds-info-icon">🗂️</span>
      <div>
        <h3 class="card-title" style="margin:0">Dataset Information</h3>
        <p class="ds-info-name">${info.name}</p>
      </div>
    </div>
    <div class="ds-info-purpose">📌 <strong>Purpose:</strong> ${info.purpose}</div>
    <p class="ds-info-desc">${info.description}</p>
    <div class="ds-meta-row">
      <div class="ds-meta-chip">📋 <strong>${info.rows.toLocaleString()}</strong> Rows</div>
      <div class="ds-meta-chip">📐 <strong>${info.columns_count}</strong> Columns</div>
      <div class="ds-meta-chip">📅 <strong>${dr}</strong></div>
    </div>
    ${catBadges ? `<div class="ds-badge-row"><span class="ds-badge-label">🏷️ Categories:</span>${catBadges}</div>` : ""}
    ${regBadges ? `<div class="ds-badge-row"><span class="ds-badge-label">🗺️ Regions:</span>${regBadges}</div>` : ""}
    ${segBadges ? `<div class="ds-badge-row"><span class="ds-badge-label">👥 Segments:</span>${segBadges}</div>` : ""}
    <details class="ds-columns-toggle">
      <summary>📊 View All ${info.columns_count} Columns</summary>
      <table class="ds-columns-table">
        <thead><tr><th>Column</th><th>Description</th></tr></thead>
        <tbody>${colRows}</tbody>
      </table>
    </details>
  </div>`;
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
    const analytical = data.analytical || [];
    if (!analytical.length) {
      container.innerHTML = emptyState("✅", "No weaknesses detected!", "Your dataset looks healthy.");
      return;
    }
    const sevColor = { critical: "#FF6584", warning: "#F7971E" };
    const sevLabel = { critical: "🔴 Critical", warning: "🟠 Warning" };
    let html = analytical.map(w => {
      const affectedRows = Object.entries(w.affected_items || {}).map(([k, v]) =>
        `<div class="weakness-item"><span>${k}</span><span class="val">${typeof v === "number" ? fmt(v) : v}</span></div>`
      ).join("");
      return `<div class="weakness-analytical-card">
        <div class="wa-header">
          <span class="wa-icon">${w.icon}</span>
          <div>
            <div class="wa-type" style="color:${sevColor[w.severity] || "#F7971E"}">${sevLabel[w.severity] || w.severity} · ${w.type}</div>
            <div class="wa-title">${w.title}</div>
          </div>
        </div>
        <div class="wa-section"><div class="wa-section-label">🔍 Root Cause</div><p>${w.root_cause}</p></div>
        <div class="wa-section"><div class="wa-section-label">📅 Timeline / Context</div><p>${w.timeline}</p></div>
        <div class="wa-section"><div class="wa-section-label">💥 Impact</div><p>${w.impact}</p></div>
        ${affectedRows ? `<div class="wa-section"><div class="wa-section-label">📋 Affected Items</div><div class="weakness-grid" style="margin-top:8px">${affectedRows}</div></div>` : ""}
      </div>`;
    }).join("");
    container.innerHTML = `<div class="weakness-analytical-list">${html}</div>`;
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
    const pred  = data.predictions;
    const cmp   = data.comparison   || [];
    const just  = data.justification|| {};
    const meta  = data.model_meta   || {};

    // Ranked comparison table
    const rankRows = cmp.map(m => `
      <tr class="${m.is_best ? 'best-row' : ''}">
        <td><span class="rank-badge">#${m.rank}</span></td>
        <td>${m.icon} ${m.label} ${m.is_best ? '<span class="badge-best">🏆 BEST</span>' : ''}</td>
        <td>${m.mse.toLocaleString()}</td>
        <td>${m.r2}</td>
        <td>${fmt(m.avg_predicted_profit)}</td>
      </tr>`).join("");

    // Model detail cards
    const modelCards = cmp.map(m => {
      const info = meta[m.key] || {};
      const r2Pct = Math.max(0, Math.min(100, m.r2 * 100)).toFixed(0);
      return `<div class="model-card${m.is_best ? ' best' : ''}">
        <div class="model-name">${m.icon} ${m.label}${m.is_best ? ' <span style="font-size:0.75rem;color:#F7971E">★ BEST</span>' : ''}</div>
        <div class="model-stat"><span class="label">MSE</span><span class="val">${m.mse.toLocaleString()}</span></div>
        <div class="model-stat"><span class="label">R² Score</span><span class="val">${m.r2}</span></div>
        <div class="model-stat"><span class="label">Avg Predicted Profit</span><span class="val">${fmt(m.avg_predicted_profit)}</span></div>
        <div class="r2-bar-wrap"><div class="r2-bar" style="width:${r2Pct}%"></div></div>
        ${info.strength ? `<div class="model-trait good">✅ ${info.strength}</div>` : ''}
        ${info.weakness ? `<div class="model-trait bad">⚠️ ${info.weakness}</div>` : ''}
      </div>`;
    }).join("");

    container.innerHTML = `
      <div class="predict-grid">${modelCards}</div>

      <div class="card mt-4">
        <div class="card-title">📊 Model Comparison Ranking</div>
        <div class="compare-table-wrap">
          <table class="compare-table">
            <thead><tr><th>Rank</th><th>Model</th><th>MSE (lower=better)</th><th>R²</th><th>Avg Predicted Profit</th></tr></thead>
            <tbody>${rankRows}</tbody>
          </table>
        </div>
      </div>

      <div class="card mt-4 best-model-justify">
        <div class="card-title">🏆 Best Model: ${pred.best_model} — Full Justification</div>
        <div class="justify-section"><div class="justify-label">🧐 Why This Model Is Best</div><p>${just.why_best || ''}</p></div>
        <div class="justify-section"><div class="justify-label">📊 Comparison Insight</div><p>${just.comparison_insight || ''}</p></div>
        <div class="justify-section"><div class="justify-label">💥 Business Impact</div><p>${just.impact_analysis || ''}</p></div>
        <div class="justify-section plain-english"><div class="justify-label">💬 In Plain English</div><p>${just.plain_english || ''}</p></div>
      </div>`;
  } catch(e) {
    container.innerHTML = emptyState("❌", "Prediction failed", e.message);
  } finally {
    hideLoading();
  }
}

// ── STRATEGY ───────────────────────────────────────────────

async function loadStrategy(growthPct) {
  const container = document.getElementById("strategy-content");
  if (!activeDataset) { container.innerHTML = emptyState("⚙️", "No dataset selected", "Load and select a dataset first."); return; }
  const pct = (growthPct != null) ? growthPct : 10;
  showLoading("Evaluating strategies…");
  try {
    const data = await apiFetch(`/strategy/${encodeURIComponent(activeDataset)}?growth_percent=${pct}`);
    const sim  = data.simulation;
    container.innerHTML = `
      <div class="strategy-grid">
        <div class="strategy-card">
          <div class="strategy-icon">💰</div>
          <div class="strategy-label">Profit Maximization</div>
          <div class="strategy-value ${data.profit_maximization >= 0 ? 'green' : 'red'}">${fmt(data.profit_maximization)}</div>
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
        <div class="sim-title">🔥 What-if Analysis — "What happens if we grow by ${sim.growth_percent}%?"</div>
        <div class="sim-compare">
          <div class="sim-box">
            <div class="sim-box-label">Current Profit</div>
            <div class="sim-box-val" style="color:var(--text-secondary)">${fmt(sim.current_profit)}</div>
            <div class="sim-box-sub">Current Sales: ${fmt(sim.current_sales)}</div>
          </div>
          <div class="sim-arrow">→</div>
          <div class="sim-box">
            <div class="sim-box-label">After +${sim.growth_percent}% Growth</div>
            <div class="sim-box-val" style="color:var(--success)">${fmt(sim.after_profit)}</div>
            <div class="sim-box-sub">Projected Sales: ${fmt(sim.after_sales)}</div>
          </div>
        </div>
        <div class="sim-deltas">
          <div class="sim-delta-item">📊 Profit Gain: <strong style="color:var(--success)">+${fmt(sim.profit_delta)}</strong></div>
          <div class="sim-delta-item">💵 Sales Gain: <strong style="color:#43CBFF">+${fmt(sim.sales_delta)}</strong></div>
        </div>
        <div style="margin-top:18px">
          <div class="sim-growth">📈 +${sim.growth_percent}% Simulated Growth Applied</div>
        </div>
      </div>

      <div class="card mt-4 whatif-input-card">
        <div class="card-title">🎯 Adjust Growth Scenario</div>
        <p style="color:var(--text-secondary);font-size:0.87rem;margin-bottom:14px">
          Enter a custom growth percentage (1–100%) to see its projected impact on profit and sales.
        </p>
        <div class="whatif-controls">
          <input type="number" id="growth-input" min="1" max="100" value="${sim.growth_percent}"
            class="whatif-input" placeholder="e.g. 25" />
          <span class="whatif-pct-label">%</span>
          <button class="btn btn-primary" id="btn-apply-growth">⚡ Apply</button>
        </div>
        <input type="range" id="growth-slider" min="1" max="100" value="${sim.growth_percent}" class="whatif-slider" />
        <div id="growth-slider-label" style="color:var(--text-secondary);font-size:0.85rem;margin-top:4px">
          Slider: ${sim.growth_percent}%
        </div>
      </div>`;

    // Wire up interactive controls
    const input  = document.getElementById("growth-input");
    const slider = document.getElementById("growth-slider");
    const sliderLabel = document.getElementById("growth-slider-label");
    const applyBtn = document.getElementById("btn-apply-growth");

    slider.addEventListener("input", () => {
      input.value = slider.value;
      sliderLabel.textContent = `Slider: ${slider.value}%`;
    });
    input.addEventListener("input", () => {
      const v = Math.max(1, Math.min(100, parseInt(input.value) || 10));
      slider.value = v;
      sliderLabel.textContent = `Slider: ${v}%`;
    });
    applyBtn.addEventListener("click", () => {
      const v = Math.max(1, Math.min(100, parseInt(input.value) || 10));
      loadStrategy(v);
    });

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
    const rec  = data.recommendations;
    const items = rec.items || [];

    if (!items.length) {
      container.innerHTML = emptyState("💡", "No recommendations", "Load a dataset to generate recommendations.");
      return;
    }

    const cards = items.map((item, i) => {
      const steps = (item.steps || []).map((s, si) =>
        `<li class="rec-step"><span class="rec-step-num">${si+1}</span>${s}</li>`
      ).join("");
      const dos   = (item.dos || []).map(d =>
        `<div class="rec-item"><span class="rec-bullet">✅</span><span>${d}</span></div>`).join("");
      const donts = (item.donts || []).map(d =>
        `<div class="rec-item"><span class="rec-bullet">❌</span><span>${d}</span></div>`).join("");
      return `<div class="rec-full-card">
        <div class="rec-full-header">
          <span class="rec-num">#${i+1}</span>
          <div class="rec-full-idea">${item.idea}</div>
        </div>
        <p class="rec-rationale">📌 ${item.rationale}</p>
        ${steps ? `<div class="rec-steps-section">
          <div class="rec-section-label">🛠️ Step-by-Step Actions</div>
          <ol class="rec-steps-list">${steps}</ol>
        </div>` : ''}
        <div class="rec-grid" style="margin-top:14px">
          <div class="rec-card rec-do">
            <div class="rec-card-title">✅ DO</div>
            ${dos || "<div class='rec-item'>No specific actions.</div>"}
          </div>
          <div class="rec-card rec-dont">
            <div class="rec-card-title">❌ DON'T</div>
            ${donts || "<div class='rec-item'>No issues detected.</div>"}
          </div>
        </div>
        ${item.growth_note ? `<div class="rec-growth-note">
          <span class="rec-growth-icon">📈</span>
          <div><span class="rec-growth-label">Growth Conclusion</span>${item.growth_note}</div>
        </div>` : ''}
      </div>`;
    }).join("");

    container.innerHTML = `<div class="rec-full-list">${cards}</div>`;
  } catch(e) {
    container.innerHTML = emptyState("❌", "Recommendation error", e.message);
  } finally {
    hideLoading();
  }
}

// ── COMPARE ────────────────────────────────────────────────

let _compareRetryTimer = null;

async function loadCompare() {
  const container = document.getElementById("compare-content");

  // Check datasets count first
  let datasetNames = [];
  try {
    const dsList = await apiFetch("/datasets");
    datasetNames = dsList.datasets || [];
  } catch(e) { /* ignore */ }

  if (datasetNames.length < 2) {
    hideLoading();
    container.innerHTML = emptyState("⚖️", "Need at least 2 datasets", "Upload or load 2 or more datasets, then come back here.");
    return;
  }

  showLoading("Preparing comparison… training ML models");

  // Check how many are already cached by calling warmup
  let warmupStatus = "warming";
  try {
    const wu = await apiFetch("/warmup-compare", { method: "POST" });
    warmupStatus = wu.status || "warming";
  } catch(e) { /* continue anyway */ }

  // If not ready yet, show progressive message and poll
  if (warmupStatus !== "already_ready") {
    container.innerHTML = `<div class="empty-state">
      <div class="empty-icon">⚙️</div>
      <h3>Training ML Models…</h3>
      <p style="color:var(--text-secondary);margin-bottom:16px">Training machine learning models on ${datasetNames.length} datasets. This takes 5–15 seconds on first run.</p>
      <div style="width:200px;height:6px;background:rgba(108,99,255,0.15);border-radius:4px;margin:0 auto 16px">
        <div id="compare-progress" style="width:10%;height:100%;background:var(--accent);border-radius:4px;transition:width 0.5s"></div>
      </div>
      <p id="compare-wait-msg" style="color:var(--text-secondary);font-size:0.83rem">Please wait…</p>
    </div>`;

    // Animate progress bar
    let pct = 10;
    const progBar = document.getElementById("compare-progress");
    const waitMsg = document.getElementById("compare-wait-msg");
    const progressTimer = setInterval(() => {
      pct = Math.min(90, pct + 8);
      if (progBar) progBar.style.width = pct + "%";
      if (waitMsg) waitMsg.textContent = pct < 40 ? "Loading data…" : pct < 70 ? "Training models…" : "Almost ready…";
    }, 600);

    // Wait for models to be ready (poll for up to 30s)
    await new Promise(resolve => setTimeout(resolve, 3000));
    clearInterval(progressTimer);
  }

  try {
    const data = await apiFetch("/compare");
    const { comparison, best_company, chart, cross_suggestion, best_justification: bj } = data;

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

    // ── Justification panel ─────────────────────────────────────
    let justHtml = "";
    if (bj) {
      // Key benefits
      const benefits = (bj.key_benefits || []).map(b =>
        `<div class="cj-benefit-item"><span class="cj-check">✅</span>${b}</div>`
      ).join("");

      // Algorithms
      const algos = (bj.algorithms_used || []).map(a =>
        `<div class="cj-algo-card">
          <div class="cj-algo-header"><span>${a.icon}</span><strong>${a.name}</strong></div>
          <div class="cj-algo-purpose">📌 ${a.purpose}</div>
          <div class="cj-algo-why">💡 ${a.why_used}</div>
        </div>`
      ).join("");

      // Strategic advantages
      const advantages = (bj.strategic_advantages || []).map((s, i) =>
        `<div class="cj-adv-item"><span class="cj-adv-num">${i+1}</span>${s}</div>`
      ).join("");

      justHtml = `
        <div class="cj-panel mt-4">
          <div class="cj-panel-title">🏆 Why <em>${best_company}</em> Is the Best Dataset</div>

          <div class="cj-section">
            <div class="cj-section-label">📋 Reason for Selection</div>
            <p class="cj-text">${bj.reason_for_selection}</p>
          </div>

          <div class="cj-section">
            <div class="cj-section-label">💎 Key Benefits</div>
            <div class="cj-benefits-list">${benefits}</div>
          </div>

          <div class="cj-section">
            <div class="cj-section-label">🤖 Algorithms &amp; Models Used</div>
            <div class="cj-algo-grid">${algos}</div>
          </div>

          <div class="cj-section">
            <div class="cj-section-label">📊 Model Effectiveness</div>
            <p class="cj-text">${bj.effectiveness}</p>
          </div>

          <div class="cj-section">
            <div class="cj-section-label">🚀 Strategic Advantages</div>
            <div class="cj-adv-list">${advantages}</div>
          </div>
        </div>`;
    }

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

      ${justHtml}

      <div class="cross-suggestion-box mt-4">
        <h4>🔥 Cross-Company Suggestion</h4>
        <p>${cross_suggestion}</p>
      </div>`;
  } catch(e) {
    const isNeedTwo = e.message && e.message.includes("Need at least 2");
    const isTimeout = e.message && (e.message.includes("timeout") || e.message.includes("504") || e.message.includes("502") || e.message.includes("Failed to fetch"));
    if (isNeedTwo) {
      container.innerHTML = emptyState("⚖️", "Need at least 2 datasets", "Upload or load 2 or more datasets, then come back here.");
    } else {
      container.innerHTML = `<div class="empty-state">
        <div class="empty-icon">${isTimeout ? "⏳" : "❌"}</div>
        <h3>${isTimeout ? "Server is Processing…" : "Comparison Error"}</h3>
        <p style="color:var(--text-secondary);margin-bottom:16px">${e.message}</p>
        ${isTimeout ? "<p style='color:var(--text-secondary);font-size:0.87rem'>The server is training ML models. This may take up to 30 seconds on a cold start. Please retry.</p>" : ""}
        <button class="btn btn-primary" style="margin-top:12px" onclick="loadCompare()">🔄 Retry Comparison</button>
      </div>`;
    }
  } finally {
    hideLoading();
  }
}

// ── CONCLUSION ─────────────────────────────────────────────

document.getElementById("btn-gen-conclusion").addEventListener("click", loadConclusion);
document.getElementById("btn-print-conclusion").addEventListener("click", () => window.print());

async function loadConclusion() {
  const container = document.getElementById("conclusion-content");
  const status    = document.getElementById("conclusion-status");
  const printBtn  = document.getElementById("btn-print-conclusion");

  status.textContent = "⏳ Generating…";
  showLoading("Generating AI conclusion…");

  try {
    const data = await apiFetch("/conclusion");
    printBtn.style.display = "";
    status.textContent = `✅ Generated at ${new Date().toLocaleTimeString()}`;

    const { datasets: ds, best_performer, avg_profit_margin,
            overall_conclusion, total_datasets, total_records } = data;

    // ── Summary metrics row ──────────────────────────────────
    const metricsHtml = `
      <div class="conc-metrics">
        <div class="conc-metric"><div class="conc-m-icon">📁</div><div class="conc-m-val">${total_datasets}</div><div class="conc-m-label">Datasets Analyzed</div></div>
        <div class="conc-metric"><div class="conc-m-icon">📊</div><div class="conc-m-val">${fmtN(total_records)}</div><div class="conc-m-label">Records Processed</div></div>
        <div class="conc-metric"><div class="conc-m-icon">💹</div><div class="conc-m-val">${avg_profit_margin}%</div><div class="conc-m-label">Avg Profit Margin</div></div>
        <div class="conc-metric"><div class="conc-m-icon">🏆</div><div class="conc-m-val">${best_performer}</div><div class="conc-m-label">Top Performer</div></div>
      </div>`;

    // ── Overall conclusion box ───────────────────────────────
    const overallHtml = `
      <div class="conc-summary-box">
        <div class="conc-summary-title">📋 Executive Summary</div>
        <p class="conc-summary-text">${overall_conclusion}</p>
      </div>`;

    // ── Per-dataset cards ────────────────────────────────────
    let datasetCards = "";
    for (const [name, c] of Object.entries(ds)) {
      const isBest = name === best_performer;
      const findings = c.key_findings.map(f =>
        `<li class="conc-finding-item">🔹 ${f}</li>`).join("");
      const dos   = c.recommendations.DOs.map(d =>
        `<div class="rec-item"><span class="rec-bullet">✅</span><span>${d}</span></div>`).join("");
      const donts = c.recommendations.DONTs.map(d =>
        `<div class="rec-item"><span class="rec-bullet">❌</span><span>${d}</span></div>`).join("");

      datasetCards += `
        <div class="conc-dataset-card${isBest ? " conc-best" : ""}">
          <div class="conc-ds-header">
            <span class="conc-ds-name">${name}</span>
            ${isBest ? `<span class="badge-best">🏆 BEST</span>` : ""}
          </div>

          <div class="conc-ds-stats">
            <div class="conc-stat"><span class="conc-stat-l">Total Sales</span><span class="conc-stat-v">${fmt(c.total_sales)}</span></div>
            <div class="conc-stat"><span class="conc-stat-l">Total Profit</span><span class="conc-stat-v" style="color:${c.total_profit>=0?"var(--success)":"var(--danger)"}">${fmt(c.total_profit)}</span></div>
            <div class="conc-stat"><span class="conc-stat-l">Profit Margin</span><span class="conc-stat-v">${c.profit_margin}%</span></div>
            <div class="conc-stat"><span class="conc-stat-l">Best ML Model</span><span class="conc-stat-v" style="color:var(--accent)">${c.best_model}</span></div>
            <div class="conc-stat"><span class="conc-stat-l">Predicted Profit</span><span class="conc-stat-v">${fmt(c.predicted_profit)}</span></div>
            <div class="conc-stat"><span class="conc-stat-l">After Strategy (+10%)</span><span class="conc-stat-v" style="color:var(--success)">${fmt(c.simulated_profit)}</span></div>
            <div class="conc-stat"><span class="conc-stat-l">Weaknesses Found</span><span class="conc-stat-v" style="color:var(--warning)">${c.weakness_count}</span></div>
          </div>

          <div class="conc-findings">
            <div class="conc-section-title">🔍 Key Findings</div>
            <ul class="conc-finding-list">${findings || "<li>No findings available.</li>"}</ul>
          </div>

          <div class="rec-grid" style="margin-top:14px">
            <div class="rec-card rec-do">
              <div class="rec-card-title">✅ DO — Actions</div>${dos || "<div class='rec-item'>N/A</div>"}
            </div>
            <div class="rec-card rec-dont">
              <div class="rec-card-title">❌ DON'T — Avoid</div>${donts || "<div class='rec-item'>N/A</div>"}
            </div>
          </div>
        </div>`;
    }

    // ── Project conclusion text (report-style) ───────────────
    const reportHtml = `
      <div class="conc-report-box">
        <div class="conc-report-title">📄 Project Conclusion</div>
        <div class="conc-report-body">
          <p>This project successfully demonstrates an <strong>AI-Based Decision Intelligence and Strategy Impact Analyzer</strong> that transforms raw business data into actionable intelligence.</p>
          <p>Through a comprehensive pipeline of <strong>data preprocessing → exploratory analysis → weakness detection → machine learning → strategy simulation → recommendations</strong>, the system provides end-to-end decision support.</p>
          <p>Key machine learning algorithms — <em>Linear Regression, Decision Tree, and Random Forest</em> — were evaluated and the best model was selected based on Mean Squared Error (MSE) to ensure accurate profit predictions.</p>
          <p>The multi-dataset comparison engine enables organizations to benchmark performance across business units, identify the best-performing strategy, and apply cross-company learning.</p>
          <p><strong>Outcome:</strong> The system reduces decision-making uncertainty, highlights revenue opportunities, and enables data-driven strategy execution with a projected <span style="color:var(--success)">+10% profit growth</span> through recommended strategy application.</p>
        </div>
      </div>`;

    container.innerHTML = metricsHtml + overallHtml + datasetCards + reportHtml;

  } catch(e) {
    status.textContent = "";
    container.innerHTML = emptyState("❌", "Failed to generate conclusion", e.message + " — Load datasets first.");
  } finally {
    hideLoading();
  }
}

// ── Bootstrap ──────────────────────────────────────────────

// Wake-up banner elements
const wakeupBanner = document.getElementById("wakeup-banner");
const wakeupSub    = document.getElementById("wakeup-sub");
const wakeupTimer  = document.getElementById("wakeup-timer");
const wakeupBar    = document.getElementById("wakeup-bar");

let _wakeupActive = false;    // banner is currently showing
let _serverOnline = false;    // we've had at least one successful ping

/**
 * Automatically fetch default datasets, populate dropdown, render chips,
 * select the first dataset, and switch to overview tab.
 */
async function autoLoadDefaults() {
  showLoading("Automatically loading default datasets…");
  try {
    const res         = await apiFetchWithRetry("/load-defaults", { method: "POST" }, 3, 2000);
    const loadedNames = res.loaded || [];

    if (loadedNames.length) {
      unlockDropdown(loadedNames);
      // Auto-select the first dataset to make it instantly usable
      activeDataset = loadedNames[0];
      datasetSelect.value = activeDataset;
      renderDatasetChips(loadedNames);
      
      // Auto-render overview tab
      switchTab("overview");
      showToast(`✅ Default datasets loaded automatically!`, "success");
    } else {
      showToast("⚠️ No default datasets were loaded", "error");
    }
  } catch(e) {
    showToast("❌ Auto-load failed: " + e.message, "error");
  } finally {
    hideLoading();
  }
}

/**
 * Show the warm-up banner and silently retry pinging the backend every 5s
 * for up to maxWaitMs ms. (Intrusive countdown banner disabled).
 */
async function startWakeupSequence(maxWaitMs = 90000) {
  // Silent no-op, handled gracefully by background retry loop
}

(async function init() {
  // ALWAYS start locked
  lockDropdown();
  try {
    // Quick ping with retry to see if Render is awake or waking up
    await apiFetchWithRetry("/ping", {}, 10, 4000);
    _serverOnline = true;
    
    // Check if there are already datasets loaded on the server (e.g. from previous sessions)
    const existing = await refreshDatasetList(true);
    activeDataset = "";
    datasetSelect.value = "";
    loadOverview();
    if (existing && existing.length) {
      showToast("🔌 Connected to backend server. Please select a dataset to begin.", "success");
    } else {
      showToast("🔌 Connected to backend server. Please click 'Load Default Datasets' or upload your own to begin.", "success");
    }
  } catch(e) {
    showToast("⚠️ Could not connect to backend server. Please refresh or try again later.", "error");
  }
})();




async function loadEngine() {
  const c = document.getElementById("engine-content");
  if (!activeDataset) { c.innerHTML = emptyState("🧠","No dataset selected","Load and select a dataset first."); return; }
  showLoading("Running Decision Engine…");
  try {
    const data = await apiFetch(`/decision-engine/${encodeURIComponent(activeDataset)}`);
    const { decisions, top_strategy, total_strategies } = data;

    const topHtml = `
      <div class="card" style="background:linear-gradient(135deg,rgba(74,222,128,0.1),rgba(108,99,255,0.06));border-color:rgba(74,222,128,0.3);margin-bottom:22px">
        <div class="card-title">🏆 Top Recommended Strategy</div>
        <div style="font-family:'Space Grotesk',sans-serif;font-size:1.2rem;font-weight:700;margin-bottom:10px">${top_strategy.strategy || "—"}</div>
        <div style="display:flex;gap:10px;flex-wrap:wrap">
          <span class="engine-badge roi">📈 ${top_strategy.expected_roi || ""}</span>
          <span class="engine-badge time">⏱ ${top_strategy.time_to_impact || ""}</span>
          <span class="engine-badge risk">⚠ Risk: ${top_strategy.risk_level || ""}</span>
        </div>
        <p style="color:var(--text-secondary);font-size:0.85rem;margin-top:12px">${top_strategy.rationale || ""}</p>
      </div>`;

    const cards = decisions.map(d => `
      <div class="engine-card priority-${(d.priority||"medium").toLowerCase()}">
        <div class="engine-rank">#${d.rank}</div>
        <div class="engine-strategy">${d.strategy}</div>
        <div class="engine-meta">
          <span class="engine-badge roi">📈 ${d.expected_roi}</span>
          <span class="engine-badge time">⏱ ${d.time_to_impact}</span>
          <span class="engine-badge risk">⚠ ${d.risk_level} Risk</span>
          <span class="engine-badge" style="background:rgba(108,99,255,0.12);color:var(--accent);border:1px solid rgba(108,99,255,0.3)">
            💰 +${fmt(d.revenue_impact)}
          </span>
        </div>
        <div class="engine-confidence-bar">
          <div class="engine-confidence-fill" style="width:${d.confidence}%"></div>
        </div>
        <div style="font-size:0.75rem;color:var(--text-secondary);margin-bottom:8px">Confidence: ${d.confidence}%</div>
        <div class="engine-rationale">${d.rationale}</div>
      </div>`).join("");

    c.innerHTML = topHtml + `<div class="engine-grid">${cards}</div>`;
  } catch(e) {
    c.innerHTML = emptyState("❌","Engine error",e.message);
  } finally { hideLoading(); }
}

// ── ANOMALY DETECTION ───────────────────────────────────────
async function loadAnomaly() {
  const c = document.getElementById("anomaly-content");
  if (!activeDataset) { c.innerHTML = emptyState("🚨","No dataset selected","Load and select a dataset first."); return; }
  showLoading("Scanning for anomalies…");
  try {
    const data = await apiFetch(`/anomaly/${encodeURIComponent(activeDataset)}`);
    const { anomalies, sudden_drops, alerts, risk_score, risk_level, risk_explanation, high_discount } = data;

    const scoreClass = risk_score < 20 ? "low" : risk_score < 50 ? "medium" : "high";
    const scoreLabel = risk_score < 20 ? "Low Risk" : risk_score < 50 ? "Moderate Risk" : "High Risk";

    const scoreHtml = `
      <div class="risk-score-ring">
        <div>
          <div class="risk-score-num ${scoreClass}">${risk_score}</div>
          <div style="font-size:0.78rem;color:var(--text-secondary);margin-top:4px">Risk Score / 100</div>
        </div>
        <div>
          <div style="font-size:1.1rem;font-weight:700;margin-bottom:4px">${risk_level || scoreLabel}</div>
          <div style="font-size:0.83rem;color:var(--text-secondary)">${anomalies.length} anomaly type(s) · ${sudden_drops.length} profit drop(s) detected</div>
        </div>
      </div>
      ${risk_explanation ? `<div class="card mt-4" style="background:rgba(255,101,132,0.06);border-color:rgba(255,101,132,0.2)"><p style="font-size:0.85rem;color:var(--text-secondary);line-height:1.7">${risk_explanation}</p></div>` : ""}`;

    const alertsHtml = alerts.length ? `
      <div class="card mt-4">
        <div class="card-title">🔔 Active Alerts</div>
        <div class="alert-list">${alerts.map(a => `<div class="alert-item">${a}</div>`).join("")}</div>
      </div>` : "";

    const anomHtml = anomalies.length ? `
      <div class="card mt-4">
        <div class="card-title">📊 Statistical Outliers (Z-score > 3σ)</div>
        <div class="anomaly-grid">${anomalies.map(a => `
          <div class="anomaly-card ${a.severity}">
            <div class="anomaly-header">
              <span class="anomaly-col">${a.column}</span>
              <span class="anomaly-sev ${a.severity}">${a.severity}</span>
            </div>
            <div class="anomaly-desc">${a.description}</div>
            ${a.why_risky ? `<div class="wa-section" style="margin-top:10px"><div class="wa-section-label">🔍 Why Risky</div><p style="font-size:0.83rem;color:var(--text-secondary)">${a.why_risky}</p></div>` : ""}
            ${a.impact ? `<div class="wa-section"><div class="wa-section-label">💥 Impact</div><p style="font-size:0.83rem;color:var(--text-secondary)">${a.impact}</p></div>` : ""}
            ${a.recommended_action ? `<div class="wa-section"><div class="wa-section-label">✅ Action</div><p style="font-size:0.83rem;color:var(--success)">${a.recommended_action}</p></div>` : ""}
            <div style="font-size:0.78rem;color:var(--text-secondary);margin-top:6px">Max: <strong>${fmt(a.max_deviation)}</strong> · Mean: <strong>${fmt(a.col_mean)}</strong></div>
          </div>`).join("")}
        </div>
      </div>` : "";

    const dropsHtml = sudden_drops.length ? `
      <div class="card mt-4">
        <div class="card-title">📉 Sudden Profit Drops (>25% month-over-month)</div>
        ${sudden_drops.map(d => `
          <div class="weakness-analytical-card" style="margin-bottom:14px">
            <div class="wa-header">
              <span class="wa-icon">📉</span>
              <div>
                <div class="wa-type" style="color:var(--danger)">🔴 Critical · Profit Drop</div>
                <div class="wa-title">${d.period}: ${fmt(d.from_val)} → ${fmt(d.to_val)} &nbsp;<strong style="color:var(--danger)">(${d.drop_pct}%)</strong></div>
              </div>
            </div>
            <div class="wa-section"><div class="wa-section-label">🔍 Why It Happened</div><p>${d.why_it_happened || "See root causes below."}</p></div>
            ${(d.root_causes||[]).length > 1 ? `<div class="wa-section"><div class="wa-section-label">📋 Root Causes</div><ul style="margin:4px 0;padding-left:18px;color:var(--text-secondary);font-size:0.85rem">${d.root_causes.map(rc=>`<li>${rc}</li>`).join("")}</ul></div>` : ""}
            <div class="wa-section"><div class="wa-section-label">💥 Business Impact</div><p>${d.impact}</p></div>
            <div class="wa-section"><div class="wa-section-label">✅ What To Do</div><p style="color:var(--success)">${d.what_to_do}</p></div>
          </div>`).join("")}
      </div>` : "";

    const discHtml = high_discount ? `
      <div class="card mt-4" style="background:rgba(247,151,30,0.06);border-color:rgba(247,151,30,0.3)">
        <div class="card-title">🏷️ High-Discount Risk (>40% orders)</div>
        <div style="display:flex;gap:16px;flex-wrap:wrap;margin-bottom:14px">
          <div class="conc-metric" style="min-width:100px"><div class="conc-m-val" style="color:var(--warning)">${high_discount.order_count}</div><div class="conc-m-label">Orders >40% off</div></div>
          <div class="conc-metric" style="min-width:100px"><div class="conc-m-val">${fmt(high_discount.combined_profit)}</div><div class="conc-m-label">Combined Profit</div></div>
          <div class="conc-metric" style="min-width:100px"><div class="conc-m-val">${fmt(high_discount.combined_sales)}</div><div class="conc-m-label">Combined Sales</div></div>
        </div>
        <div class="wa-section"><div class="wa-section-label">🔍 Why Risky</div><p>${high_discount.why_risky}</p></div>
        <div class="wa-section"><div class="wa-section-label">💥 If Unchecked</div><p>${high_discount.impact_if_unchecked}</p></div>
        <div class="wa-section"><div class="wa-section-label">✅ Action</div><p style="color:var(--success)">${high_discount.recommended_action}</p></div>
      </div>` : "";

    const content = scoreHtml + alertsHtml + anomHtml + dropsHtml + discHtml;
    c.innerHTML = content || emptyState("✅","No anomalies detected","Your dataset looks healthy!");
  } catch(e) {
    c.innerHTML = emptyState("❌","Anomaly error",e.message);
  } finally { hideLoading(); }
}

// ── MODEL EXPLAINABILITY ────────────────────────────────────
async function loadExplainability() {
  const c = document.getElementById("explainability-content");
  if (!activeDataset) { c.innerHTML = emptyState("🔬","No dataset selected","Load and select a dataset first."); return; }
  showLoading("Computing feature importance…");
  try {
    const data = await apiFetch(`/explain/${encodeURIComponent(activeDataset)}`);
    const { feature_importance, directions, chart, insight, plain_summary, model_behaviour, how_to_use } = data;

    const bars = feature_importance.map(f => {
      const dir = f.direction || directions[f.feature] || "positive";
      const corr = f.correlation != null ? (f.correlation > 0 ? "+" : "") + f.correlation : "";
      return `<div class="explain-bar-row">
        <span class="explain-label">${f.feature}</span>
        <div class="explain-bar-wrap">
          <div class="explain-bar-fill ${dir === "positive" ? "pos" : "neg"}" style="width:${f.pct}%"></div>
        </div>
        <span class="explain-pct" style="color:${dir==="positive"?"var(--accent)":"var(--danger)"};">${f.pct}%</span>
        <span style="font-size:0.75rem;color:var(--text-secondary)">${dir === "positive" ? "↑ Positive" : "↓ Negative"}${corr ? ` (r=${corr})` : ""}</span>
      </div>`;
    }).join("");

    const featureCards = feature_importance.map(f => {
      const dir = f.direction || directions[f.feature] || "positive";
      return `<div class="card" style="margin-bottom:14px;background:rgba(108,99,255,0.04);border-color:rgba(108,99,255,0.15)">
        <div class="card-title" style="font-size:1rem">${f.feature} <span style="font-size:0.78rem;color:${dir==="positive"?"var(--success)":"var(--danger)"};font-weight:600">${dir === "positive" ? "↑ Positive Impact" : "↓ Negative Impact"}</span></div>
        <div style="font-size:0.85rem;font-weight:600;color:var(--accent);margin-bottom:8px">Model Weight: ${f.pct}%</div>
        ${f.plain_english ? `<p style="font-size:0.84rem;color:var(--text-secondary);line-height:1.7;margin-bottom:10px">${f.plain_english}</p>` : ""}
        ${f.impact_narrative ? `<div class="wa-section"><div class="wa-section-label">💥 Impact on Profit</div><p style="font-size:0.83rem">${f.impact_narrative}</p></div>` : ""}
        ${f.when_profit_rises ? `<div class="wa-section"><div class="wa-section-label">📈 When This Variable Increases</div><p style="font-size:0.83rem">${f.when_profit_rises}</p></div>` : ""}
        ${f.what_to_watch ? `<div class="wa-section"><div class="wa-section-label">👁️ Monitor</div><p style="font-size:0.83rem;color:var(--success)">${f.what_to_watch}</p></div>` : ""}
      </div>`;
    }).join("");

    const howHtml = (how_to_use||[]).map((tip,i) =>
      `<div class="rec-item"><span class="rec-step-num">${i+1}</span>${tip}</div>`
    ).join("");

    c.innerHTML = `
      <div class="explain-insight">💡 ${insight}</div>
      ${plain_summary ? `<div class="card mt-4" style="background:rgba(74,222,128,0.06);border-color:rgba(74,222,128,0.2)"><div class="card-title">🗣️ Plain-English Summary</div><p style="font-size:0.87rem;color:var(--text-secondary);line-height:1.8">${plain_summary}</p></div>` : ""}
      ${model_behaviour ? `<div class="card mt-4" style="background:rgba(108,99,255,0.06);border-color:rgba(108,99,255,0.2)"><div class="card-title">🤖 Model Behaviour Explained</div><p style="font-size:0.86rem;color:var(--text-secondary);line-height:1.7">${model_behaviour}</p></div>` : ""}
      <div class="card mt-4">
        <div class="card-title">📊 Feature Importance Breakdown</div>
        <div class="explain-bars">${bars}</div>
      </div>
      <div class="card mt-4">
        <div class="card-title">🔍 Per-Feature Deep Dive</div>
        ${featureCards}
      </div>
      <div class="chart-card mt-4">
        <h4>📈 Feature Importance Chart (Random Forest)</h4>
        <img src="data:image/png;base64,${chart}" alt="Feature Importance" style="width:100%;max-width:500px" />
      </div>
      ${howHtml ? `<div class="card mt-4" style="background:rgba(67,203,255,0.06);border-color:rgba(67,203,255,0.2)"><div class="card-title">🛠️ How to Use These Insights</div><div>${howHtml}</div></div>` : ""}`;
  } catch(e) {
    c.innerHTML = emptyState("❌","Explainability error",e.message);
  } finally { hideLoading(); }
}

// ── KPI FILTERS ─────────────────────────────────────────────
async function loadKpiFilters() {
  if (!activeDataset) return;
  const data = await apiFetch(`/kpi-filters/${encodeURIComponent(activeDataset)}`);
  const opts = data.filter_options;
  const bar  = document.getElementById("kpi-filter-bar");
  bar.style.display = "flex";

  const populate = (selectId, values) => {
    const sel = document.getElementById(selectId);
    const cur = sel.value;
    sel.innerHTML = `<option value="">All ${selectId.replace("filter-","")}</option>`;
    values.forEach(v => { const o = document.createElement("option"); o.value = o.textContent = v; if (v===cur) o.selected=true; sel.appendChild(o); });
  };
  populate("filter-region",   opts.regions    || []);
  populate("filter-category", opts.categories || []);
  populate("filter-segment",  opts.segments   || []);

  document.getElementById("btn-apply-filters").onclick = async () => {
    const r   = document.getElementById("filter-region").value;
    const cat = document.getElementById("filter-category").value;
    const seg = document.getElementById("filter-segment").value;
    const params = new URLSearchParams();
    if (r)   params.set("region",   r);
    if (cat) params.set("category", cat);
    if (seg) params.set("segment",  seg);
    const filtered = await apiFetch(`/kpi-filters/${encodeURIComponent(activeDataset)}?${params}`);
    const grid = document.getElementById("kpi-cards");
    grid.innerHTML =
      kpiCard("💰","Total Sales",fmt(filtered.total_sales),"#6C63FF") +
      kpiCard("📈","Total Profit",fmt(filtered.total_profit),filtered.total_profit>=0?"#4ade80":"#FF6584") +
      kpiCard("📦","Total Orders",fmtN(filtered.total_orders),"#43CBFF") +
      kpiCard("💹","Profit Margin",`${filtered.profit_margin}%`,"#F7971E");
    showToast(`✅ Filters applied — ${fmtN(filtered.rows_after_filter)} rows`,"success");
  };
  document.getElementById("btn-reset-filters").onclick = () => {
    ["filter-region","filter-category","filter-segment"].forEach(id => document.getElementById(id).value="");
    loadOverview();
  };
}


const exportBtn = document.getElementById("btn-export-report");

if (exportBtn) {
  exportBtn.addEventListener("click", async () => {
    if (!activeDataset) { showToast("Select a dataset first","info"); return; }
    try {
      const meta = await apiFetch(`/report-meta/${encodeURIComponent(activeDataset)}`);
      showToast(`📄 Preparing report: ${meta.report_title}`,"info");
      setTimeout(() => window.print(), 600);
    } catch(e) {
      showToast("❌ " + e.message, "error");
    }
  });
}

// ── LIVE REFRESH STATUS ─────────────────────────────────────
async function checkLiveStatus() {
  try {
    const data = await apiFetch("/refresh-status");
    // Server just came back after being offline
    if (_wasOffline) {
      _wasOffline = false;
      showToast("✅ Backend is online!", "success");
    }
    liveDot.classList.remove("offline");
    liveDot.title = `Backend live · ${data.datasets_loaded} dataset(s) · ${data.server_time}`;
  } catch {
    liveDot.classList.add("offline");
    liveDot.title = "Backend is sleeping (Render free tier)";
    _wasOffline = true;
  }
}
checkLiveStatus();
setInterval(checkLiveStatus, 30000);

