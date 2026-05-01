/* ============================================================
   app.js  —  Decision Intelligence Analyzer frontend logic
   ============================================================ */

const API = "https://ai-based-decision-intelligence-and-hh6v.onrender.com/api";

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
  // conclusion and compare work across all loaded datasets — no single activeDataset needed
  const needsDataset = !["compare", "overview", "conclusion"].includes(tab);
  if (needsDataset && !activeDataset) {
    showToast("Please load and select a dataset first.", "info");
    return;
  }
  switch (tab) {
    case "overview":    loadOverview();    break;
    case "eda":         loadEDA();         break;
    case "weakness":    loadWeakness();    break;
    case "predict":     loadPredict();     break;
    case "strategy":    loadStrategy();    break;
    case "recommend":   loadRecommend();   break;
    case "compare":     loadCompare();     break;
    case "conclusion":  /* manual button only */  break;
  }
}

// ── Dataset Management ─────────────────────────────────────

async function refreshDatasetList() {
  const data = await apiFetch("/datasets");
  const names = data.datasets;

  // ── Rebuild dropdown ──────────────────────────
  datasetSelect.innerHTML = `<option value="">— select dataset —</option>`;
  names.forEach(n => {
    const opt = document.createElement("option");
    opt.value = n; opt.textContent = n;
    if (n === activeDataset) opt.selected = true;
    datasetSelect.appendChild(opt);
  });

  // ── Render chips bar ──────────────────────────
  renderDatasetChips(names);

  return names;
}

/** Show a pill/chip for each loaded dataset with a ✕ remove button */
function renderDatasetChips(names) {
  const bar   = document.getElementById("datasets-bar");
  const chips = document.getElementById("dataset-chips");
  const cmpBtn = document.getElementById("btn-compare-now");

  if (!names.length) { bar.style.display = "none"; return; }
  bar.style.display = "flex";

  // show Compare button only when 2+ datasets are loaded
  cmpBtn.style.display = names.length >= 2 ? "" : "none";

  chips.innerHTML = names.map(n => `
    <div class="dataset-chip ${n === activeDataset ? 'active' : ''}" data-name="${n}">
      <span class="chip-name" data-name="${n}">${n}</span>
      <button class="chip-remove" data-name="${n}" title="Remove dataset">✕</button>
    </div>`).join("");

  // click chip name → set as active dataset
  chips.querySelectorAll(".chip-name").forEach(el => {
    el.addEventListener("click", () => {
      activeDataset = el.dataset.name;
      datasetSelect.value = activeDataset;
      renderDatasetChips(names);
      loadTab(activeTab);
    });
  });

  // click ✕ → remove dataset from server
  chips.querySelectorAll(".chip-remove").forEach(el => {
    el.addEventListener("click", async () => {
      const name = el.dataset.name;
      try {
        await apiFetch(`/remove/${encodeURIComponent(name)}`, { method: "DELETE" });
        if (activeDataset === name) activeDataset = "";
        const updated = await refreshDatasetList();
        if (!activeDataset && updated.length) {
          activeDataset = updated[0];
          datasetSelect.value = activeDataset;
        }
        showToast(`🗑️ Removed: ${name}`, "info");
        loadTab(activeTab);
      } catch(e) {
        showToast("❌ " + e.message, "error");
      }
    });
  });
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
      // keep previous activeDataset if still valid, else pick first
      if (!activeDataset || !names.includes(activeDataset)) {
        activeDataset = names[0];
        datasetSelect.value = activeDataset;
      }
    }
    showToast(`✅ Loaded: ${res.loaded.join(", ")}`, "success");
    if (names.length >= 2) {
      showToast(`📊 ${names.length} datasets ready — Comparison available!`, "success");
    }
    loadTab(activeTab);
  } catch(e) {
    showToast("❌ " + e.message, "error");
  } finally {
    hideLoading();
  }
});

csvUpload.addEventListener("change", async () => {
  const files = Array.from(csvUpload.files);
  if (!files.length) return;

  let lastLoaded = "";
  let successCount = 0;

  showLoading(`Uploading ${files.length} file(s)…`);
  for (const file of files) {
    const name = file.name.replace(/\.csv$/i, "");
    const fd   = new FormData();
    fd.append("file", file);
    fd.append("name", name);
    try {
      loadingMsg.textContent = `Uploading ${file.name}…`;
      const res = await apiFetch("/upload", { method: "POST", body: fd });
      lastLoaded = name;
      successCount++;
      showToast(`✅ ${res.message} (${res.rows} rows)`, "success");
    } catch(e) {
      showToast(`❌ ${file.name}: ${e.message}`, "error");
    }
  }

  csvUpload.value = "";   // reset so same files can be re-selected

  const names = await refreshDatasetList();
  if (lastLoaded) {
    activeDataset = lastLoaded;
    datasetSelect.value = activeDataset;
  }

  // Auto-open comparison if 2+ datasets are now available
  if (names.length >= 2) {
    showToast(`📊 ${names.length} datasets loaded — Comparison is ready!`, "success");
  }

  hideLoading();
  loadTab(activeTab);
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
    return;
  }
  try {
    const data = await apiFetch(`/summary/${encodeURIComponent(activeDataset)}`);
    grid.innerHTML = kpiCard("💰", "Total Sales",   fmt(data.total_sales),   "#6C63FF")
                   + kpiCard("📈", "Total Profit",  fmt(data.total_profit),  data.total_profit >= 0 ? "#4ade80" : "#FF6584")
                   + kpiCard("📦", "Total Orders",  fmtN(data.total_orders), "#43CBFF")
                   + kpiCard("👥", "Customers",     fmtN(data.total_customers), "#F7971E");
    if (infoBox && data.dataset_info) renderDatasetInfo(infoBox, data.dataset_info);
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

async function loadCompare() {
  const container = document.getElementById("compare-content");
  showLoading("Comparing datasets…");
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
            <div class="cj-section-label">🤖 Algorithms & Models Used</div>
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
    container.innerHTML = emptyState("⚖️", "Need at least 2 datasets",
      e.message.includes("Need at least 2")
        ? "Upload or load 2 or more datasets, then come back here."
        : e.message);
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

// Wire up conclusion tab load on switch
const _origSwitchTab = switchTab;
