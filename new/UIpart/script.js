/* =====================================================
   FinanceInsight - Frontend Controller
   Updated for Structured Backend JSON
===================================================== */

/* -----------------------------
   BACKEND CONFIG
------------------------------ */
const API_BASE = "https://financeinsight-backend-4yub.onrender.com";
const DATA_SOURCE = `${API_BASE}/api/extract`;

/* -----------------------------
   AUTO LOAD
------------------------------ */
document.addEventListener("DOMContentLoaded", () => {
  loadData();
});

/* -----------------------------
   FETCH DATA
------------------------------ */
function loadData() {
  fetch(DATA_SOURCE)
    .then(res => {
      if (!res.ok) throw new Error("API error");
      return res.json();
    })
    .then(data => {
      console.log("Dashboard Data:", data);
      renderDashboard(data);
    })
    .catch(err => {
      console.error(err);
      document.body.innerHTML =
        "<h2 style='color:red;text-align:center'>Error loading dashboard</h2>";
    });
}

/* -----------------------------
   MAIN RENDER
------------------------------ */
function renderDashboard(data) {
  renderSummary(data.document_metadata, data.dashboard_summary);
  renderMetrics(data.financial_metrics);
  renderEvents(data.key_events);
  renderRegions(data.regional_insights);
}

/* -----------------------------
   SUMMARY
------------------------------ */
function renderSummary(meta, summary) {
  const box = document.getElementById("summary");
  if (!box) return;

  box.innerHTML = `
    <h2>📄 Document Summary</h2>
    <p><b>Document ID:</b> ${safe(meta?.document_id)}</p>
    <p><b>Company:</b> ${safe(meta?.company)}</p>
    <p><b>Financial Year:</b> ${safe(meta?.financial_year)}</p>
    <p><b>Processed Date:</b> ${safe(meta?.processed_date)}</p>
    <p><b>Key Highlight:</b> ${safe(summary?.key_highlight)}</p>
    <p><b>Overall Sentiment:</b> ${safe(summary?.overall_sentiment)}</p>
  `;
}

/* -----------------------------
   METRICS
------------------------------ */
function renderMetrics(metrics) {
  const container = document.getElementById("metricsContainer");
  if (!container || !metrics) return;

  container.innerHTML = "";

  Object.keys(metrics).forEach(type => {
    const rows = metrics[type];
    if (!rows || rows.length === 0) return;

    let html = `
      <h3>${formatTitle(type)}</h3>
      <table>
        <tr>
          ${Object.keys(rows[0]).map(col => `<th>${formatTitle(col)}</th>`).join("")}
        </tr>
    `;

    rows.forEach(row => {
      html += `
        <tr>
          ${Object.values(row).map(val => `<td>${safe(val)}</td>`).join("")}
        </tr>
      `;
    });

    html += "</table>";
    container.innerHTML += html;
  });
}

/* -----------------------------
   EVENTS
------------------------------ */
function renderEvents(events) {
  const list = document.getElementById("eventsList");
  if (!list || !events) return;

  list.innerHTML = "";

  events.forEach(e => {
    const li = document.createElement("li");
    li.innerHTML = `
      <b>${safe(e.event_type)}</b> (${safe(e.time_period)})<br>
      ${safe(e.description)}<br>
      <i>Impact: ${safe(e.impact)}</i>
    `;
    list.appendChild(li);
  });
}

/* -----------------------------
   REGIONAL INSIGHTS
------------------------------ */
function renderRegions(regions) {
  const list = document.getElementById("regionsList");
  if (!list || !regions) return;

  list.innerHTML = "";

  regions.forEach(r => {
    const li = document.createElement("li");
    li.innerHTML = `
      <b>${safe(r.region)}</b> – ${safe(r.metric)}<br>
      ${safe(r.details)}<br>
      <i>Impact: ${safe(r.impact)}</i>
    `;
    list.appendChild(li);
  });
}

/* -----------------------------
   UTILITIES
------------------------------ */
function safe(v) {
  return v === null || v === undefined || v === "" ? "-" : v;
}

function formatTitle(text) {
  return text.replace(/_/g, " ").toUpperCase();
}
