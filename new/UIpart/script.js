/* =====================================================
   FinanceInsight - Frontend Controller
   Old Method | Auto-load | Evaluation-ready
===================================================== */

/* -----------------------------
   BACKEND CONFIG
------------------------------ */
const API_BASE = "https://financeinsight-backend-4yub.onrender.com";
const DATA_SOURCE = `${API_BASE}/api/extract`;

/* -----------------------------
   AUTO LOAD ON PAGE OPEN
------------------------------ */
document.addEventListener("DOMContentLoaded", () => {
  loadData();
});

/* -----------------------------
   FETCH DATA FROM BACKEND
------------------------------ */
function loadData() {
  fetch(DATA_SOURCE)
    .then(response => {
      if (!response.ok) {
        throw new Error("Failed to fetch data");
      }
      return response.json();
    })
    .then(data => {
      renderDashboard(data);
    })
    .catch(error => {
      console.error(error);
      document.body.innerHTML =
        "<h2 style='color:red;text-align:center'>Error loading dashboard</h2>";
    });
}

/* -----------------------------
   MAIN RENDER FUNCTION
------------------------------ */
function renderDashboard(data) {
  renderSummary(data);
  renderMetrics(data.metrics);
  renderEvents(data.events);
  renderRegions(data.regional_insights);
  renderTables(data.tables);
}

/* -----------------------------
   SUMMARY
------------------------------ */
function renderSummary(data) {
  const summary = document.getElementById("summary");
  if (!summary) return;

  summary.innerHTML = `
    <h2>📄 Document Summary</h2>
    <p><b>Document ID:</b> ${safe(data.document_id)}</p>
    <p><b>Primary Company:</b> ${safe(data.company)}</p>
  `;
}

/* -----------------------------
   METRICS
------------------------------ */
function renderMetrics(metrics) {
  const container = document.getElementById("metricsContainer");
  if (!container || !metrics) return;

  container.innerHTML = "";

  Object.keys(metrics).forEach(metricType => {
    const rows = metrics[metricType];
    if (!rows || rows.length === 0) return;

    let html = `
      <h3>${metricType.toUpperCase()}</h3>
      <table>
        <tr>
          <th>Company</th>
          <th>Value</th>
          <th>Trend</th>
          <th>Period</th>
          <th>Sentence</th>
        </tr>
    `;

    rows.forEach(item => {
      html += `
        <tr>
          <td>${safe(item.company)}</td>
          <td>${safe(item.value)}</td>
          <td class="${trendClass(item.trend)}">${safe(item.trend)}</td>
          <td>${safe(item.period)}</td>
          <td>${safe(item.sentence)}</td>
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
    li.innerHTML = `<b>${safe(e.event).toUpperCase()}</b> – ${safe(e.sentence)}`;
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
    li.innerHTML = `<b>${safe(r.location)}</b> – ${safe(r.sentence)}`;
    list.appendChild(li);
  });
}

/* -----------------------------
   TABLE EXTRACTION
------------------------------ */
function renderTables(tables) {
  const container = document.getElementById("tablesContainer");
  if (!container || !tables) return;

  container.innerHTML = "";

  Object.keys(tables).forEach(tableName => {
    const rows = tables[tableName];
    if (!rows || rows.length === 0) return;

    let html = `
      <h3>${formatTitle(tableName)}</h3>
      <table>
        <tr>
          ${Object.keys(rows[0]).map(col => `<th>${col}</th>`).join("")}
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
   UTILITIES
------------------------------ */
function safe(value) {
  return value === null || value === undefined || value === ""
    ? "-"
    : value;
}

function trendClass(trend) {
  if (!trend) return "";
  if (trend.toLowerCase() === "positive") return "positive";
  if (trend.toLowerCase() === "negative") return "negative";
  return "";
}

function formatTitle(text) {
  return text.replace(/_/g, " ").toUpperCase();
}
