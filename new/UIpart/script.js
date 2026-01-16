/* ===============================
   BACKEND CONFIG
================================ */
const API_BASE = "http://127.0.0.1:5000";


/* Wake up backend (Render cold start fix) */
fetch(`${API_BASE}/`).catch(() => {});

/* ===============================
   LOADING UI HELPERS
================================ */
function showLoading(message) {
  let loader = document.getElementById("global-loader");

  if (!loader) {
    loader = document.createElement("div");
    loader.id = "global-loader";
    loader.innerHTML = `
      <div class="loader-box">
        <div class="spinner"></div>
        <p id="loader-text"></p>
      </div>
    `;
    document.body.appendChild(loader);
  }

  document.getElementById("loader-text").innerText = message;
  loader.style.display = "flex";
}

function hideLoading() {
  const loader = document.getElementById("global-loader");
  if (loader) loader.style.display = "none";
}

/* ===============================
   UPLOAD DOCUMENT
================================ */
function uploadDocument(retry = false) {
  const fileInput = document.getElementById("fileInput");

  if (!fileInput || !fileInput.files.length) {
    alert("Please select a financial document");
    return;
  }

  const file = fileInput.files[0];

  /* File size limit (2MB) */
  if (file.size > 2 * 1024 * 1024) {
    alert("Please upload a file smaller than 2MB for faster analysis");
    return;
  }

  const formData = new FormData();
  formData.append("document", file);

  showLoading("Uploading document...");

  fetch(`${API_BASE}/api/upload`, {
    method: "POST",
    body: formData
  })
    .then(res => {
      showLoading("Extracting financial data...");
      if (!res.ok) throw new Error("Server error");
      return res.json();
    })
    .then(data => {
      showLoading("Generating insights...");
      sessionStorage.setItem("financeData", JSON.stringify(data));
      hideLoading();
      window.location.href = "dashboard.html";
    })
    .catch(err => {
      console.error("Upload error:", err);
      hideLoading();

      if (!retry) {
        alert("Backend is waking up. Retrying once...");
        setTimeout(() => uploadDocument(true), 3000);
      } else {
        alert("Upload failed. Please try again.");
      }
    });
}

/* ===============================
   LOAD DATA ON DASHBOARD
================================ */
document.addEventListener("DOMContentLoaded", () => {
  const storedData = sessionStorage.getItem("financeData");
  if (storedData) {
    renderDashboard(JSON.parse(storedData));
  }
});

/* ===============================
   RENDER DASHBOARD
================================ */
function renderDashboard(data) {
  renderSummary(data.document_metadata, data.dashboard_summary);
  renderMetrics(data.financial_metrics);
  renderEvents(data.key_events);
  renderRegions(data.regional_insights);
}

/* ===============================
   SUMMARY
================================ */
function renderSummary(meta, summary) {
  const box = document.getElementById("summary");
  if (!box) return;

  box.innerHTML = `
    <h2>📄 Document Summary</h2>
    <p><b>Company:</b> ${meta.company}</p>
    <p><b>Financial Year:</b> ${meta.financial_year}</p>
    <p><b>Processed Date:</b> ${meta.processed_date}</p>
    <p><b>Highlight:</b> ${summary.key_highlight}</p>
    <p><b>Sentiment:</b> ${summary.overall_sentiment}</p>
  `;
}

/* ===============================
   METRICS
================================ */
function renderMetrics(metrics) {
  const container = document.getElementById("metricsContainer");
  if (!container) return;

  container.innerHTML = "";

  Object.keys(metrics).forEach(type => {
    const rows = metrics[type];
    if (!rows || rows.length === 0) return;

    let html = `<h3>${type.toUpperCase()}</h3><table><tr>`;
    Object.keys(rows[0]).forEach(col => html += `<th>${col}</th>`);
    html += "</tr>";

    rows.forEach(row => {
      html += "<tr>";
      Object.values(row).forEach(v => html += `<td>${v}</td>`);
      html += "</tr>";
    });

    html += "</table>";
    container.innerHTML += html;
  });
}

/* ===============================
   EVENTS
================================ */
function renderEvents(events) {
  const list = document.getElementById("eventsList");
  if (!list) return;

  list.innerHTML = "";

  if (!events || events.length === 0) {
    list.innerHTML = "<li>No financial events detected</li>";
    return;
  }

  events.forEach(e => {
    const li = document.createElement("li");
    li.innerHTML = `<b>${e.event_type}</b> – ${e.description}`;
    list.appendChild(li);
  });
}

/* ===============================
   REGIONS
================================ */
function renderRegions(regions) {
  const list = document.getElementById("regionsList");
  if (!list) return;

  list.innerHTML = "";

  if (!regions || regions.length === 0) {
    list.innerHTML = "<li>No regional insights detected</li>";
    return;
  }

  regions.forEach(r => {
    const li = document.createElement("li");
    li.innerHTML = `<b>${r.region}</b> – ${r.details}`;
    list.appendChild(li);
  });
}

/* ===============================
   DOWNLOAD JSON
================================ */
function downloadJSON() {
  const data = sessionStorage.getItem("financeData");
  if (!data) return alert("No data available");

  const blob = new Blob([data], { type: "application/json" });
  const url = URL.createObjectURL(blob);

  const a = document.createElement("a");
  a.href = url;
  a.download = "financeinsight_result.json";
  a.click();

  URL.revokeObjectURL(url);
}

/* ===============================
   DOWNLOAD TEXT
================================ */
function downloadText() {
  const raw = sessionStorage.getItem("financeData");
  if (!raw) return alert("No data available");

  const data = JSON.parse(raw);

  let report = `FinanceInsight – Financial Analysis Report\n\n`;
  report += `Company: ${data.document_metadata.company}\n`;
  report += `Financial Year: ${data.document_metadata.financial_year}\n`;
  report += `Processed Date: ${data.document_metadata.processed_date}\n\n`;

  report += "FINANCIAL METRICS:\n";
  Object.keys(data.financial_metrics).forEach(type => {
    data.financial_metrics[type].forEach(r => {
      report += `- ${type}: ${r.amount}\n`;
    });
  });

  const blob = new Blob([report], { type: "text/plain" });
  const url = URL.createObjectURL(blob);

  const a = document.createElement("a");
  a.href = url;
  a.download = "financeinsight_report.txt";
  a.click();

  URL.revokeObjectURL(url);
}
