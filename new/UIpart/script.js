const API_BASE = "http://localhost:5000";

/* ===============================
   UPLOAD DOCUMENT (LANDING PAGE)
================================ */
function uploadDocument() {
  const fileInput = document.getElementById("fileInput");
  if (!fileInput || !fileInput.files.length) {
    alert("Please select a file");
    return;
  }

  const formData = new FormData();
  formData.append("document", fileInput.files[0]);

  fetch(`${API_BASE}/api/upload`, {
    method: "POST",
    body: formData
  })
    .then(res => res.json())
    .then(data => {
      // 🔴 STORE RESULT
      sessionStorage.setItem("financeData", JSON.stringify(data));

      // 🔴 REDIRECT TO DASHBOARD
      window.location.href = "dashboard.html";
    })
    .catch(err => {
      console.error(err);
      alert("Upload failed");
    });
}

/* ===============================
   LOAD DATA ON DASHBOARD PAGE
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
    if (!rows.length) return;

    let html = `<h3>${type.toUpperCase()}</h3><table border="1"><tr>`;
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

  if (!events.length) {
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

  if (!regions.length) {
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
   DOWNLOAD AS JSON
================================ */
function downloadJSON() {
  const data = sessionStorage.getItem("financeData");
  if (!data) {
    alert("No data available to download");
    return;
  }

  const blob = new Blob([data], { type: "application/json" });
  const url = URL.createObjectURL(blob);

  const a = document.createElement("a");
  a.href = url;
  a.download = "financeinsight_result.json";
  a.click();

  URL.revokeObjectURL(url);
}

/* ===============================
   DOWNLOAD AS TEXT REPORT
================================ */
function downloadText() {
  const raw = sessionStorage.getItem("financeData");
  if (!raw) {
    alert("No data available to download");
    return;
  }

  const data = JSON.parse(raw);

  let report = `
FinanceInsight – Financial Analysis Report
------------------------------------------

Company: ${data.document_metadata.company}
Financial Year: ${data.document_metadata.financial_year}
Processed Date: ${data.document_metadata.processed_date}

Key Highlight:
${data.dashboard_summary.key_highlight}

FINANCIAL METRICS:
`;

  Object.keys(data.financial_metrics).forEach(type => {
    report += `\n${type.toUpperCase()}:\n`;
    data.financial_metrics[type].forEach(r => {
      report += `- ${r.amount} (${r.year})\n`;
    });
  });

  report += `\nFINANCIAL EVENTS:\n`;
  if (data.key_events.length === 0) {
    report += "No events detected\n";
  } else {
    data.key_events.forEach(e => {
      report += `- ${e.event_type}: ${e.description}\n`;
    });
  }

  report += `\nREGIONAL INSIGHTS:\n`;
  if (data.regional_insights.length === 0) {
    report += "No regional insights detected\n";
  } else {
    data.regional_insights.forEach(r => {
      report += `- ${r.region}: ${r.details}\n`;
    });
  }

  const blob = new Blob([report], { type: "text/plain" });
  const url = URL.createObjectURL(blob);

  const a = document.createElement("a");
  a.href = url;
  a.download = "financeinsight_report.txt";
  a.click();

  URL.revokeObjectURL(url);
}
