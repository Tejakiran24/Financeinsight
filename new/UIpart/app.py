from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import uuid
import re
from datetime import datetime

from utils.text_extractor import extract_text

# ===============================
# APP SETUP
# ===============================
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ===============================
# HEALTH CHECK
# ===============================
@app.route("/")
def health():
    return "FinanceInsight backend running"

# ===============================
# HELPER FUNCTIONS
# ===============================
def detect_company(text):
    for line in text.splitlines():
        if "limited" in line.lower() or "ltd" in line.lower():
            return line.strip()
    return "Not detected"

def detect_amounts(pattern, text):
    return re.findall(pattern, text, re.IGNORECASE)

# ===============================
# UPLOAD + PROCESS API
# ===============================
@app.route("/api/upload", methods=["POST"])
def upload_document():

    # ---- VALIDATE FILE ----
    if "document" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["document"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # ---- SAVE FILE ----
    filename = f"{uuid.uuid4()}_{file.filename}"
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)

    print("File saved:", path)

    # ---- EXTRACT TEXT ----
    try:
        text = extract_text(path)
    except Exception as e:
        print("Text extraction error:", e)
        text = ""

    # ---- FALLBACK IF NO TEXT ----
    if not text or not text.strip():
        text = "No readable content detected. Proceeding with metadata only."

    print("Extracted text length:", len(text))

    # ===============================
    # COMPANY
    # ===============================
    company = detect_company(text)

    # ===============================
    # FINANCIAL METRICS
    # ===============================
    revenue_matches = detect_amounts(
        r"revenue\s*(?:of|was|is)?\s*₹?\s*([\d,.]+)", text
    )
    profit_matches = detect_amounts(
        r"profit\s*(?:of|was|is)?\s*₹?\s*([\d,.]+)", text
    )

    revenue = [{"year": "Detected", "amount": f"₹{amt}"} for amt in revenue_matches] or []
    profit = [{"year": "Detected", "amount": f"₹{amt}"} for amt in profit_matches] or []

    print("Revenue detected:", revenue)
    print("Profit detected:", profit)

    # ===============================
    # FINANCIAL EVENTS
    # ===============================
    events = []
    event_keywords = [
        ("merger", "Merger"),
        ("acquisition", "Acquisition"),
        ("launch", "Product Launch"),
        ("expansion", "Business Expansion"),
        ("partnership", "Partnership")
    ]

    for keyword, label in event_keywords:
        if keyword in text.lower():
            events.append({
                "event_type": label,
                "time_period": "Detected",
                "description": f"{label} mentioned in the document",
                "impact": "Medium"
            })

    print("Events detected:", events)

    # ===============================
    # REGIONAL INSIGHTS
    # ===============================
    regions = []
    region_keywords = ["india", "asia", "europe", "usa", "america", "global"]

    for region in region_keywords:
        if region in text.lower():
            regions.append({
                "region": region.capitalize(),
                "metric": "Market / Revenue Activity",
                "details": f"Business activity detected in {region.capitalize()}",
                "impact": "Medium"
            })

    print("Regions detected:", regions)

    # ===============================
    # FINAL RESPONSE (ALWAYS RETURNS)
    # ===============================
    return jsonify({
        "document_metadata": {
            "document_id": filename,
            "company": company,
            "financial_year": "Detected from document",
            "processed_date": datetime.now().strftime("%Y-%m-%d")
        },
        "dashboard_summary": {
            "key_highlight": "Financial information extracted from uploaded document",
            "overall_sentiment": "Positive" if revenue or profit else "Neutral"
        },
        "financial_metrics": {
            "revenue": revenue,
            "profit": profit
        },
        "key_events": events,
        "regional_insights": regions
    })
