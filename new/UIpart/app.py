from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import uuid
import re
from datetime import datetime

# utils must contain __init__.py
from utils.text_extractor import extract_text

# ===============================
# APP SETUP
# ===============================
app = Flask(__name__)
CORS(app)

app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024  # 2MB

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ===============================
# HEALTH CHECK
# ===============================
@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "message": "FinanceInsight backend running"
    })

# ===============================
# HELPER FUNCTIONS
# ===============================
def detect_company(text):
    for line in text.splitlines():
        if "limited" in line.lower() or "ltd" in line.lower():
            return line.strip()
    return "Not detected"

def detect_year(text):
    match = re.search(r"(FY\d{4}[-–]\d{2}|\b20\d{2}\b)", text)
    return match.group(0) if match else "Detected"

def extract_unique_amounts(pattern, text):
    matches = re.findall(pattern, text, re.IGNORECASE)
    return list(dict.fromkeys(matches))  # preserves order + removes duplicates

# ===============================
# UPLOAD + PROCESS API
# ===============================
@app.route("/api/upload", methods=["POST"])
def upload_document():

    file_path = None

    try:
        # ---------- FILE VALIDATION ----------
        if "document" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["document"]
        if not file or file.filename.strip() == "":
            return jsonify({"error": "Empty filename"}), 400

        # ---------- SAVE FILE ----------
        filename = f"{uuid.uuid4()}_{file.filename}"
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        # ---------- EXTRACT TEXT ----------
        try:
            text = extract_text(file_path)
            if not text.strip():
                raise ValueError("Empty content")
        except Exception:
            text = "No readable content detected"

        text_lower = text.lower()

        # ===============================
        # BASIC DETECTION
        # ===============================
        company = detect_company(text)
        year = detect_year(text)

        # ===============================
        # FINANCIAL METRICS (DEDUP FIXED)
        # ===============================
        revenue_matches = extract_unique_amounts(
            r"(?:revenue|total revenue)[^₹\d]*₹?\s*([\d,]+)",
            text
        )

        profit_matches = extract_unique_amounts(
            r"(?:net profit|profit)[^₹\d]*₹?\s*([\d,]+)",
            text
        )

        revenue = (
            [{"year": year, "amount": f"₹{revenue_matches[0]}"}]
            if revenue_matches else []
        )

        profit = (
            [{"year": year, "amount": f"₹{profit_matches[0]}"}]
            if profit_matches else []
        )

        # ===============================
        # FINANCIAL EVENTS
        # ===============================
        events = []
        event_keywords = [
            ("merger", "Merger"),
            ("acquisition", "Acquisition"),
            ("launch", "Product Launch"),
            ("expansion", "Business Expansion"),
            ("partnership", "Partnership"),
            ("dividend", "Dividend Announcement"),
            ("approved", "Board Approval"),
            ("announced", "Official Announcement")
        ]

        for keyword, label in event_keywords:
            if keyword in text_lower:
                events.append({
                    "event_type": label,
                    "time_period": year,
                    "description": f"{label} detected in document",
                    "impact": "High" if label in ["Merger", "Acquisition"] else "Medium"
                })

        # ===============================
        # REGIONAL INSIGHTS
        # ===============================
        regions = []
        region_keywords = ["india", "asia", "europe", "usa", "america", "global"]

        for region in region_keywords:
            if region in text_lower:
                regions.append({
                    "region": region.capitalize(),
                    "metric": "Business Activity",
                    "details": f"Business activity detected in {region.capitalize()}",
                    "impact": "Medium"
                })

        # ===============================
        # EXTRACTED TABLES (ALWAYS PRESENT)
        # ===============================
        extracted_tables = [
            {
                "table_name": "Income Statement",
                "rows": [
                    {
                        "metric": "Revenue",
                        "amount": revenue[0]["amount"] if revenue else "Not detected",
                        "year": year
                    },
                    {
                        "metric": "Net Profit",
                        "amount": profit[0]["amount"] if profit else "Not detected",
                        "year": year
                    }
                ]
            }
        ]

        # ===============================
        # FINAL RESPONSE
        # ===============================
        response = {
            "document_metadata": {
                "document_id": filename,
                "company": company,
                "financial_year": year,
                "processed_date": datetime.now().strftime("%Y-%m-%d")
            },
            "dashboard_summary": {
                "key_highlight": "Financial insights extracted successfully",
                "overall_sentiment": "Positive" if revenue or profit else "Neutral"
            },
            "financial_metrics": {
                "revenue": revenue,
                "profit": profit
            },
            "key_events": events,
            "regional_insights": regions,
            "extracted_tables": extracted_tables
        }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": "Internal server error"}), 500

    finally:
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception:
                pass

# ===============================
# ENTRY POINT
# ===============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
