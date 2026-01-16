import re
from datetime import datetime

def analyze_text(text):
    text_lower = text.lower()

    # Company (simple heuristic)
    company = extract_company(text)

    # Financial year
    fy_match = re.search(r"(20\d{2}[-–]20\d{2})", text)
    financial_year = fy_match.group(1) if fy_match else "Not detected"

    # Revenue & Profit
    revenue = extract_amount(text_lower, ["revenue", "turnover"])
    profit = extract_amount(text_lower, ["profit", "net profit"])

    # Sentiment
    sentiment = "Positive" if revenue or profit else "Neutral"

    return {
        "document_metadata": {
            "document_id": "AUTO-GENERATED",
            "company": company,
            "financial_year": financial_year,
            "processed_date": datetime.now().strftime("%Y-%m-%d")
        },
        "dashboard_summary": {
            "key_highlight": "Financial data extracted from uploaded report",
            "overall_sentiment": sentiment
        },
        "financial_metrics": {
            "revenue": revenue,
            "profit": profit
        },
        "key_events": [],
        "regional_insights": []
    }

def extract_company(text):
    lines = text.splitlines()
    for line in lines[:20]:
        if "limited" in line.lower() or "ltd" in line.lower():
            return line.strip()
    return "Company not detected"

def extract_amount(text, keywords):
    results = []
    for kw in keywords:
        pattern = rf"{kw}.*?(\₹|\$)?\s?\d+[,.]?\d*\s?(cr|crore|million|billion)?"
        matches = re.findall(pattern, text)
        for m in matches:
            results.append({
                "metric": kw.upper(),
                "amount": "".join(m)
            })
    return results
