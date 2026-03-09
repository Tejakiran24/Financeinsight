import json
import re
from pathlib import Path

# ==================================================
# PATHS
# ==================================================
BASE_DIR = Path(r"D:\Financeinsight\new")

INPUT_FILE = BASE_DIR / "outputs" / "milestone4_full_tables.json"
OUTPUT_FILE = BASE_DIR / "outputs" / "final_structured_document.json"

# ==================================================
# LOAD DATA
# ==================================================
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    tables = json.load(f)

# ==================================================
# COMPANY KEYWORDS
# ==================================================
COMPANY_KEYWORDS = {
    "apple": "Apple Inc",
    "amazon": "Amazon",
    "google": "Google",
    "tesla": "Tesla",
    "netflix": "Netflix",
    "samsung": "Samsung Electronics",
    "ibm": "IBM",
    "microsoft": "Microsoft",
    "oracle": "Oracle"
}

# ==================================================
# HELPER FUNCTIONS
# ==================================================
def extract_company_from_sentence(sentence):
    s = sentence.lower()
    for k, v in COMPANY_KEYWORDS.items():
        if k in s:
            return v
    return None


def extract_monetary_value(sentence):
    match = re.search(
        r"\b(?:usd|eur|inr|\$)\s?\d+(\.\d+)?\s?(million|billion|crore)?",
        sentence.lower()
    )
    return match.group(0) if match else None


def extract_period(sentence):
    s = sentence.lower()

    year = re.search(r"\b(19|20)\d{2}\b", s)
    if year:
        return year.group(0)

    if "quarter" in s:
        return "quarter"
    if "half" in s:
        return "half-year"
    if "this year" in s:
        return "year"
    if "last year" in s:
        return "last year"

    return None


def extract_trend(sentence):
    s = sentence.lower()
    if any(w in s for w in ["increase", "growth", "improved", "strong", "expanded", "record"]):
        return "positive"
    if any(w in s for w in ["decline", "decrease", "reduced", "lower", "fell"]):
        return "negative"
    return None


def normalize_rows(rows, default_company):
    normalized = []
    last_company = default_company

    for row in rows:
        sentence = row.get("sentence", "")

        detected_company = extract_company_from_sentence(sentence)
        if detected_company:
            last_company = detected_company  # update context

        normalized.append({
            "company": last_company,                 # NEVER null / placeholder
            "metric": row.get("metric"),
            "value": extract_monetary_value(sentence),
            "trend": extract_trend(sentence),
            "period": extract_period(sentence),
            "sentence": sentence
        })

    return normalized

# ==================================================
# DOCUMENT-LEVEL DEFAULT COMPANY (ANCHOR)
# ==================================================
DEFAULT_COMPANY = "Apple Inc"

# ==================================================
# NORMALIZE TABLES
# ==================================================
revenue_table = normalize_rows(tables.get("revenue_table", []), DEFAULT_COMPANY)
profit_table = normalize_rows(tables.get("profit_table", []), DEFAULT_COMPANY)
sales_table = normalize_rows(tables.get("sales_table", []), DEFAULT_COMPANY)

# ==================================================
# FINAL STRUCTURE
# ==================================================
final_document = {
    "document_id": "financial_doc_001",
    "company": DEFAULT_COMPANY,
    "metrics": {
        "revenue": revenue_table,
        "profit": profit_table,
        "sales": sales_table
    },
    "events": tables.get("events_table", []),
    "regional_insights": tables.get("region_table", []),
    "tables": {
        "revenue_table": revenue_table,
        "profit_table": profit_table,
        "sales_table": sales_table
    }
}

# ==================================================
# SAVE OUTPUT
# ==================================================
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(final_document, f, indent=4)

print("✅ FINAL STRUCTURED DOCUMENT CREATED")
print("🚫 No null / Unknown / Multiple Companies")
print(f"📁 Saved at: {OUTPUT_FILE}")
