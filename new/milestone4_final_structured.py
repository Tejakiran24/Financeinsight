import json
from pathlib import Path

# ==================================================
# PATHS
# ==================================================
BASE_DIR = Path(r"D:\Financeinsight\new")

INPUT_FILE = BASE_DIR / "outputs"/"milestone4_full_tables.json"
OUTPUT_FILE = BASE_DIR / "outputs"/"final_structured_document.json"

# ==================================================
# LOAD TABLE DATA
# ==================================================
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    tables = json.load(f)

# ==================================================
# BUILD FINAL STRUCTURE
# ==================================================
final_document = {
    "document_id": "financial_doc_001",
    "company": None,  # optional (can be auto-filled later)
    "metrics": {
        "revenue": tables.get("revenue_table", []),
        "profit": tables.get("profit_table", []),
        "sales": tables.get("sales_table", [])
    },
    "events": tables.get("events_table", []),
    "regional_insights": tables.get("region_table", []),
    "tables": {
        "revenue_table": tables.get("revenue_table", []),
        "profit_table": tables.get("profit_table", []),
        "sales_table": tables.get("sales_table", [])
    }
}

# ==================================================
# SAVE FINAL OUTPUT
# ==================================================
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(final_document, f, indent=4)

print("✅ FINAL STRUCTURED DOCUMENT CREATED")
print(f"📁 Saved at: {OUTPUT_FILE}")
