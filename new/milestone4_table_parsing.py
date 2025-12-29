import ast
import json
from pathlib import Path

# ==================================================
# PATHS
# ==================================================
BASE_DIR = Path(r"D:\Financeinsight\new")
INPUT_TXT = BASE_DIR / "mile_4_output.txt"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_JSON = OUTPUT_DIR / "milestone4_full_tables.json"

OUTPUT_DIR.mkdir(exist_ok=True)

# ==================================================
# INFERENCE FUNCTIONS
# ==================================================
def infer_period(sentence):
    s = sentence.lower()
    if "q1" in s or "q2" in s or "q3" in s or "q4" in s:
        return "quarter"
    if "quarter" in s:
        return "quarter"
    if "year" in s or "annually" in s:
        return "year"
    if "half" in s:
        return "half-year"
    return None


def infer_value(money, sentence):
    if money:
        return money[0]

    s = sentence.lower()
    if any(w in s for w in ["increase", "increased", "growth", "grew", "improved", "expanded", "strong", "record"]):
        return "positive trend"
    if any(w in s for w in ["decline", "declined", "decrease", "reduced", "fell", "lower"]):
        return "negative trend"
    return None

# ==================================================
# TABLE CONTAINERS
# ==================================================
revenue_table = []
profit_table = []
sales_table = []
events_table = []
region_table = []

print("Processing TXT document and generating tables...")

# ==================================================
# PARSE TXT FILE
# ==================================================
with open(INPUT_TXT, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()

        # Skip section headers and empty lines
        if not line or line.startswith("["):
            continue

        try:
            row = ast.literal_eval(line)
        except:
            continue

        sentence = row.get("sentence", "")
        entities = row.get("entities", {})

        metrics = entities.get("METRIC", [])
        money = entities.get("MONEY", [])
        dates = entities.get("DATE", [])
        events = entities.get("EVENT", [])
        locs = entities.get("LOC", [])

        # ---------------- REVENUE TABLE ----------------
        if "revenue" in metrics or "revenues" in metrics:
            revenue_table.append({
                "metric": "revenue",
                "value": infer_value(money, sentence),
                "period": " ".join(dates) if dates else infer_period(sentence),
                "sentence": sentence
            })

        # ---------------- PROFIT TABLE ----------------
        if "profit" in metrics or "profits" in metrics:
            profit_table.append({
                "metric": "profit",
                "value": infer_value(money, sentence),
                "period": " ".join(dates) if dates else infer_period(sentence),
                "sentence": sentence
            })

        # ---------------- SALES TABLE ----------------
        if "sales" in metrics:
            sales_table.append({
                "metric": "sales",
                "value": infer_value(money, sentence),
                "region": locs[0] if locs else None,
                "sentence": sentence
            })

        # ---------------- EVENTS TABLE ----------------
        for ev in events:
            events_table.append({
                "event": ev,
                "period": " ".join(dates) if dates else infer_period(sentence),
                "sentence": sentence
            })

        # ---------------- REGION TABLE ----------------
        for loc in locs:
            region_table.append({
                "location": loc,
                "related_metric": metrics[0] if metrics else None,
                "sentence": sentence
            })

# ==================================================
# FINAL OUTPUT
# ==================================================
final_tables = {
    "revenue_table": revenue_table,
    "profit_table": profit_table,
    "sales_table": sales_table,
    "events_table": events_table,
    "region_table": region_table
}

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(final_tables, f, indent=4)

print("All tables generated successfully.")
print(f"Saved at: {OUTPUT_JSON}")
