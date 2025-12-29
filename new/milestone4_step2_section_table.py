import json
import ast
from pathlib import Path

# ==================================================
# PATHS
# ==================================================
BASE_DIR = Path(r"D:\Financeinsight\new")

INPUT_FILE = BASE_DIR / "outputs" / "sectioned_doc_fixed.json"
OUTPUT_FILE = BASE_DIR / "outputs" / "milestone4_tables_fixed.json"

# ==================================================
# LOAD SECTIONED DOCUMENT
# ==================================================
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    sectioned_data = json.load(f)

# ==================================================
# TABLE CONTAINERS
# ==================================================
revenue_table = []
profit_table = []
sales_table = []
events_table = []
region_table = []

# ==================================================
# PROCESS SECTIONS
# ==================================================
for section, content in sectioned_data.items():

    # -------- TEXT ENTRIES --------
    for text_item in content["text"]:
        try:
            data = ast.literal_eval(text_item)  # 🔑 FIX
        except:
            continue

        sentence = data.get("sentence")
        entities = data.get("entities", {})

        metrics = [m.lower() for m in entities.get("METRIC", [])]
        money = entities.get("MONEY", [])
        dates = entities.get("DATE", [])
        events = entities.get("EVENT", [])
        locs = entities.get("LOC", [])

        # ---------- REVENUE ----------
        if "revenue" in metrics or "revenues" in metrics:
            revenue_table.append({
                "section": section,
                "metric": "revenue",
                "value": money[0] if money else None,
                "period": " ".join(dates) if dates else None,
                "sentence": sentence
            })

        # ---------- PROFIT ----------
        if "profit" in metrics or "profits" in metrics:
            profit_table.append({
                "section": section,
                "metric": "profit",
                "value": money[0] if money else None,
                "period": " ".join(dates) if dates else None,
                "sentence": sentence
            })

        # ---------- SALES ----------
        if "sales" in metrics:
            sales_table.append({
                "section": section,
                "metric": "sales",
                "value": money[0] if money else None,
                "region": locs[0] if locs else None,
                "sentence": sentence
            })

        # ---------- EVENTS ----------
        for ev in events:
            events_table.append({
                "section": section,
                "event": ev,
                "period": " ".join(dates) if dates else None,
                "sentence": sentence
            })

        # ---------- REGION ----------
        for loc in locs:
            region_table.append({
                "section": section,
                "location": loc,
                "related_metric": metrics[0] if metrics else None,
                "sentence": sentence
            })

    # -------- TABLE ENTRIES --------
    for table in content["tables"]:
        for row in table:
            try:
                data = ast.literal_eval(row)
            except:
                continue

            sentence = data.get("sentence")
            entities = data.get("entities", {})

            metrics = [m.lower() for m in entities.get("METRIC", [])]
            money = entities.get("MONEY", [])
            dates = entities.get("DATE", [])

            if "revenue" in metrics:
                revenue_table.append({
                    "section": section,
                    "metric": "revenue",
                    "value": money[0] if money else None,
                    "period": " ".join(dates) if dates else None,
                    "sentence": sentence
                })

# ==================================================
# FINAL TABLE OUTPUT
# ==================================================
final_tables = {
    "revenue_table": revenue_table,
    "profit_table": profit_table,
    "sales_table": sales_table,
    "events_table": events_table,
    "region_table": region_table
}

# ==================================================
# SAVE OUTPUT
# ==================================================
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(final_tables, f, indent=4)

print("Tables created correctly for full document.")
print(f"Saved at: {OUTPUT_FILE}")
