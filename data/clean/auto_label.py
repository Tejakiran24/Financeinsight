import json
import re

# ==============================
# FILE PATHS
# ==============================
INPUT_FILE = r"D:\Financeinsight\data\clean\labelstudio_cleaned.json"
OUTPUT_FILE = r"D:\Financeinsight\data\clean\labelstudio_autolabeled.json"

# ==============================
# REGEX PATTERNS (FINANCIAL)
# ==============================
money_re = re.compile(
    r"(eur|usd|₹|\$)\s?\d+(\.\d+)?\s?(mn|m|million|crore)?",
    re.IGNORECASE
)

percent_re = re.compile(
    r"\d+(\.\d+)?\s?%",
    re.IGNORECASE
)

date_re = re.compile(
    r"\b(19|20)\d{2}(\s?-\s?(19|20)\d{2})?\b"
)

# simple ORG list (you can extend)
org_list = [
    "basware", "aspocomp", "componenta",
    "teliasonera", "eesti telekom",
    "technopolis", "gran"
]

# ==============================
# LOAD DATA
# ==============================
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# ==============================
# AUTO LABELING
# ==============================
for item in data:
    text = item.get("data", {}).get("text", "")
    results = []

    def add_label(start, end, value, label):
        results.append({
            "from_name": "label",
            "to_name": "text",
            "type": "labels",
            "value": {
                "start": start,
                "end": end,
                "text": value,
                "labels": [label]
            }
        })

    # MONEY
    for m in money_re.finditer(text):
        add_label(m.start(), m.end(), m.group(), "MONEY")

    # PERCENT
    for p in percent_re.finditer(text):
        add_label(p.start(), p.end(), p.group(), "PERCENT")

    # DATE
    for d in date_re.finditer(text):
        add_label(d.start(), d.end(), d.group(), "DATE")

    # ORG
    for org in org_list:
        for match in re.finditer(r"\b" + re.escape(org) + r"\b", text, re.IGNORECASE):
            add_label(match.start(), match.end(), match.group(), "ORG")

    # Replace predictions (remove UNKNOWN)
    item["predictions"] = [{
        "model_version": "auto",
        "result": results
    }]

    # Clear manual annotations
    item["annotations"] = []

# ==============================
# SAVE OUTPUT
# ==============================
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("✅ AUTO-LABELING COMPLETED")
print("📁 Output file:", OUTPUT_FILE)
