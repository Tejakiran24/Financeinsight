import json
import spacy
import re

INPUT_FILE = r"D:\Financeinsight\data\clean\labelstudio_cleaned.json"
OUTPUT_FILE = r"D:\Financeinsight\data\clean\labelstudio_autolabeled_all.json"

nlp = spacy.load("en_core_web_sm")

LABEL_MAP = {
    "ORG": "ORG",
    "PERSON": "PERSON",
    "GPE": "LOC",
    "LOC": "LOC",
    "DATE": "DATE",
    "MONEY": "MONEY",
    "PERCENT": "PERCENT",
    "PRODUCT": "PRODUCT",
    "EVENT": "EVENT"
}

metric_keywords = re.compile(
    r"(profit|loss|revenue|sales|growth|capacity|production|margin|demand|income)",
    re.IGNORECASE
)

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

for item in data:
    text = item["data"]["text"]
    doc = nlp(text)

    results = []

    # --- spaCy entities ---
    for ent in doc.ents:
        if ent.label_ in LABEL_MAP:
            results.append({
                "from_name": "label",
                "to_name": "text",
                "type": "labels",
                "value": {
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "text": ent.text,
                    "labels": [LABEL_MAP[ent.label_]]
                }
            })

    # --- finance metrics (regex) ---
    for m in metric_keywords.finditer(text):
        results.append({
            "from_name": "label",
            "to_name": "text",
            "type": "labels",
            "value": {
                "start": m.start(),
                "end": m.end(),
                "text": m.group(),
                "labels": ["METRIC"]
            }
        })

    # --- GUARANTEE: label every sentence ---
    if not results:
        results.append({
            "from_name": "label",
            "to_name": "text",
            "type": "labels",
            "value": {
                "start": 0,
                "end": len(text),
                "text": text,
                "labels": ["METRIC"]
            }
        })

    item["predictions"] = [{
        "model_version": "auto-all",
        "result": results
    }]

    item["annotations"] = []

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print("✅ Auto-labeling completed — EVERY line labeled")
