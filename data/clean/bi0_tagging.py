import json
import re

INPUT_FILE = r"D:\Financeinsight\data\clean\labelstudio_autolabeled.json"
OUTPUT_FILE = r"D:\Financeinsight\data\clean\ner_data.bio"

def tokenize(text):
    tokens = []
    spans = []
    for match in re.finditer(r"\S+", text):
        tokens.append(match.group())
        spans.append((match.start(), match.end()))
    return tokens, spans

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

bio_lines = []

for item in data:
    text = item["data"]["text"]

    annotations = item.get("annotations", [])
    predictions = item.get("predictions", [])
    source = annotations if annotations else predictions

    entities = []
    for ann in source:
        for res in ann.get("result", []):
            start = res["value"]["start"]
            end = res["value"]["end"]
            label = res["value"]["labels"][0]
            entities.append((start, end, label))

    tokens, token_spans = tokenize(text)

    prev_entity = None

    for token, (t_start, t_end) in zip(tokens, token_spans):
        tag = "O"
        current_entity = None

        for e_start, e_end, e_label in entities:
            if t_start >= e_start and t_end <= e_end:
                current_entity = (e_start, e_end, e_label)
                if prev_entity == current_entity:
                    tag = f"I-{e_label}"
                else:
                    tag = f"B-{e_label}"
                break

        prev_entity = current_entity
        bio_lines.append(f"{token}\t{tag}")

    bio_lines.append("")

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(bio_lines))

print("✅ BIO CONVERSION FIXED & COMPLETED")
print("📁 Output file:", OUTPUT_FILE)
