import json
from collections import defaultdict
import re

# ===============================
# FILE PATHS (WINDOWS SAFE)
# ===============================
INPUT_FILE = r"D:/Financeinsight/new/grouped_output.txt"
OUTPUT_JSON = r"D:/Financeinsight/new/user_entities.json"


entities_by_type = defaultdict(list)

# ===============================
# READ & PARSE GROUPED OUTPUT
# ===============================
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        # Find all patterns like: [entity] → LABEL
        matches = re.findall(r"\[(.*?)\]\s*→\s*([A-Z\-]+)", line)

        for entity, label in matches:
            entities_by_type[label].append(entity)


# ===============================
# SAVE JSON
# ===============================
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(entities_by_type, f, indent=4)


# ===============================
# USER-FRIENDLY DISPLAY
# ===============================
print("\n📌 USER INTEREST SUMMARY\n")

for label, values in entities_by_type.items():
    print(f"{label} ({len(values)}):")
    for v in values:
        print(f"  - {v}")

print("\n✅ User-interest entities saved to:")
print(OUTPUT_JSON)
