# STEP 1: Read text document and group NER entities
# Milestone 3 – Correct for your folder structure

import re
import json
from collections import defaultdict
from pathlib import Path

# ---------------- PATH CONFIG ----------------

BASE_PATH = Path(r"D:\Financeinsight\new")
INPUT_FILE = BASE_PATH / "grouped_output.txt"
OUTPUT_FILE = BASE_PATH / "step1_grouped_entities.json"

# ---------------- FUNCTIONS ----------------

def load_text(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def group_entities(text):
    """
    Extract patterns like:
    [profit] → METRIC
    """
    pattern = re.compile(r"\[(.*?)\]\s*→\s*(\w+)")
    grouped = defaultdict(list)

    for value, label in pattern.findall(text):
        grouped[label].append(value.strip())

    return dict(grouped)

def save_output(data, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    print(f"Grouped output saved to: {file_path}")

# ---------------- RUN STEP 1 ----------------

if __name__ == "__main__":

    print("Loading text document...")
    raw_text = load_text(INPUT_FILE)

    print("Grouping entities...")
    grouped_entities = group_entities(raw_text)

    print("Saving grouped output...")
    save_output(grouped_entities, OUTPUT_FILE)

    print("\nSTEP 1 COMPLETED SUCCESSFULLY ✅")
