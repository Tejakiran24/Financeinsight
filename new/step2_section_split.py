# STEP 2: Separate MD&A and Financial Statements
import json
import re
from pathlib import Path

# ---------------- PATH CONFIG ----------------

BASE_PATH = Path(r"D:\Financeinsight\new")
TEXT_FILE = BASE_PATH / "grouped_output.txt"
STEP1_FILE = BASE_PATH / "step1_grouped_entities.json"
OUTPUT_FILE = BASE_PATH / "step2_sections.json"

# ---------------- FUNCTIONS ----------------

def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def load_step1(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def is_financial_statement(sentence):
    """
    Numeric-heavy lines are considered financial statements
    """
    numbers = re.findall(r"\d+", sentence)
    return len(numbers) >= 2

def is_mda(sentence, metrics):
    """
    Sentences with metrics but no strong numeric pattern → MD&A
    """
    for metric in metrics:
        if metric.lower() in sentence.lower():
            return True
    return False

# ---------------- RUN STEP 2 ----------------

if __name__ == "__main__":

    print("Loading files...")
    text = load_text(TEXT_FILE)
    grouped_entities = load_step1(STEP1_FILE)

    metrics = grouped_entities.get("METRIC", [])

    mda_records = []
    financial_statement_lines = []

    sentences = [s.strip() for s in text.split("\n") if s.strip()]

    for sentence in sentences:
        if is_financial_statement(sentence):
            financial_statement_lines.append(sentence)
        elif is_mda(sentence, metrics):
            mda_records.append({
                "metric": sentence,
                "type": "qualitative_insight"
            })

    final_output = {
        "MD&A": mda_records,
        "Financial_Statements": financial_statement_lines
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=4)

    print("STEP 2 COMPLETED SUCCESSFULLY ✅")
