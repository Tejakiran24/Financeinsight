INPUT_FILE = r"D:/Financeinsight/new/trainf2.txt"
OUTPUT_FILE = r"D:/Financeinsight/new/grouped_output.txt"



def group_sentence(lines):
    grouped = []
    current_entity = ""
    current_label = ""

    for line in lines:
        token, label = line.rsplit(" ", 1)

        if label.startswith("B-"):
            if current_entity:
                grouped.append((current_entity, current_label))
            current_entity = token
            current_label = label[2:]

        elif label.startswith("I-") and current_label == label[2:]:
            # handle decimals like 10 . 6
            if token == ".":
                current_entity += token
            elif current_entity.endswith("."):
                current_entity += token
            else:
                current_entity += " " + token

        else:  # O
            if current_entity:
                grouped.append((current_entity, current_label))
                current_entity = ""
                current_label = ""

    if current_entity:
        grouped.append((current_entity, current_label))

    return grouped


# ===== READ FILE & SPLIT SENTENCES =====
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f.readlines()]

sentences = []
current = []

for line in lines:
    if line == "":
        if current:
            sentences.append(current)
            current = []
    else:
        current.append(line)

if current:
    sentences.append(current)


# ===== PROCESS FULL DOCUMENT =====
with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
    for sentence in sentences:
        grouped_entities = group_sentence(sentence)
        if grouped_entities:
            output_line = "  ".join(
                f"[{entity}] → {label}"
                for entity, label in grouped_entities
            )
            out.write(output_line + "\n")

print("✅ Entire document grouped successfully")
print(f"📄 Output saved to: {OUTPUT_FILE}")
