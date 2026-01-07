import json
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud

# ---------------------------------
# STEP 1: Load JSON file
# ---------------------------------
FILE_PATH = r"D:\Financeinsight\data\clean\labelstudio_cleaned.json"

with open(FILE_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

print("Total Records:", len(data))


# ---------------------------------
# STEP 2: Extract text and entities
# ---------------------------------
texts = []
entity_labels = []
entity_values = []
entities_per_text = []

for record in data:
    text = record.get("data", {}).get("text", "")
    texts.append(text)

    count = 0
    annotations = record.get("annotations", [])
    predictions = record.get("predictions", [])
    source = annotations if annotations else predictions

    for ann in source:
        for res in ann.get("result", []):
            labels = res.get("value", {}).get("labels", [])
            value = res.get("value", {}).get("text", "")

            if labels:
                entity_labels.append(labels[0])
                entity_values.append(value)
                count += 1

    entities_per_text.append(count)


# ---------------------------------
# STEP 3: Create DataFrame
# ---------------------------------
df = pd.DataFrame({
    "text": texts,
    "text_length": [len(t) for t in texts],
    "entity_count": entities_per_text
})

print("\nSample Data:")
print(df.head())


# ---------------------------------
# STEP 4: Text Length Statistics
# ---------------------------------
print("\nText Length Statistics:")
print(df["text_length"].describe())


# ---------------------------------
# STEP 5: Entity Distribution
# ---------------------------------
entity_counts = Counter(entity_labels)

print("\nEntity Distribution:")
for entity, count in entity_counts.items():
    print(f"{entity}: {count}")


# ---------------------------------
# STEP 6: Bar Plot – Entity Types
# ---------------------------------
plt.figure()
plt.bar(entity_counts.keys(), entity_counts.values())
plt.xlabel("Entity Type")
plt.ylabel("Frequency")
plt.title("Distribution of Financial Entity Types")
plt.show()


# ---------------------------------
# STEP 7: Word Cloud – Financial Text
# ---------------------------------
all_text = " ".join(texts)

wordcloud = WordCloud(
    width=900,
    height=450,
    background_color="white"
).generate(all_text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud)
plt.axis("off")
plt.title("Word Cloud of Financial Text")
plt.show()


# ---------------------------------
# STEP 8: Scatter Plot – Text Length vs Entity Count
# ---------------------------------
plt.figure()
plt.scatter(df["text_length"], df["entity_count"])
plt.xlabel("Text Length")
plt.ylabel("Number of Entities")
plt.title("Text Length vs Entity Count")
plt.show()


# ---------------------------------
# STEP 9: Class Imbalance Table
# ---------------------------------
entity_df = pd.DataFrame(entity_counts.items(), columns=["Entity", "Count"])
entity_df["Percentage"] = (entity_df["Count"] / entity_df["Count"].sum()) * 100

print("\nEntity Distribution with Percentage:")
print(entity_df)


print("\nEDA COMPLETED SUCCESSFULLY ✅")
