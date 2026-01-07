import pandas as pd
import ast

# Load cleaned dataset with only token rows
df = pd.read_csv(r"D:\Financeinsight\data\clean\NER_only.csv")

# Convert token list string to sentence
def tokens_to_sentence(x):
    if pd.isna(x):
        return ""
    try:
        tokens = ast.literal_eval(x)  # Convert string list to real list
        if isinstance(tokens, list):
            return " ".join(tokens)
        return ""
    except:
        return ""

# Create new text column
df["text_sentence"] = df["tokens"].apply(tokens_to_sentence)

# Remove empty rows (just in case)
df = df[df["text_sentence"].str.strip() != ""]

# Save
df.to_csv(r"D:\Financeinsight\data\clean\NER_with_sentences.csv", index=False)

print("Successfully added text sentences!")
print("Final rows:", len(df))
print(df[["tokens", "text_sentence"]].head())
