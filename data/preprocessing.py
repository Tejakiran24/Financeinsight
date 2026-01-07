import pandas as pd
import re
import spacy

# Load SpaCy English model (make sure installed)
nlp = spacy.load("en_core_web_sm")

# Load your dataset
df = pd.read_csv(r"D:\Financeinsight\data\clean\NER_with_sentences.csv")

# -----------------------------
# 1️⃣ REMOVE URLs + SEC LINKS
# -----------------------------
def remove_urls(text):
    if pd.isna(text):
        return ""
    return re.sub(r'http\S+|www\S+|sec\.gov\S+', '', text)


# -----------------------------
# 2️⃣ REMOVE SPECIAL SYMBOLS
# (keep only letters, numbers, spaces, ., ,)
# -----------------------------
def remove_special(text):
    text = re.sub(r"[^A-Za-z0-9\s.,]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# -----------------------------
# 3️⃣ FINAL CLEANING PIPELINE
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = remove_urls(text)
    text = remove_special(text)
    return text


df["clean_text"] = df["text_sentence"].apply(clean_text)


# -----------------------------
# 4️⃣ ADD PARTS OF SPEECH (POS TAGS)
# Example: "The company reported earnings" → 
# "DET NOUN VERB NOUN"
# -----------------------------
def get_pos_tags(sentence):
    if pd.isna(sentence):
        return ""
    doc = nlp(sentence)
    return " ".join([token.pos_ for token in doc])


df["pos_tags"] = df["clean_text"].apply(get_pos_tags)


# -----------------------------
# 5️⃣ SAVE FINAL CLEAN DATASET
# -----------------------------
df.to_csv(r"D:\Financeinsight\data\clean\NER_preprocessed_final.csv", index=False)

print("Preprocessing completed!")
print("Final dataset saved.")
