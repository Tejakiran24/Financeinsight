import pandas as pd
import re
import spacy
import ast

nlp = spacy.load("en_core_web_sm")

# Load merged dataset
data = pd.read_csv(r"D:\Financeinsight\data\clean\merged.csv")

# Convert tokens (string list) into sentence
def tokens_to_text(token_string):
    try:
        tokens = ast.literal_eval(token_string)
        return " ".join(tokens)
    except:
        return None

data["text"] = data["tokens"].apply(tokens_to_text)

# Preprocess text
def preprocess(text):
    if pd.isna(text):
        return ""
    text = text.replace("\n", " ")
    text = re.sub(r'[\$,€₹]', ' CUR ', text)
    text = re.sub(r'[^A-Za-z0-9\s\.\,\%]', ' ', text)
    doc = nlp(text)
    return " ".join([token.lemma_.lower() 
                     for token in doc if not token.is_stop])

data["clean_text"] = data["text"].apply(preprocess)

data.to_csv(r"D:\Financeinsight\data\clean\final_clean.csv", index=False)
print("Preprocessing Completed Successfully!")
