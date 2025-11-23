import pandas as pd
import os
import ast

raw_path = r"D:\Financeinsight\data\raw"
csv_files = [f for f in os.listdir(raw_path) if f.endswith(".csv")]

df_list = []
for file in csv_files:
    file_path = os.path.join(raw_path, file)
    print("Reading:", file)
    df = pd.read_csv(file_path, encoding="latin1", engine="python", on_bad_lines="skip")
    df_list.append(df)

data = pd.concat(df_list, ignore_index=True)
print("Total Rows:", len(data))
print("Columns:", data.columns)

data.to_csv(r"D:\Financeinsight\data\clean\merged.csv", index=False)
print("Merged successfully!")
print(data.columns)
  # to safely convert string lists back to lists

def tokens_to_text(token_string):
    try:
        tokens = ast.literal_eval(token_string)  # convert "['a','b']" → ['a','b']
        return " ".join(tokens)
    except:
        return None

data["text"] = data["tokens"].apply(tokens_to_text)
print(data["text"].head())
