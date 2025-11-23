import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import seaborn as sns
import os

# Create output directory if not exists
os.makedirs(r"D:\Financeinsight\results\eda", exist_ok=True)

# Load cleaned dataset
data = pd.read_csv(r"D:\Financeinsight\data\clean\final_clean.csv")

# =======================
# 1️⃣ WORD CLOUD
# =======================
all_text = " ".join(data["clean_text"].dropna())
wc = WordCloud(width=1200, height=600, background_color="white").generate(all_text)
plt.figure(figsize=(14, 7))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud of Financial Text")
plt.savefig(r"D:\Financeinsight\results\eda\wordcloud.png")
plt.close()

# =======================
# 2️⃣ TOP WORD FREQUENCY BAR CHART
# =======================
words = all_text.split()
counts = Counter(words).most_common(20)
freq_df = pd.DataFrame(counts, columns=["word", "count"])
plt.figure(figsize=(10, 8))
sns.barplot(x="count", y="word", data=freq_df)
plt.title("Top 20 Most Frequent Words")
plt.tight_layout()
plt.savefig(r"D:\Financeinsight\results\eda\word_frequency.png")
plt.close()

# =======================
# 3️⃣ SENTENCE LENGTH DISTRIBUTION
# =======================
data['text_length'] = data["clean_text"].apply(lambda x: len(str(x).split()))
plt.figure(figsize=(12, 6))
sns.histplot(data['text_length'], bins=40, kde=True)
plt.title("Sentence Length Distribution")
plt.xlabel("Number of Words")
plt.ylabel("Frequency")
plt.savefig(r"D:\Financeinsight\results\eda\sentence_length.png")
plt.close()

print("EDA Completed Successfully! All images saved in results/eda/")
