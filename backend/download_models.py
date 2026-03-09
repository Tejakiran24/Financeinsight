import spacy
import subprocess
import sys
from transformers import pipeline

def download():
    print("Downloading spaCy model...")
    try:
        spacy.load("en_core_web_sm")
    except:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    
    print("Downloading BERT model...")
    pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
    print("Pre-download complete!")

if __name__ == "__main__":
    download()
