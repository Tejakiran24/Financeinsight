import spacy
from transformers import pipeline
import re

nlp_spacy = None
nlp_bert = None

def load_models():
    global nlp_spacy, nlp_bert
    if nlp_spacy is None:
        try:
            nlp_spacy = spacy.load("en_core_web_sm")
        except:
            import subprocess, sys
            subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
            nlp_spacy = spacy.load("en_core_web_sm")
    if nlp_bert is None:
        # Using a smaller model to avoid long download times which cause timeouts
        nlp_bert = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")

def extract_entities(text: str):
    load_models()
    
    entities = []
    
    # Mocking financial specific extraction via regex for typical metrics
    metrics_pattern = re.compile(r'\b(P/E|EPS|Revenue|Earnings)\b[:\s\$]*([\d\.]+)', re.IGNORECASE)
    for match in metrics_pattern.finditer(text):
        entities.append({
            "text": f"{match.group(1)} {match.group(2)}",
            "type": "FINANCIAL_METRIC",
            "source": "regex"
        })

    event_pattern = re.compile(r'(IPO|Merger|Acquisition|Earnings Call|Stock Split)s?\b', re.IGNORECASE)
    for match in event_pattern.finditer(text):
        entities.append({
            "text": match.group(0),
            "type": "FINANCIAL_EVENT",
            "source": "regex"
        })

    # 1. Spacy Extraction
    try:
        doc = nlp_spacy(text)
        for ent in doc.ents:
            if ent.label_ in ["ORG", "MONEY", "DATE", "CARDINAL", "PERCENT"]:
                # Map to requirements:
                label_map = {
                    "ORG": "COMPANY",
                    "MONEY": "FINANCE",
                    "DATE": "DATE",
                    "CARDINAL": "NUMBER",
                    "PERCENT": "PERCENT"
                }
                entities.append({
                    "text": ent.text,
                    "type": label_map.get(ent.label_, ent.label_),
                    "source": "spacy"
                })
    except Exception as e:
        print(f"Spacy error: {e}")

    # 2. BERT Extraction (limit text to 512 chars to avoid model limits easily, in real app chunk it)
    try:
        bert_ents = nlp_bert(text[:512])
        for ent in bert_ents:
            entities.append({
                "text": ent["word"],
                "type": "COMPANY" if ent["entity_group"] == "ORG" else ent["entity_group"],
                "score": float(ent["score"]),
                "source": "bert"
            })
    except Exception as e:
        print(f"BERT error: {e}")

    # Deduplicate loosely based on text
    seen = set()
    unique_entities = []
    for e in entities:
        txt = e["text"].strip().lower()
        if txt not in seen and len(txt) > 1:
            seen.add(txt)
            unique_entities.append(e)

    return unique_entities
