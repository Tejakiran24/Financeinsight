from transformers import AutoTokenizer, AutoModelForTokenClassification

MODEL_NAME = "ProsusAI/finbert"
SAVE_DIR = "./finbert_ner_model/checkpoint-681"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)

tokenizer.save_pretrained(SAVE_DIR)
model.save_pretrained(SAVE_DIR)

print("FinBERT model downloaded successfully.")
