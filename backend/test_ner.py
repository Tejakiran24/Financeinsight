import sys
import traceback

sys.path.append("d:\\Financeinsight\\backend")
from ner_model import extract_entities

if __name__ == "__main__":
    try:
        print("Testing extraction...")
        res = extract_entities("Apple Inc. is doing well.")
        print("Success:", res)
    except Exception as e:
        print("Error during extract_entities:")
        traceback.print_exc()
