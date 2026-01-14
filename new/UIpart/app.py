from flask import Flask, jsonify
from flask_cors import CORS
import json
import os

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(BASE_DIR, "output.json")

@app.route("/api/extract", methods=["GET"])
def extract_data():
    if not os.path.exists(OUTPUT_FILE):
        return jsonify({"error": "output.json not found"}), 404

    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    return jsonify(data)

@app.route("/")
def health_check():
    return "FinanceInsight backend is running"

if __name__ == "__main__":
    app.run(debug=True)
