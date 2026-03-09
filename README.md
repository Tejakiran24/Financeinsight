# FinanceInsight

FinanceInsight is an AI-powered financial document analysis platform designed to extract structured financial data from unstructured text such as earnings reports, financial news, analyst reports, and SEC filings.

## Features
- **Document Upload**: Support for plain text and raw file processing.
- **Financial Entity Extraction**: Powered by spaCy and Transformers (BERT). Extracts companies, monetary values, and specialized financial metrics.
- **Results Dashboard**: View extracted entities across documents with responsive charting via Recharts.
- **Modern UI**: Built with React, TailwindCSS, and Framer Motion.

## Local Environment Setup

### 1. Backend Setup

The backend uses Python and FastAPI.

```bash
# Navigate to project root
cd financeinsight

# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install requirements
cd backend
pip install -r requirements.txt

# Run the backend server
uvicorn app:app --reload
```
The application will be accessible at http://127.0.0.1:8000.

### 2. Frontend Setup

The frontend uses React and Vite.

```bash
# Navigate to the frontend directory
cd frontend

# Install dependencies
npm install

# Start the dev server
npm run dev
```

The app will typically be accessible at http://localhost:5173.

## Project Structure
```
financeinsight/
│
├── backend/
│   ├── app.py
│   ├── ner_model.py
│   ├── utils.py
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── App.jsx
│   │   └── index.css
│   ├── package.json
│   └── tailwind.config.js
│
├── data/
├── models/
└── README.md
```
