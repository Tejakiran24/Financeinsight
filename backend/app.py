from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
import logging
from ner_model import extract_entities
from utils import clean_text

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="FinanceInsight API")

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage
STORED_ANALYSES = []

class AnalyzeRequest(BaseModel):
    text: str

@app.get("/health")
def health_check():
    """Check API status."""
    return {"status": "healthy"}

@app.post("/analyze")
async def analyze(
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None)
):
    """Accept text or uploaded file and return extracted entities."""
    content = ""
    if file:
        content_bytes = await file.read()
        content = content_bytes.decode('utf-8', errors='ignore')
    elif text:
        content = text
        
    if not content.strip():
        raise HTTPException(status_code=400, detail="No content provided for analysis.")

    cleaned_content = clean_text(content)
    
    try:
        entities = extract_entities(cleaned_content)
    except Exception as e:
        import traceback
        print("ERROR IN EXTRACT ENTITIES:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal inference error.")
        
    result = {
        "id": len(STORED_ANALYSES) + 1,
        "content_length": len(content),
        "snippet": content[:150] + "...",
        "entities": entities
    }
    
    STORED_ANALYSES.append(result)
    return result

@app.get("/entities")
def get_entities():
    """Return stored extracted data."""
    return STORED_ANALYSES
import os
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)