"""
FastAPI Server for IMDB Sentiment Classification

This module provides a REST API for the sentiment classification model
using FastAPI.

Usage:
    python api.py
    
    # Or using uvicorn directly:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
from typing import Optional
from contextlib import asynccontextmanager

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.model import TransformerClassifier
from src.data_preprocessing import clean_text, text_to_indices


# Global variables for model
model = None
vocab = None
config = None
device = None


def find_model_path():
    """Try to find the model file in common locations."""
    common_paths = [
        'saved_models/sentiment_model.pth',
        'best_model.pth',
    ]
    
    for path in common_paths:
        if os.path.exists(path):
            return path
    
    return None


def load_model():
    """Load the sentiment classification model."""
    global model, vocab, config, device
    
    model_path = find_model_path()
    if model_path is None:
        raise RuntimeError(
            "Model file not found. Please train the model first or place "
            "the model file in saved_models/sentiment_model.pth"
        )
    
    print(f"Loading model from {model_path}...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle both old format (without config) and new format (with config)
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        # Default config for models saved without config
        vocab = checkpoint['vocab']
        config = {
            'vocab_size': len(vocab),
            'd_model': 256,
            'num_heads': 8,
            'num_layers': 4,
            'd_ff': 512,
            'max_len': 256,
            'num_classes': 2,
            'dropout': 0.1
        }
    
    vocab = checkpoint['vocab']
    
    # Initialize model
    model = TransformerClassifier(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        d_ff=config['d_ff'],
        max_len=config['max_len'],
        num_classes=config['num_classes'],
        dropout=config['dropout']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print("Model loaded successfully!")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    try:
        load_model()
    except Exception as e:
        print(f"Warning: Could not load model: {e}")
        print("API will start but predictions will not work until model is available.")
    
    yield
    
    # Shutdown
    print("Shutting down...")


# Initialize FastAPI app
app = FastAPI(
    title="IMDB Sentiment Classification API",
    description="A transformer-based API for predicting movie review sentiments",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup templates
templates_dir = os.path.join(os.path.dirname(__file__), "templates")
if os.path.exists(templates_dir):
    templates = Jinja2Templates(directory=templates_dir)
else:
    templates = None


# Pydantic models for request/response
class PredictionRequest(BaseModel):
    """Request model for sentiment prediction."""
    text: str = Field(
        ..., 
        min_length=1, 
        max_length=10000,
        description="The movie review text to analyze"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "This movie was absolutely fantastic! Great acting and amazing story."
            }
        }


class PredictionResponse(BaseModel):
    """Response model for sentiment prediction."""
    text: str = Field(..., description="The original review text (truncated if long)")
    sentiment: str = Field(..., description="Predicted sentiment (positive/negative)")
    confidence: float = Field(..., description="Confidence score of the prediction")
    positive_probability: float = Field(..., description="Probability of positive sentiment")
    negative_probability: float = Field(..., description="Probability of negative sentiment")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "This movie was absolutely fantastic!...",
                "sentiment": "positive",
                "confidence": 0.95,
                "positive_probability": 0.95,
                "negative_probability": 0.05
            }
        }


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    model_loaded: bool
    device: str


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the main web interface."""
    if templates is not None:
        return templates.TemplateResponse("index.html", {"request": request})
    else:
        return HTMLResponse(content="""
        <!DOCTYPE html>
        <html>
        <head>
            <title>IMDB Sentiment Classifier</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                h1 { color: #333; }
                .info { background: #f0f0f0; padding: 15px; border-radius: 5px; }
            </style>
        </head>
        <body>
            <h1>🎬 IMDB Sentiment Classification API</h1>
            <div class="info">
                <p>Welcome to the IMDB Sentiment Classification API!</p>
                <p>Use the <code>/predict</code> endpoint to analyze movie reviews.</p>
                <p>API Documentation: <a href="/docs">/docs</a></p>
            </div>
        </body>
        </html>
        """)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check the health status of the API."""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        device=str(device) if device else "not initialized"
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_sentiment(request: PredictionRequest):
    """
    Predict the sentiment of a movie review.
    
    - **text**: The movie review text to analyze
    
    Returns the predicted sentiment with confidence scores.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure the model file exists."
        )
    
    try:
        # Clean text
        cleaned_text = clean_text(request.text)
        
        if not cleaned_text.strip():
            raise HTTPException(
                status_code=400,
                detail="Text contains no valid content after cleaning."
            )
        
        # Convert to indices
        indices = text_to_indices(cleaned_text, vocab, config['max_len'])
        
        # Convert to tensor
        input_tensor = torch.tensor([indices], dtype=torch.long).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        sentiment = "positive" if predicted_class == 1 else "negative"
        
        # Truncate text for response
        display_text = request.text[:200] + "..." if len(request.text) > 200 else request.text
        
        return PredictionResponse(
            text=display_text,
            sentiment=sentiment,
            confidence=confidence,
            positive_probability=probabilities[0][1].item(),
            negative_probability=probabilities[0][0].item()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


@app.get("/api/info")
async def api_info():
    """Get information about the API and model."""
    model_info = None
    if config is not None:
        model_info = {
            "vocab_size": config['vocab_size'],
            "d_model": config['d_model'],
            "num_heads": config['num_heads'],
            "num_layers": config['num_layers'],
            "max_length": config['max_len'],
        }
    
    return {
        "name": "IMDB Sentiment Classification API",
        "version": "1.0.0",
        "model_loaded": model is not None,
        "model_info": model_info,
        "device": str(device) if device else None,
        "endpoints": {
            "GET /": "Web interface",
            "GET /health": "Health check",
            "POST /predict": "Predict sentiment",
            "GET /docs": "API documentation",
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("IMDB Sentiment Classification API")
    print("=" * 60)
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
