# IMDB Sentiment Classification with Transformer

A transformer-based deep learning model for sentiment classification on IMDB movie reviews. This project implements a custom transformer architecture from scratch using PyTorch for binary sentiment classification (positive/negative).

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Usage](#usage)
- [API Deployment](#api-deployment)
- [Frontend Interface](#frontend-interface)
- [Results](#results)

## Overview

This project implements a transformer-based sentiment classifier that predicts whether a movie review is positive or negative. The model is built from scratch, including:

- Multi-Head Self-Attention mechanism
- Positional Encoding
- Transformer Encoder Blocks
- Custom tokenization and vocabulary building

## Project Structure

```
Imdb_Movie_Review/
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git ignore rules
├── .gitattributes             # Git LFS configuration
├── Starter Notebook.ipynb     # Complete notebook with all steps
├── best_model.pth             # Trained model checkpoint (Git LFS)
├── api.py                     # FastAPI server for deployment
│
├── src/                       # Source code
│   ├── __init__.py            # Package initialization
│   ├── model.py               # Transformer model implementation
│   ├── data_preprocessing.py  # Data preprocessing utilities
│   ├── train.py               # Training script
│   └── predict.py             # Inference script
│
└── templates/                 # Frontend templates
    └── index.html             # Web interface
```

## Installation

### Prerequisites

- Python 3.8 or higher
- Git LFS (for downloading the model file)

### Setup

1. **Install Git LFS (required for model download):**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install git-lfs
   
   # macOS
   brew install git-lfs
   
   # Windows
   # Download from https://git-lfs.github.com/
   
   # Initialize Git LFS
   git lfs install
   ```

2. **Clone the repository:**
   ```bash
   git clone https://github.com/V1629/Imdb_Movie_Review.git
   cd Imdb_Movie_Review
   ```

3. **Create a virtual environment (recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Dataset

The project uses the IMDB Movie Reviews dataset containing 50,000 reviews labeled as positive or negative.

- **Download:** [IMDB Dataset](https://drive.google.com/file/d/1aU7Vv7jgodZ0YFOLY7kmSjrPcDDwtRfU/view?usp=sharing)
- **Structure:**
  - `review`: Text of the movie review
  - `sentiment`: Label (positive/negative)

Place the downloaded `IMDB Dataset.csv` file in the project root directory for training.

> **Note:** The dataset is not included in the repository due to size constraints.

## Model Architecture

The model uses a custom Transformer encoder architecture:

```
Input Token Indices
        │
        ▼
┌─────────────────┐
│   Embedding     │  (vocab_size → d_model=256)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Positional    │  (Sinusoidal encoding)
│   Encoding      │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│   Transformer Encoder Block (x4)   │
│   ├── Multi-Head Attention (8 heads)│
│   ├── Add & LayerNorm               │
│   ├── Feed-Forward Network          │
│   └── Add & LayerNorm               │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│  Global Average │  (Masked pooling)
│    Pooling      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Classification │  (d_model → d_model//2 → 2)
│      Head       │
└────────┬────────┘
         │
         ▼
Output Logits (Positive/Negative)
```

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| d_model | 256 |
| num_heads | 8 |
| num_layers | 4 |
| d_ff | 512 |
| max_length | 256 |
| dropout | 0.1 |
| vocab_size | ~25,000 |

## Usage

### Command Line Prediction

**Single prediction:**
```bash
python3 src/predict.py --text "This movie was absolutely fantastic! Great acting and story."
```

**Interactive mode:**
```bash
python3 src/predict.py --interactive
```

**Example output:**
```
============================================================
SENTIMENT PREDICTION RESULT
============================================================

Review: This movie was absolutely fantastic!

Predicted Sentiment: POSITIVE
Confidence: 87.50%

   Positive Probability: 87.50%
   Negative Probability: 12.50%
============================================================
```

### Python API

```python
from src.predict import SentimentPredictor

# Initialize predictor
predictor = SentimentPredictor("best_model.pth")

# Predict sentiment
result = predictor.predict("This movie was fantastic!")
print(f"Sentiment: {result['sentiment']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### Training (Optional)

If you want to retrain the model:

```bash
python3 src/train.py --data_path "IMDB Dataset.csv" --epochs 10 --batch_size 32
```

**Training options:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_path` | `IMDB Dataset.csv` | Path to the dataset |
| `--epochs` | 10 | Number of training epochs |
| `--batch_size` | 32 | Batch size for training |
| `--learning_rate` | 1e-4 | Learning rate |
| `--max_length` | 256 | Maximum sequence length |
| `--save_path` | `saved_models/` | Path to save checkpoints |

## API Deployment

The project includes a FastAPI server for model deployment.

### Start the API Server

```bash
python3 api.py
```

The server will start at `http://localhost:8000`

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface |
| `/predict` | POST | Predict sentiment |
| `/health` | GET | Health check |
| `/docs` | GET | API documentation (Swagger UI) |

### API Usage Example

**Using curl:**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "This movie was amazing!"}'
```

**Using Python requests:**
```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={"text": "This movie was amazing!"}
)
print(response.json())
```

**Response format:**
```json
{
    "text": "This movie was amazing!",
    "sentiment": "positive",
    "confidence": 0.95,
    "positive_probability": 0.95,
    "negative_probability": 0.05
}
```

## Frontend Interface

Access the web interface at `http://localhost:8000` after starting the API server.

### Features

- Clean, professional UI design
- Real-time sentiment prediction
- Confidence visualization with progress bars
- Word and character count metrics
- Example reviews for quick testing
- Responsive design for mobile and desktop
- Keyboard shortcut (Ctrl+Enter to analyze)

## Results

### Model Performance

| Metric | Value |
|--------|-------|
| Test Accuracy | ~87% |
| Test F1 Score | ~0.87 |
| Model Size | 99 MB |

### Sample Predictions

| Review | Predicted | Confidence |
|--------|-----------|------------|
| "Fantastic movie! Great acting..." | Positive | 95% |
| "Terrible waste of time..." | Negative | 93% |
| "Average film, nothing special..." | Negative | 62% |

## Technical Details

### Text Preprocessing

1. Convert to lowercase
2. Remove HTML tags
3. Remove URLs
4. Remove special characters (keep only letters and spaces)
5. Remove extra whitespace
6. Tokenize and convert to vocabulary indices
7. Pad/truncate to fixed length (256 tokens)

### Training Configuration

- **Optimizer:** AdamW with weight decay (0.01)
- **Scheduler:** Cosine Annealing LR
- **Loss Function:** CrossEntropyLoss
- **Gradient Clipping:** max_norm=1.0

## Git LFS Note

This repository uses Git Large File Storage (LFS) for the model file (`best_model.pth`). 

If you haven't installed Git LFS, the model file will not download correctly. Install it with:

```bash
# Install Git LFS
sudo apt-get install git-lfs  # Ubuntu/Debian
brew install git-lfs          # macOS

# Initialize
git lfs install

# Pull LFS files if already cloned
git lfs pull
```

## Requirements

Key dependencies (see `requirements.txt` for full list):

- torch >= 2.0.0
- fastapi >= 0.100.0
- uvicorn >= 0.23.0
- pandas >= 2.0.0
- scikit-learn >= 1.3.0

## License

This project is created for the RealAI Text Classification Challenge.

## Acknowledgments

- IMDB dataset for movie reviews
- PyTorch team for the deep learning framework
- "Attention is All You Need" paper for transformer architecture
