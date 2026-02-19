# IMDB Sentiment Classification with Transformer

A transformer-based deep learning model for sentiment classification on IMDB movie reviews. This project implements a custom transformer architecture from scratch using PyTorch for binary sentiment classification (positive/negative).

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Inference](#inference)
- [API Deployment](#api-deployment)
- [Frontend Interface](#frontend-interface)
- [Results](#results)
- [License](#license)

## 🎯 Overview

This project implements a transformer-based sentiment classifier that predicts whether a movie review is positive or negative. The model is built from scratch, including:

- Multi-Head Self-Attention mechanism
- Positional Encoding
- Transformer Encoder Blocks
- Custom tokenization and vocabulary building

## 📁 Project Structure

```
RealAi/
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── IMDB Dataset.csv           # Dataset file
├── Starter Notebook.ipynb     # Complete notebook with all steps
├── best_model.pth             # Saved model checkpoint
│
├── src/                       # Source code
│   ├── __init__.py
│   ├── model.py               # Transformer model implementation
│   ├── data_preprocessing.py  # Data preprocessing utilities
│   ├── train.py               # Training script
│   └── predict.py             # Inference script
│
├── api.py                     # FastAPI server for deployment
│
├── templates/                 # Frontend templates
│   └── index.html             # Web interface
│
└── saved_models/              # Saved model files (generated after training)
    ├── sentiment_model.pth
    ├── vocab.json
    └── model_config.json
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA (optional, for GPU acceleration)

### Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd RealAi
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Dataset

The project uses the IMDB Movie Reviews dataset containing 50,000 reviews labeled as positive or negative.

- **Download:** [IMDB Dataset](https://drive.google.com/file/d/1aU7Vv7jgodZ0YFOLY7kmSjrPcDDwtRfU/view?usp=sharing)
- **Structure:**
  - `review`: Text of the movie review
  - `sentiment`: Label (positive/negative)

Place the downloaded `IMDB Dataset.csv` file in the project root directory.

## 🏗️ Model Architecture

The model uses a custom Transformer encoder architecture:

```
Input Token Indices
        ↓
┌─────────────────┐
│   Embedding     │  (vocab_size → d_model=256)
└────────┬────────┘
         ↓
┌─────────────────┐
│   Positional    │  (Sinusoidal encoding)
│   Encoding      │
└────────┬────────┘
         ↓
┌─────────────────────────────────────┐
│   Transformer Encoder Block (x4)   │
│   ├── Multi-Head Attention (8 heads)│
│   ├── Add & LayerNorm               │
│   ├── Feed-Forward Network          │
│   └── Add & LayerNorm               │
└─────────────────────────────────────┘
         ↓
┌─────────────────┐
│  Global Average │  (Masked pooling)
│    Pooling      │
└────────┬────────┘
         ↓
┌─────────────────┐
│  Classification │  (d_model → d_model//2 → 2)
│      Head       │
└────────┬────────┘
         ↓
Output Logits (Positive/Negative)
```

### Key Components:

- **Multi-Head Self-Attention:** 8 attention heads for capturing different aspects of relationships
- **Positional Encoding:** Sinusoidal encodings for sequence position information
- **Feed-Forward Network:** Position-wise MLP with GELU activation
- **Layer Normalization:** For stable training with residual connections

### Hyperparameters:

| Parameter | Value |
|-----------|-------|
| d_model | 256 |
| num_heads | 8 |
| num_layers | 4 |
| d_ff | 512 |
| max_length | 256 |
| dropout | 0.1 |
| vocab_size | ~25,000 |

## 🎓 Training

### Using the Training Script

```bash
python src/train.py --data_path "IMDB Dataset.csv" --epochs 10 --batch_size 32 --learning_rate 1e-4
```

### Training Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_path` | `IMDB Dataset.csv` | Path to the dataset |
| `--epochs` | 10 | Number of training epochs |
| `--batch_size` | 32 | Batch size for training |
| `--learning_rate` | 1e-4 | Learning rate |
| `--max_length` | 256 | Maximum sequence length |
| `--d_model` | 256 | Model dimension |
| `--num_heads` | 8 | Number of attention heads |
| `--num_layers` | 4 | Number of transformer layers |
| `--save_path` | `saved_models/` | Path to save model checkpoints |

### Using the Notebook

Alternatively, run all cells in `Starter Notebook.ipynb` for an interactive training experience with visualizations.

## 📈 Evaluation

The model is evaluated using the following metrics:

- **Accuracy:** Overall correct predictions
- **Precision:** True positive rate among positive predictions
- **Recall:** True positive rate among actual positives
- **F1 Score:** Harmonic mean of precision and recall
- **Confusion Matrix:** Visual representation of predictions vs actuals

### Run Evaluation

```bash
python src/train.py --evaluate_only --model_path saved_models/sentiment_model.pth
```

## 🔮 Inference

### Using the Prediction Script

```bash
python src/predict.py --text "This movie was absolutely fantastic! Great acting and amazing story."
```

### Interactive Mode

```bash
python src/predict.py --interactive
```

### Python API

```python
from src.predict import SentimentPredictor

# Initialize predictor
predictor = SentimentPredictor("saved_models/sentiment_model.pth")

# Predict sentiment
result = predictor.predict("This movie was fantastic!")
print(f"Sentiment: {result['sentiment']}")
print(f"Confidence: {result['confidence']:.2%}")
```

## 🌐 API Deployment

The project includes a FastAPI server for model deployment.

### Start the API Server

```bash
# Start the server
python api.py

# Or using uvicorn directly
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface |
| `/predict` | POST | Predict sentiment |
| `/health` | GET | Health check |

### API Usage Example

```bash
# Using curl
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "This movie was amazing!"}'
```

```python
# Using Python requests
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={"text": "This movie was amazing!"}
)
print(response.json())
```

### Response Format

```json
{
    "text": "This movie was amazing!",
    "sentiment": "positive",
    "confidence": 0.95,
    "positive_probability": 0.95,
    "negative_probability": 0.05
}
```

## 🖥️ Frontend Interface

Access the web interface at `http://localhost:8000` after starting the API server.

Features:
- Clean, modern UI
- Real-time sentiment prediction
- Confidence visualization
- Responsive design

## 📊 Results

### Training Results (10 epochs)

| Metric | Value |
|--------|-------|
| Training Accuracy | ~92% |
| Validation Accuracy | ~87% |
| Test F1 Score | ~0.87 |

### Sample Predictions

| Review | Predicted | Confidence |
|--------|-----------|------------|
| "Fantastic movie! Great acting..." | Positive | 95% |
| "Terrible waste of time..." | Negative | 93% |
| "Average film, nothing special..." | Negative | 62% |

## 🛠️ Technical Details

### Text Preprocessing

1. Convert to lowercase
2. Remove HTML tags
3. Remove URLs
4. Remove special characters (keep only letters and spaces)
5. Remove extra whitespace
6. Tokenize and convert to indices
7. Pad/truncate to fixed length (256 tokens)

### Training Configuration

- **Optimizer:** AdamW with weight decay (0.01)
- **Scheduler:** Cosine Annealing LR
- **Loss Function:** CrossEntropyLoss
- **Gradient Clipping:** max_norm=1.0
- **Early Stopping:** Based on validation accuracy

## 📄 License

This project is created for the RealAI Text Classification Challenge.

## 🙏 Acknowledgments

- IMDB dataset for movie reviews
- PyTorch team for the deep learning framework
- "Attention is All You Need" paper for transformer architecture

---

**Note:** Make sure to place the `IMDB Dataset.csv` file in the project root before training.
