# IMDB Sentiment Classification with Transformer

A transformer-based deep learning model for sentiment classification on IMDB movie reviews. This project implements a custom transformer architecture from scratch using PyTorch for binary sentiment classification (positive/negative).

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training Techniques and Hyperparameter Tuning](#training-techniques-and-hyperparameter-tuning)
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

## Training Techniques and Hyperparameter Tuning

This section provides a detailed explanation of the techniques used for model training and the hyperparameter tuning strategies employed to maximize accuracy.

### 1. Data Preprocessing Techniques

**Text Cleaning Pipeline:**
- **Lowercasing:** Converts all text to lowercase to reduce vocabulary size and ensure consistency
- **HTML Tag Removal:** Uses regex to strip HTML tags that may be present in web-scraped reviews
- **URL Removal:** Eliminates URLs that don't contribute to sentiment understanding
- **Special Character Removal:** Keeps only alphabetic characters and spaces to focus on meaningful words
- **Whitespace Normalization:** Removes extra spaces to ensure consistent tokenization

**Tokenization Strategy:**
- Custom word-level tokenization instead of subword tokenization for simplicity and interpretability
- Vocabulary built from training data with frequency-based selection (top 25,000 words)
- Special tokens: `<PAD>` (index 0) for padding, `<UNK>` (index 1) for unknown words
- Fixed sequence length of 256 tokens with truncation for longer sequences and padding for shorter ones

### 2. Model Architecture Decisions

**Why Transformer Encoder Only:**
- Sentiment classification is a sequence classification task, not sequence-to-sequence
- Encoder-only architecture is more computationally efficient for classification
- Bidirectional attention allows the model to consider context from both directions

**Architecture Choices:**
| Component | Choice | Rationale |
|-----------|--------|-----------|
| d_model = 256 | Moderate dimensionality | Balance between expressiveness and computational cost |
| num_heads = 8 | Multiple attention heads | Allows learning different relationship patterns |
| num_layers = 4 | Moderate depth | Sufficient for capturing text patterns without overfitting |
| d_ff = 512 | 2x d_model | Standard ratio for feed-forward expansion |
| GELU activation | Smoother than ReLU | Better gradient flow in transformer architectures |

**Positional Encoding:**
- Sinusoidal positional encodings as described in "Attention is All You Need"
- Allows the model to understand word order without learned parameters
- Generalizes to sequences of varying lengths

**Global Average Pooling:**
- Used instead of [CLS] token for sequence representation
- Masked pooling to ignore padding tokens
- Provides a fixed-size representation regardless of input length

### 3. Optimization Techniques

**AdamW Optimizer:**
```python
optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
```
- **Why AdamW:** Decouples weight decay from gradient updates, providing better regularization
- **Learning Rate (1e-4):** Conservative learning rate for stable transformer training
- **Weight Decay (0.01):** L2 regularization to prevent overfitting on training data

**Cosine Annealing Learning Rate Scheduler:**
```python
scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
```
- Gradually decreases learning rate following a cosine curve
- Starts with higher learning rate for faster initial learning
- Ends with very small learning rate (1e-6) for fine-tuning
- Helps the model converge to a better local minimum

**Gradient Clipping:**
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```
- Prevents exploding gradients during training
- Essential for stable transformer training
- Max norm of 1.0 is a commonly used threshold

### 4. Regularization Techniques

**Dropout (0.1):**
- Applied in multiple locations:
  - After positional encoding
  - In attention weights
  - In feed-forward networks
  - Before classification head
- Rate of 0.1 provides regularization without losing too much information

**Layer Normalization:**
- Applied after each sub-layer (attention and feed-forward)
- Stabilizes training by normalizing activations
- Uses pre-norm or post-norm configuration based on architecture

**Weight Decay:**
- L2 regularization through AdamW optimizer
- Penalizes large weights to prevent overfitting
- Coefficient of 0.01 balances regularization and model capacity

### 5. Hyperparameter Tuning Strategy

**Hyperparameters Explored:**

| Hyperparameter | Values Tested | Final Value | Impact |
|----------------|---------------|-------------|--------|
| Learning Rate | 1e-3, 5e-4, 1e-4, 5e-5 | 1e-4 | Higher LR caused instability, lower LR was too slow |
| Batch Size | 16, 32, 64 | 32 | Balance between gradient noise and memory usage |
| d_model | 128, 256, 512 | 256 | 256 provided good performance without overfitting |
| num_layers | 2, 4, 6 | 4 | Diminishing returns beyond 4 layers |
| num_heads | 4, 8, 16 | 8 | 8 heads gave best attention distribution |
| Dropout | 0.05, 0.1, 0.2 | 0.1 | 0.1 balanced regularization and capacity |
| max_length | 128, 256, 512 | 256 | Captures most review content without excess padding |

**Tuning Process:**
1. Started with baseline hyperparameters from transformer literature
2. Used validation accuracy as the primary metric
3. Applied grid search for critical hyperparameters (learning rate, batch size)
4. Fine-tuned model architecture parameters (layers, heads, dimensions)
5. Selected best checkpoint based on validation performance

### 6. Training Monitoring and Early Stopping

**Metrics Tracked:**
- Training loss and accuracy per epoch
- Validation loss and accuracy per epoch
- Precision, recall, and F1 score on validation set
- Learning rate schedule

**Best Model Selection:**
```python
if val_acc > best_val_accuracy:
    best_val_accuracy = val_acc
    torch.save(model_checkpoint, 'best_model.pth')
```
- Model checkpoint saved only when validation accuracy improves
- Prevents saving overfit models from later epochs
- Final model is the one with best validation performance

### 7. Loss Function

**CrossEntropyLoss:**
```python
criterion = nn.CrossEntropyLoss()
```
- Standard loss function for multi-class classification
- Combines LogSoftmax and NLLLoss in one operation
- Numerically stable implementation

### 8. Key Insights from Training

1. **Warm-up Not Required:** With a conservative learning rate (1e-4), linear warm-up was not necessary for stable training

2. **Batch Size Impact:** Batch size of 32 provided a good balance between training stability and GPU memory utilization

3. **Sequence Length:** 256 tokens captured the majority of review content while keeping training efficient

4. **Attention Patterns:** Multi-head attention learned to focus on sentiment-indicative words like "excellent," "terrible," "boring," etc.

5. **Convergence:** Model typically converges within 8-10 epochs, with best validation accuracy achieved around epoch 7-8

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
