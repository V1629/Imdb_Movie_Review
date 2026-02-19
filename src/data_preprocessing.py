"""
Data Preprocessing Module for IMDB Sentiment Classification

This module contains utilities for loading, cleaning, and preprocessing
the IMDB movie review dataset for sentiment classification.
"""

import re
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter
from typing import Dict, List, Tuple, Optional


def clean_text(text: str) -> str:
    """
    Clean and preprocess text data.
    
    Args:
        text: Raw text string
        
    Returns:
        Cleaned text string
    """
    # Convert to lowercase
    text = text.lower()
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    # Remove special characters and digits (keep only letters and spaces)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text


def build_vocab(texts: List[str], max_vocab_size: int = 25000) -> Dict[str, int]:
    """
    Build vocabulary from text data.
    
    Args:
        texts: List of cleaned text strings
        max_vocab_size: Maximum vocabulary size
        
    Returns:
        Vocabulary dictionary mapping words to indices
    """
    word_counts = Counter()
    for text in texts:
        word_counts.update(text.split())
    
    # Special tokens
    vocab = {'<PAD>': 0, '<UNK>': 1}
    
    # Add most common words
    for word, _ in word_counts.most_common(max_vocab_size - 2):
        vocab[word] = len(vocab)
    
    return vocab


def text_to_indices(text: str, vocab: Dict[str, int], max_length: int = 256) -> List[int]:
    """
    Convert text to indices with padding/truncation.
    
    Args:
        text: Cleaned text string
        vocab: Vocabulary dictionary
        max_length: Maximum sequence length
        
    Returns:
        List of token indices
    """
    tokens = text.split()
    indices = [vocab.get(token, vocab['<UNK>']) for token in tokens]
    
    # Truncate or pad
    if len(indices) > max_length:
        indices = indices[:max_length]
    else:
        indices = indices + [vocab['<PAD>']] * (max_length - len(indices))
    
    return indices


class IMDBDataset(Dataset):
    """
    PyTorch Dataset for IMDB movie reviews.
    
    Args:
        texts: List of token indices (list of lists)
        labels: List of sentiment labels (0 or 1)
    """
    def __init__(self, texts: List[List[int]], labels: List[int]):
        self.texts = torch.tensor(texts, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.texts[idx], self.labels[idx]


def load_and_preprocess_data(
    data_path: str,
    max_vocab_size: int = 25000,
    max_length: int = 256,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[DataLoader, DataLoader, Dict[str, int], Dict[str, int]]:
    """
    Load and preprocess the IMDB dataset.
    
    Args:
        data_path: Path to the IMDB Dataset CSV file
        max_vocab_size: Maximum vocabulary size
        max_length: Maximum sequence length
        test_size: Fraction of data for testing
        random_state: Random seed for reproducibility
        
    Returns:
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        vocab: Vocabulary dictionary
        label_map: Label mapping dictionary
    """
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Sentiment distribution:\n{df['sentiment'].value_counts()}")
    
    # Clean text
    print("\nCleaning text...")
    df['cleaned_review'] = df['review'].apply(clean_text)
    
    # Encode labels
    label_map = {'positive': 1, 'negative': 0}
    df['label'] = df['sentiment'].map(label_map)
    
    # Build vocabulary
    print("Building vocabulary...")
    vocab = build_vocab(df['cleaned_review'].tolist(), max_vocab_size)
    print(f"Vocabulary size: {len(vocab)}")
    
    # Convert to indices
    print("Converting text to indices...")
    df['indices'] = df['cleaned_review'].apply(
        lambda x: text_to_indices(x, vocab, max_length)
    )
    
    # Split data
    X = df['indices'].tolist()
    y = df['label'].tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test, vocab, label_map


def create_dataloaders(
    X_train: List[List[int]],
    X_test: List[List[int]],
    y_train: List[int],
    y_test: List[int],
    batch_size: int = 32
) -> Tuple[DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders from preprocessed data.
    
    Args:
        X_train: Training token indices
        X_test: Test token indices
        y_train: Training labels
        y_test: Test labels
        batch_size: Batch size for dataloaders
        
    Returns:
        train_loader: Training DataLoader
        test_loader: Test DataLoader
    """
    train_dataset = IMDBDataset(X_train, y_train)
    test_dataset = IMDBDataset(X_test, y_test)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of test batches: {len(test_loader)}")
    
    return train_loader, test_loader


def get_data_statistics(df: pd.DataFrame) -> Dict:
    """
    Calculate statistics about the dataset.
    
    Args:
        df: DataFrame with 'review' and 'sentiment' columns
        
    Returns:
        Dictionary containing various statistics
    """
    df_temp = df.copy()
    df_temp['review_length'] = df_temp['review'].apply(len)
    df_temp['word_count'] = df_temp['review'].apply(lambda x: len(x.split()))
    
    stats = {
        'total_samples': len(df_temp),
        'positive_samples': len(df_temp[df_temp['sentiment'] == 'positive']),
        'negative_samples': len(df_temp[df_temp['sentiment'] == 'negative']),
        'avg_review_length': df_temp['review_length'].mean(),
        'avg_word_count': df_temp['word_count'].mean(),
        'max_review_length': df_temp['review_length'].max(),
        'max_word_count': df_temp['word_count'].max(),
        'min_review_length': df_temp['review_length'].min(),
        'min_word_count': df_temp['word_count'].min(),
    }
    
    return stats


if __name__ == "__main__":
    # Test the preprocessing functions
    print("Testing data preprocessing module...")
    
    # Test clean_text
    sample_text = "<br>This is a TEST! Visit http://example.com for more... 123"
    cleaned = clean_text(sample_text)
    print(f"\nOriginal: {sample_text}")
    print(f"Cleaned: {cleaned}")
    
    # Test build_vocab
    sample_texts = ["hello world", "hello there", "world is beautiful"]
    vocab = build_vocab(sample_texts, max_vocab_size=100)
    print(f"\nVocabulary: {vocab}")
    
    # Test text_to_indices
    indices = text_to_indices("hello world", vocab, max_length=10)
    print(f"\nIndices: {indices}")
    
    print("\nPreprocessing module test passed!")
