"""
Training Script for IMDB Sentiment Classification

This script handles the training and evaluation of the transformer-based
sentiment classification model.

Usage:
    python src/train.py --data_path "IMDB Dataset.csv" --epochs 10
"""

import argparse
import os
import json
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix, 
    classification_report
)
from tqdm import tqdm

# Add parent directory to path for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import TransformerClassifier, get_model_summary
from src.data_preprocessing import (
    load_and_preprocess_data, 
    create_dataloaders
)


def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    
    Args:
        model: TransformerClassifier model
        dataloader: Training DataLoader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        
    Returns:
        avg_loss: Average loss for the epoch
        accuracy: Training accuracy
    """
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch_texts, batch_labels in progress_bar:
        batch_texts = batch_texts.to(device)
        batch_labels = batch_labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(batch_texts)
        loss = criterion(outputs, batch_labels)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        # Collect predictions
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(batch_labels.cpu().numpy())
        
        progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """
    Evaluate the model.
    
    Args:
        model: TransformerClassifier model
        dataloader: Evaluation DataLoader
        criterion: Loss function
        device: Device to evaluate on
        
    Returns:
        avg_loss: Average loss
        accuracy: Accuracy score
        precision: Precision score
        recall: Recall score
        f1: F1 score
        all_preds: List of predictions
        all_labels: List of true labels
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_texts, batch_labels in tqdm(dataloader, desc="Evaluating"):
            batch_texts = batch_texts.to(device)
            batch_labels = batch_labels.to(device)
            
            outputs = model(batch_texts)
            loss = criterion(outputs, batch_labels)
            
            total_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch_labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    return avg_loss, accuracy, precision, recall, f1, all_preds, all_labels


def train(args):
    """
    Main training function.
    
    Args:
        args: Command line arguments
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load and preprocess data
    print("\n" + "=" * 70)
    print("DATA LOADING AND PREPROCESSING")
    print("=" * 70)
    
    X_train, X_test, y_train, y_test, vocab, label_map = load_and_preprocess_data(
        data_path=args.data_path,
        max_vocab_size=args.max_vocab_size,
        max_length=args.max_length,
        test_size=args.test_size,
        random_state=args.random_state
    )
    
    # Create dataloaders
    train_loader, test_loader = create_dataloaders(
        X_train, X_test, y_train, y_test, batch_size=args.batch_size
    )
    
    # Initialize model
    print("\n" + "=" * 70)
    print("MODEL INITIALIZATION")
    print("=" * 70)
    
    model = TransformerClassifier(
        vocab_size=len(vocab),
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        max_len=args.max_length,
        num_classes=2,
        dropout=args.dropout
    ).to(device)
    
    print(get_model_summary(model))
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(), 
        lr=args.learning_rate, 
        weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs, 
        eta_min=1e-6
    )
    
    # Create save directory
    os.makedirs(args.save_path, exist_ok=True)
    
    # Training loop
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)
    
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    best_val_accuracy = 0
    
    for epoch in range(args.epochs):
        start_time = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # Evaluate
        val_loss, val_acc, val_prec, val_rec, val_f1, _, _ = evaluate(
            model, test_loader, criterion, device
        )
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # Update learning rate
        scheduler.step()
        
        epoch_time = time.time() - start_time
        
        print(f"\nEpoch {epoch + 1}/{args.epochs} (Time: {epoch_time:.2f}s)")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        print(f"  Val Precision: {val_prec:.4f} | Val Recall: {val_rec:.4f} | Val F1: {val_f1:.4f}")
        print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            
            # Save complete checkpoint
            model_checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_acc,
                'vocab': vocab,
                'config': {
                    'vocab_size': len(vocab),
                    'd_model': args.d_model,
                    'num_heads': args.num_heads,
                    'num_layers': args.num_layers,
                    'd_ff': args.d_ff,
                    'max_len': args.max_length,
                    'num_classes': 2,
                    'dropout': args.dropout
                },
                'label_map': label_map
            }
            
            checkpoint_path = os.path.join(args.save_path, 'sentiment_model.pth')
            torch.save(model_checkpoint, checkpoint_path)
            print(f"  ✓ New best model saved! (Accuracy: {val_acc:.4f})")
    
    # Save vocabulary and config separately
    vocab_path = os.path.join(args.save_path, 'vocab.json')
    with open(vocab_path, 'w') as f:
        json.dump({str(k): v for k, v in vocab.items()}, f, indent=2)
    
    config_path = os.path.join(args.save_path, 'model_config.json')
    config = {
        'vocab_size': len(vocab),
        'd_model': args.d_model,
        'num_heads': args.num_heads,
        'num_layers': args.num_layers,
        'd_ff': args.d_ff,
        'max_len': args.max_length,
        'num_classes': 2,
        'dropout': args.dropout,
        'label_map': label_map,
        'reverse_label_map': {v: k for k, v in label_map.items()}
    }
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Training history
    history_path = os.path.join(args.save_path, 'training_history.json')
    history = {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'best_val_accuracy': best_val_accuracy
    }
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "=" * 70)
    print(f"Training Complete! Best Validation Accuracy: {best_val_accuracy:.4f}")
    print("=" * 70)
    
    # Final evaluation
    print("\n" + "=" * 70)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 70)
    
    # Load best model
    checkpoint = torch.load(os.path.join(args.save_path, 'sentiment_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    _, final_acc, final_prec, final_rec, final_f1, all_preds, all_labels = evaluate(
        model, test_loader, criterion, device
    )
    
    print(f"\nTest Accuracy: {final_acc:.4f}")
    print(f"Test Precision: {final_prec:.4f}")
    print(f"Test Recall: {final_rec:.4f}")
    print(f"Test F1 Score: {final_f1:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Negative', 'Positive']))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(f"              Predicted")
    print(f"              Neg    Pos")
    print(f"Actual Neg   {cm[0][0]:5d}  {cm[0][1]:5d}")
    print(f"       Pos   {cm[1][0]:5d}  {cm[1][1]:5d}")
    
    print("\n" + "=" * 70)
    print("Saved files:")
    print(f"  - {os.path.join(args.save_path, 'sentiment_model.pth')}")
    print(f"  - {vocab_path}")
    print(f"  - {config_path}")
    print(f"  - {history_path}")
    print("=" * 70)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train IMDB Sentiment Classification Model"
    )
    
    # Data arguments
    parser.add_argument(
        '--data_path', 
        type=str, 
        default='IMDB Dataset.csv',
        help='Path to the IMDB dataset CSV file'
    )
    parser.add_argument(
        '--max_vocab_size', 
        type=int, 
        default=25000,
        help='Maximum vocabulary size'
    )
    parser.add_argument(
        '--max_length', 
        type=int, 
        default=256,
        help='Maximum sequence length'
    )
    parser.add_argument(
        '--test_size', 
        type=float, 
        default=0.2,
        help='Test set size fraction'
    )
    parser.add_argument(
        '--random_state', 
        type=int, 
        default=42,
        help='Random seed for reproducibility'
    )
    
    # Model arguments
    parser.add_argument(
        '--d_model', 
        type=int, 
        default=256,
        help='Model dimension'
    )
    parser.add_argument(
        '--num_heads', 
        type=int, 
        default=8,
        help='Number of attention heads'
    )
    parser.add_argument(
        '--num_layers', 
        type=int, 
        default=4,
        help='Number of transformer layers'
    )
    parser.add_argument(
        '--d_ff', 
        type=int, 
        default=512,
        help='Feed-forward dimension'
    )
    parser.add_argument(
        '--dropout', 
        type=float, 
        default=0.1,
        help='Dropout rate'
    )
    
    # Training arguments
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=10,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=32,
        help='Batch size'
    )
    parser.add_argument(
        '--learning_rate', 
        type=float, 
        default=1e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--weight_decay', 
        type=float, 
        default=0.01,
        help='Weight decay'
    )
    
    # Save arguments
    parser.add_argument(
        '--save_path', 
        type=str, 
        default='saved_models',
        help='Path to save model checkpoints'
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
