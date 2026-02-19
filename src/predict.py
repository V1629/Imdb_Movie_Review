"""
Inference Script for IMDB Sentiment Classification

This script provides prediction functionality for the trained
sentiment classification model.

Usage:
    python src/predict.py --text "This movie was fantastic!"
    python src/predict.py --interactive
"""

import argparse
import os
import sys
import torch
import torch.nn.functional as F

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import TransformerClassifier
from src.data_preprocessing import clean_text, text_to_indices


class SentimentPredictor:
    """
    Sentiment Predictor class for making predictions on movie reviews.
    
    Args:
        model_path: Path to the saved model checkpoint
        device: Device to run inference on ('cpu' or 'cuda')
    """
    
    def __init__(self, model_path: str, device: str = None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Loading model from {model_path}...")
        print(f"Using device: {self.device}")
        
        # Load checkpoint
        self.checkpoint = torch.load(model_path, map_location=self.device)
        
        # Get config and vocab
        # Handle both old format (without config) and new format (with config)
        if 'config' in self.checkpoint:
            self.config = self.checkpoint['config']
        else:
            # Default config for models saved without config
            self.vocab = self.checkpoint['vocab']
            self.config = {
                'vocab_size': len(self.vocab),
                'd_model': 256,
                'num_heads': 8,
                'num_layers': 4,
                'd_ff': 512,
                'max_len': 256,
                'num_classes': 2,
                'dropout': 0.1
            }
        
        self.vocab = self.checkpoint['vocab']
        self.label_map = self.checkpoint.get('label_map', {'positive': 1, 'negative': 0})
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}
        
        # Initialize model
        self.model = TransformerClassifier(
            vocab_size=self.config['vocab_size'],
            d_model=self.config['d_model'],
            num_heads=self.config['num_heads'],
            num_layers=self.config['num_layers'],
            d_ff=self.config['d_ff'],
            max_len=self.config['max_len'],
            num_classes=self.config['num_classes'],
            dropout=self.config['dropout']
        )
        
        # Load weights
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        self.max_length = self.config['max_len']
        
        print("Model loaded successfully!")
    
    def predict(self, text: str) -> dict:
        """
        Predict sentiment for a given text.
        
        Args:
            text: Movie review text
            
        Returns:
            Dictionary containing prediction results
        """
        # Clean text
        cleaned_text = clean_text(text)
        
        # Convert to indices
        indices = text_to_indices(cleaned_text, self.vocab, self.max_length)
        
        # Convert to tensor
        input_tensor = torch.tensor([indices], dtype=torch.long).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        sentiment = self.reverse_label_map.get(predicted_class, 'unknown')
        
        return {
            'text': text[:200] + '...' if len(text) > 200 else text,
            'cleaned_text': cleaned_text[:200] + '...' if len(cleaned_text) > 200 else cleaned_text,
            'sentiment': sentiment,
            'confidence': confidence,
            'positive_probability': probabilities[0][1].item(),
            'negative_probability': probabilities[0][0].item(),
            'predicted_class': predicted_class
        }
    
    def predict_batch(self, texts: list) -> list:
        """
        Predict sentiment for a batch of texts.
        
        Args:
            texts: List of movie review texts
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        for text in texts:
            result = self.predict(text)
            results.append(result)
        return results


def display_prediction(result: dict):
    """
    Display prediction results in a formatted way.
    
    Args:
        result: Prediction result dictionary
    """
    print("\n" + "=" * 60)
    print("SENTIMENT PREDICTION RESULT")
    print("=" * 60)
    print(f"\nReview: {result['text']}")
    
    # Color-coded sentiment display
    sentiment = result['sentiment'].upper()
    confidence = result['confidence']
    
    print(f"\n🎬 Predicted Sentiment: {sentiment}")
    print(f"📊 Confidence: {confidence:.2%}")
    print(f"\n   Positive Probability: {result['positive_probability']:.2%}")
    print(f"   Negative Probability: {result['negative_probability']:.2%}")
    print("=" * 60)


def interactive_mode(predictor: SentimentPredictor):
    """
    Run interactive prediction mode.
    
    Args:
        predictor: SentimentPredictor instance
    """
    print("\n" + "=" * 60)
    print("INTERACTIVE SENTIMENT PREDICTOR")
    print("=" * 60)
    print("Enter a movie review to predict its sentiment.")
    print("Type 'quit' or 'exit' to stop.\n")
    
    while True:
        try:
            user_input = input("\nEnter your review: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Exiting interactive predictor. Goodbye!")
                break
            
            if not user_input:
                print("Please enter a valid review.")
                continue
            
            result = predictor.predict(user_input)
            display_prediction(result)
            
        except KeyboardInterrupt:
            print("\n\nExiting interactive predictor. Goodbye!")
            break


def find_model_path():
    """
    Try to find the model file in common locations.
    
    Returns:
        str: Path to the model file, or None if not found
    """
    common_paths = [
        'saved_models/sentiment_model.pth',
        'best_model.pth',
        '../saved_models/sentiment_model.pth',
        '../best_model.pth',
    ]
    
    for path in common_paths:
        if os.path.exists(path):
            return path
    
    return None


def main():
    """Main function for command-line inference."""
    parser = argparse.ArgumentParser(
        description="IMDB Sentiment Classification Inference"
    )
    
    parser.add_argument(
        '--model_path',
        type=str,
        default=None,
        help='Path to the saved model checkpoint'
    )
    parser.add_argument(
        '--text',
        type=str,
        default=None,
        help='Text to predict sentiment for'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run in interactive mode'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cpu', 'cuda'],
        help='Device to run inference on'
    )
    
    args = parser.parse_args()
    
    # Find model path
    if args.model_path is None:
        args.model_path = find_model_path()
        if args.model_path is None:
            print("Error: Could not find model file. Please specify --model_path")
            print("Common locations: saved_models/sentiment_model.pth, best_model.pth")
            sys.exit(1)
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        sys.exit(1)
    
    # Initialize predictor
    predictor = SentimentPredictor(args.model_path, args.device)
    
    if args.interactive:
        # Interactive mode
        interactive_mode(predictor)
    elif args.text:
        # Single prediction
        result = predictor.predict(args.text)
        display_prediction(result)
    else:
        # Demo predictions
        print("\n" + "=" * 60)
        print("DEMO PREDICTIONS")
        print("=" * 60)
        
        demo_reviews = [
            "This movie was absolutely fantastic! The acting was superb, the plot was engaging, and I was on the edge of my seat the entire time. Highly recommend!",
            "What a terrible waste of time. The story made no sense, the characters were flat and uninteresting, and the special effects were laughably bad.",
            "An average film with some good moments but nothing spectacular. The lead actor did a decent job, but the script could have been better.",
        ]
        
        for i, review in enumerate(demo_reviews, 1):
            print(f"\n--- Demo Review #{i} ---")
            result = predictor.predict(review)
            display_prediction(result)
        
        print("\n\nTo run your own predictions, use:")
        print("  python src/predict.py --text 'Your review here'")
        print("  python src/predict.py --interactive")


if __name__ == "__main__":
    main()
