"""
CodeBERT-based OCL Pattern Classifier
Model: microsoft/codebert-base
Embedding dimension: 768
Training: Fine-tuned classification head
"""

from .classifier import CodeBERTClassifier, CodeBERTOCLClassifier
from .train import train_codebert_classifier, load_training_data_from_json

__all__ = [
    'CodeBERTClassifier',
    'CodeBERTOCLClassifier',
    'train_codebert_classifier',
    'load_training_data_from_json'
]
