"""
OCL Pattern Classifiers Module

Two separate implementations:
1. SentenceTransformer (all-MiniLM-L6-v2) - Lightweight, fast inference
2. CodeBERT (microsoft/codebert-base) - Better accuracy, code-specific

Usage:
    from ssr_ocl.classifiers.sentence_transformer import SentenceTransformerClassifier
    from ssr_ocl.classifiers.codebert import CodeBERTOCLClassifier
"""

from . import sentence_transformer
from . import codebert

__all__ = ['sentence_transformer', 'codebert']
