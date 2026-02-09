"""
Neural-Symbolic Reasoning Module for OCL Verification

This module integrates neural and symbolic components:
- Neural pattern classification for OCL constraints
- Neural Z3 encoding generation  
- Neural explanation of verification results
- Symbolic Z3 solving (unchanged)

Note: Classifiers have been moved to ssr_ocl.classifiers module:
- SentenceTransformer classifier: src/ssr_ocl/classifiers/sentence_transformer/
- CodeBERT classifier: src/ssr_ocl/classifiers/codebert/
"""

# Neural-Symbolic components
from .encoding_generator import NeuralEncodingGenerator, get_neural_encoding_generator
from .explainer import NeuralExplainer, get_neural_explainer

# Import classifiers from new location for backward compatibility
try:
    from ..classifiers.sentence_transformer import SentenceTransformerClassifier
except ImportError:
    SentenceTransformerClassifier = None

try:
    from ..classifiers.codebert import CodeBERTOCLClassifier
except ImportError:
    CodeBERTOCLClassifier = None

__all__ = [
    'NeuralEncodingGenerator',
    'get_neural_encoding_generator', 
    'NeuralExplainer',
    'get_neural_explainer',
    'SentenceTransformerClassifier',
    'CodeBERTOCLClassifier'
]
