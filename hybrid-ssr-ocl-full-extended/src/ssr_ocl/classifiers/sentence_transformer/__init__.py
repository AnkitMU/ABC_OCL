"""
SentenceTransformer-based OCL Pattern Classifier with XMI-based Domain Adaptation
Model: all-MiniLM-L6-v2
Embedding dimension: 384
Training: LogisticRegression on embeddings
Domain Adaptation: XMI-based automatic vocabulary extraction
"""

from .classifier import SentenceTransformerClassifier, OCLPatternType
from .xmi_based_domain_adapter import GenericDomainDataGenerator

# Dummy functions for backward compatibility
def quick_adapt_domain(xmi_file, constraints=None):
    """Placeholder - use framework_integration.py instead"""
    pass

class DomainAdaptationPipeline:
    """Placeholder - use framework_integration.py instead"""
    pass

__all__ = [
    'SentenceTransformerClassifier',
    'OCLPatternType',
    'DomainAdaptationPipeline',
    'quick_adapt_domain',
    'GenericDomainDataGenerator'
]
