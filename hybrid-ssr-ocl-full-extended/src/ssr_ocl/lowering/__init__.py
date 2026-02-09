"""
OCL to SMT Lowering Module
===========================

This module handles the translation of OCL constraints to SMT (Z3) format.
"""

from .association_backed_encoder import XMIMetadataExtractor
from .unified_smt_encoder import UnifiedSMTEncoder

__all__ = [
    'XMIMetadataExtractor',
    'UnifiedSMTEncoder',
]
