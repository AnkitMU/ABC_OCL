"""
Validation module for OCL verification framework
"""

from .model_consistency_checker import (
    ModelConsistencyChecker,
    ValidationResult,
    OCLConstraintParser,
    quick_validate
)

__all__ = [
    'ModelConsistencyChecker',
    'ValidationResult',
    'OCLConstraintParser',
    'quick_validate'
]
