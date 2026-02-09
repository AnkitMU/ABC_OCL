#!/usr/bin/env python3
"""
Auto-generated Pattern Classification Regression Tests
======================================================

This test suite ensures no regressions in pattern classification.
Run with: pytest test_pattern_regression.py -v
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from ssr_ocl.validation.pattern_validation import PatternValidator


@pytest.fixture
def validator():
    """Create validator instance"""
    return PatternValidator(use_neural_classifier=False)


def test_all_ground_truth_examples(validator):
    """Test all ground truth examples for correctness"""
    summary = validator.validate_all()
    
    # Require 100% acceptable accuracy
    assert summary['acceptable_accuracy'] == 100.0, (
        f"Pattern validation failed: {summary['acceptable_accuracy']:.1f}% accuracy "
        f"({summary['acceptable']} / {summary['total']} examples)"
    )


def test_no_false_positives_pairwise_uniqueness(validator):
    """Ensure pairwise uniqueness not confused with other patterns"""
    constraint = "self.items->forAll(x, y | x <> y implies x.id <> y.id)"
    result = validator.validate_example(
        validator.ground_truth[0]  # First example is pairwise uniqueness
    )
    assert result.is_acceptable, f"Failed: {result.issues}"


def test_normalization_improves_classification(validator):
    """Ensure normalization helps classification"""
    # Test guarded implication with normalization
    constraint = "self.items->isEmpty() or self.items->forAll(i | i.valid)"
    
    # Without normalization
    pattern_raw = validator.detector.detect_pattern(constraint)
    
    # With normalization
    normalized = validator.normalizer.normalize(constraint)
    pattern_norm = validator.detector.detect_pattern(normalized)
    
    # Normalized should be implies or contractual, not boolean_operations
    assert pattern_norm.value in ['boolean_guard_implies', 'contractual_temporal'], (
        f"Normalization failed: {pattern_norm.value}"
    )


def test_contractual_temporal_detection(validator):
    """Ensure contractual/temporal patterns correctly identified"""
    # This is the ValidWindowAndBranch case
    constraint = "self.dateTo > self.dateFrom and (self.vehicle->isEmpty() or self.vehicle.branch = self.branch)"
    normalized = validator.normalizer.normalize(constraint)
    pattern = validator.detector.detect_pattern(normalized)
    
    # Should be contractual_temporal or boolean_guard_implies
    assert pattern.value in ['contractual_temporal', 'boolean_guard_implies'], (
        f"Unexpected pattern: {pattern.value}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
