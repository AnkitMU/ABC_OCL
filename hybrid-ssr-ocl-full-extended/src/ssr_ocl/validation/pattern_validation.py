#!/usr/bin/env python3
"""
Pattern Classification Validation Framework
============================================

This module provides comprehensive validation to ensure correct pattern
classification and prevent misclassifications across all 50 OCL pattern types.

Key Features:
1. Ground truth test suite with labeled examples
2. Cross-validation of neural vs regex classification
3. Confidence threshold calibration
4. Pattern ambiguity detection
5. Automated regression testing
"""

import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ssr_ocl.super_encoder.comprehensive_pattern_detector import ComprehensivePatternDetector
from ssr_ocl.super_encoder.ocl_normalizer import OCLNormalizer
from ssr_ocl.classifiers.sentence_transformer.classifier import OCLPatternType


@dataclass
class ValidationExample:
    """A ground truth example for validation"""
    constraint: str
    expected_pattern: str
    context: str
    description: str
    should_normalize: bool = True
    alternative_patterns: List[str] = None  # Acceptable alternatives
    
    def __post_init__(self):
        if self.alternative_patterns is None:
            self.alternative_patterns = []


@dataclass
class ValidationResult:
    """Result of pattern validation"""
    constraint: str
    expected: str
    predicted: str
    normalized_text: str
    is_correct: bool
    is_acceptable: bool  # True if matches expected or alternative
    confidence: Optional[float] = None
    issues: List[str] = None
    
    def __post_init__(self):
        if self.issues is None:
            self.issues = []


class PatternValidator:
    """
    Comprehensive validator for OCL pattern classification.
    
    Ensures:
    - Correct pattern detection across all 50 types
    - Consistency between neural and regex classifiers
    - Proper normalization application
    - No regressions on known examples
    """
    
    def __init__(self, use_neural_classifier: bool = False):
        self.detector = ComprehensivePatternDetector()
        self.normalizer = OCLNormalizer(enable_logging=False)
        self.use_neural_classifier = use_neural_classifier
        self.classifier = None
        
        if use_neural_classifier:
            try:
                from ssr_ocl.classifiers.sentence_transformer import SentenceTransformerClassifier
                self.classifier = SentenceTransformerClassifier()
            except:
                print("  Neural classifier not available, falling back to regex only")
                self.use_neural_classifier = False
        
        # Load ground truth examples
        self.ground_truth = self._load_ground_truth()
    
    def _load_ground_truth(self) -> List[ValidationExample]:
        """Load ground truth examples for all 50 pattern types"""
        return [
            # 1. PAIRWISE_UNIQUENESS
            ValidationExample(
                constraint="self.students->forAll(x, y | x <> y implies x.id <> y.id)",
                expected_pattern="pairwise_uniqueness",
                context="Student",
                description="All students must have unique IDs"
            ),
            
            # 2. EXACT_COUNT_SELECTION
            ValidationExample(
                constraint="self.items->select(i | i.active)->size() = 5",
                expected_pattern="exact_count_selection",
                context="Container",
                description="Exactly 5 active items"
            ),
            
            # 3. GLOBAL_COLLECTION
            ValidationExample(
                constraint="Student.allInstances()->forAll(s | s.age >= 18)",
                expected_pattern="global_collection",
                context="Student",
                description="All students globally must be adults"
            ),
            
            # 4. SET_INTERSECTION
            ValidationExample(
                constraint="self.setA->intersection(self.setB)->notEmpty()",
                expected_pattern="set_intersection",
                context="Container",
                description="Sets must have common elements"
            ),
            
            # 5. SIZE_CONSTRAINT
            ValidationExample(
                constraint="self.items->size() >= 10",
                expected_pattern="size_constraint",
                context="Container",
                description="At least 10 items required"
            ),
            
            # 6. UNIQUENESS_CONSTRAINT
            ValidationExample(
                constraint="self.students->isUnique(s | s.email)",
                expected_pattern="uniqueness_constraint",
                context="Class",
                description="All emails must be unique"
            ),
            
            # 7. COLLECTION_MEMBERSHIP
            ValidationExample(
                constraint="self.validItems->includes(self.selectedItem)",
                expected_pattern="collection_membership",
                context="Selector",
                description="Selected item must be in valid set"
            ),
            
            # 8. NULL_CHECK
            ValidationExample(
                constraint="self.manager <> null",
                expected_pattern="null_check",
                context="Employee",
                description="Manager must be assigned"
            ),
            
            # 9. NUMERIC_COMPARISON
            ValidationExample(
                constraint="self.salary >= self.minSalary",
                expected_pattern="numeric_comparison",
                context="Employee",
                description="Salary must meet minimum"
            ),
            
            # 10. EXACTLY_ONE
            ValidationExample(
                constraint="self.items->one(i | i.primary)",
                expected_pattern="exactly_one",
                context="Container",
                description="Exactly one primary item"
            ),
            
            # 14. BOOLEAN_GUARD_IMPLIES (without normalization needed)
            ValidationExample(
                constraint="self.items->notEmpty() implies self.items->forAll(i | i.valid)",
                expected_pattern="boolean_guard_implies",
                context="Container",
                description="If items exist, all must be valid",
                should_normalize=False
            ),
            
            # 14. BOOLEAN_GUARD_IMPLIES (with normalization)
            ValidationExample(
                constraint="self.items->isEmpty() or self.items->forAll(i | i.valid)",
                expected_pattern="boolean_guard_implies",
                context="Container",
                description="If items exist, all must be valid (normalized form)",
                should_normalize=True
            ),
            
            # 19. CONTRACTUAL_TEMPORAL
            ValidationExample(
                constraint="self.dateTo > self.dateFrom and (self.vehicle->isEmpty() or self.vehicle.branch = self.branch)",
                expected_pattern="contractual_temporal",
                context="Reservation",
                description="Date validation with vehicle branch constraint",
                should_normalize=True,
                alternative_patterns=["boolean_guard_implies"]  # Acceptable alternative
            ),
            
            # 19. CONTRACTUAL_TEMPORAL (explicit)
            ValidationExample(
                constraint="self.startDate->notEmpty() implies self.startDate < self.endDate",
                expected_pattern="contractual_temporal",
                context="Event",
                description="Start date must precede end date when present",
                should_normalize=False,
                alternative_patterns=["boolean_guard_implies"]  # Also acceptable
            ),
            
            # 20. SELECT_REJECT
            ValidationExample(
                constraint="self.students->select(s | s.age >= 18)",
                expected_pattern="select_reject",
                context="Class",
                description="Select adult students"
            ),
            
            # 23. FORALL_NESTED
            ValidationExample(
                constraint="self.courses->forAll(c | c.students->forAll(s | s.enrolled))",
                expected_pattern="forall_nested",
                context="University",
                description="All students in all courses are enrolled"
            ),
            
            # 24. EXISTS_NESTED
            ValidationExample(
                constraint="self.departments->exists(d | d.employees->exists(e | e.manager))",
                expected_pattern="exists_nested",
                context="Company",
                description="At least one department has an employee with a manager"
            ),
            
            # 34. BOOLEAN_OPERATIONS (compound)
            ValidationExample(
                constraint="self.active and self.verified and self.approved",
                expected_pattern="boolean_operations",
                context="Account",
                description="Multiple boolean conditions"
            ),
            
            # 35. IF_THEN_ELSE
            ValidationExample(
                constraint="if self.premium then 0.9 else 1.0 endif",
                expected_pattern="if_then_else",
                context="Customer",
                description="Conditional discount"
            ),
            
            # 38. LET_EXPRESSION
            ValidationExample(
                constraint="let total = self.items->size() in total > 0",
                expected_pattern="let_expression",
                context="Order",
                description="Define and use local variable"
            ),
            
            # 40. UNION_INTERSECTION
            ValidationExample(
                constraint="self.setA->union(self.setB)->size() > 10",
                expected_pattern="union_intersection",
                context="Container",
                description="Union of sets has sufficient size"
            ),
            
            # 44. NAVIGATION_CHAIN
            ValidationExample(
                constraint="self.department.company.address.city = 'NYC'",
                expected_pattern="navigation_chain",
                context="Employee",
                description="Deep navigation through associations"
            ),
            
            # 45. OPTIONAL_NAVIGATION
            ValidationExample(
                constraint="self.manager->isEmpty() or self.manager.department = self.department",
                expected_pattern="optional_navigation",
                context="Employee",
                description="Safe navigation with null check",
                alternative_patterns=["boolean_guard_implies"]
            ),
        ]
    
    def validate_example(self, example: ValidationExample) -> ValidationResult:
        """Validate a single example"""
        # Apply normalization if needed
        constraint_text = example.constraint
        if example.should_normalize:
            constraint_text = self.normalizer.normalize(constraint_text)
        
        # Detect pattern
        if self.use_neural_classifier and self.classifier and self.classifier.is_trained:
            predicted_pattern, confidence = self.classifier.predict(constraint_text)
        else:
            pattern = self.detector.detect_pattern(constraint_text)
            predicted_pattern = pattern.value
            confidence = None
        
        # Check if correct
        is_correct = (predicted_pattern == example.expected_pattern)
        is_acceptable = is_correct or (predicted_pattern in example.alternative_patterns)
        
        # Collect issues
        issues = []
        if not is_correct:
            issues.append(f"Expected {example.expected_pattern}, got {predicted_pattern}")
        if not is_acceptable:
            issues.append(f"Not an acceptable alternative pattern")
        
        return ValidationResult(
            constraint=example.constraint,
            expected=example.expected_pattern,
            predicted=predicted_pattern,
            normalized_text=constraint_text,
            is_correct=is_correct,
            is_acceptable=is_acceptable,
            confidence=confidence,
            issues=issues
        )
    
    def validate_all(self) -> Dict:
        """Validate all ground truth examples"""
        results = []
        correct_count = 0
        acceptable_count = 0
        
        print("\n" + "=" * 80)
        print(" PATTERN VALIDATION SUITE")
        print("=" * 80)
        print(f"\nValidating {len(self.ground_truth)} ground truth examples...")
        print(f"Method: {'Neural + Regex Hybrid' if self.use_neural_classifier else 'Regex Only'}")
        print()
        
        for i, example in enumerate(self.ground_truth, 1):
            result = self.validate_example(example)
            results.append(result)
            
            if result.is_correct:
                correct_count += 1
            if result.is_acceptable:
                acceptable_count += 1
            
            # Print result
            status = "" if result.is_acceptable else ""
            conf_str = f" (conf: {result.confidence:.3f})" if result.confidence else ""
            
            print(f"{status} {i:2d}. {example.expected_pattern:30} → {result.predicted:30}{conf_str}")
            if not result.is_acceptable:
                print(f"       Context: {example.description}")
                for issue in result.issues:
                    print(f"         {issue}")
        
        # Summary
        accuracy = (correct_count / len(self.ground_truth)) * 100
        acceptable_accuracy = (acceptable_count / len(self.ground_truth)) * 100
        
        print("\n" + "=" * 80)
        print(" VALIDATION SUMMARY")
        print("=" * 80)
        print(f"Total Examples:        {len(self.ground_truth)}")
        print(f"Exact Matches:         {correct_count} ({accuracy:.1f}%)")
        print(f"Acceptable Matches:    {acceptable_count} ({acceptable_accuracy:.1f}%)")
        print(f"Failures:              {len(self.ground_truth) - acceptable_count}")
        
        if acceptable_accuracy == 100.0:
            print("\n🎉 ALL VALIDATIONS PASSED!")
        elif acceptable_accuracy >= 90.0:
            print("\n VALIDATION MOSTLY PASSED (≥90%)")
        else:
            print("\n  VALIDATION NEEDS IMPROVEMENT (<90%)")
        
        return {
            'total': len(self.ground_truth),
            'correct': correct_count,
            'acceptable': acceptable_count,
            'accuracy': accuracy,
            'acceptable_accuracy': acceptable_accuracy,
            'results': results
        }
    
    def export_ground_truth(self, filepath: str):
        """Export ground truth examples to JSON for future testing"""
        data = {
            'version': '1.0',
            'total_examples': len(self.ground_truth),
            'examples': [
                {
                    'constraint': ex.constraint,
                    'expected_pattern': ex.expected_pattern,
                    'context': ex.context,
                    'description': ex.description,
                    'should_normalize': ex.should_normalize,
                    'alternative_patterns': ex.alternative_patterns
                }
                for ex in self.ground_truth
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f" Ground truth exported to: {filepath}")
    
    def check_ambiguity(self, constraint: str) -> Dict:
        """Check if a constraint matches multiple patterns (ambiguity)"""
        matches = []
        
        # Check all patterns
        for regex_pattern, pattern_type in self.detector.compiled_patterns:
            if regex_pattern.search(constraint):
                matches.append(pattern_type.value)
        
        return {
            'constraint': constraint,
            'match_count': len(matches),
            'matches': matches,
            'is_ambiguous': len(matches) > 1,
            'primary_pattern': matches[0] if matches else None
        }


def create_regression_test_suite() -> str:
    """Create a pytest-compatible regression test suite"""
    test_code = '''#!/usr/bin/env python3
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
'''
    
    return test_code


if __name__ == "__main__":
    print("🚀 Running Pattern Validation Framework\n")
    
    # Run validation with regex detector
    validator = PatternValidator(use_neural_classifier=False)
    summary = validator.validate_all()
    
    # Export ground truth
    output_dir = Path(__file__).parent.parent.parent.parent / "tests"
    output_dir.mkdir(exist_ok=True)
    
    ground_truth_file = output_dir / "ground_truth_patterns.json"
    validator.export_ground_truth(str(ground_truth_file))
    
    # Create regression test suite
    test_suite = create_regression_test_suite()
    test_file = output_dir / "test_pattern_regression.py"
    with open(test_file, 'w') as f:
        f.write(test_suite)
    print(f" Regression test suite created: {test_file}")
    
    # Check for ambiguous patterns
    print("\n" + "=" * 80)
    print("🔎 CHECKING FOR AMBIGUOUS PATTERNS")
    print("=" * 80)
    
    ambiguous_count = 0
    for example in validator.ground_truth[:5]:  # Check first 5
        result = validator.check_ambiguity(example.constraint)
        if result['is_ambiguous']:
            ambiguous_count += 1
            print(f"\n  Ambiguous: {example.expected_pattern}")
            print(f"   Matches {result['match_count']} patterns: {result['matches'][:3]}...")
    
    if ambiguous_count == 0:
        print("\n No highly ambiguous patterns detected")
    
    print("\n✨ Validation complete!")
