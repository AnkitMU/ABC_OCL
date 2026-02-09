#!/usr/bin/env python3
"""
OCL Constraint Normalization Module
====================================

This module provides normalization utilities for OCL constraints to improve 
pattern classification by rewriting logically equivalent but syntactically 
different OCL idioms into canonical forms.

Key transformations:
1. Guarded Implication: X->isEmpty() or P  →  X->notEmpty() implies P
2. De Morgan's Laws: not (A and B)  →  not A or not B
3. Double Negation: not not P  →  P
4. Multiplicity-aware rewrites using XMI metadata

This normalization runs BEFORE classification to help the neural classifier
recognize semantic patterns that may be expressed differently.
"""

import re
from typing import Optional


class OCLNormalizer:
    """
    Normalizes OCL constraints to canonical forms for better classification.
    
    Transformations:
    - Guarded implication idioms (isEmpty() or P → notEmpty() implies P)
    - Boolean logic simplification
    - Multiplicity-aware normalization with XMI context
    """
    
    # Normalization patterns (priority ordered)
    NORMALIZATION_RULES = [
        # ═══════════════════════════════════════════════════════════════════
        # 1. Guarded Implication Patterns
        # ═══════════════════════════════════════════════════════════════════
        
        # Pattern: X->isEmpty() or P  →  X->notEmpty() implies P
        (r'(\w+(?:\.\w+)*)->isEmpty\(\)\s+or\s+(.+)',
         r'\1->notEmpty() implies \2',
         'guarded_implication_isEmpty'),
        
        # Pattern: X->isEmpty() or (P)  →  X->notEmpty() implies (P)
        (r'(\w+(?:\.\w+)*)->isEmpty\(\)\s+or\s+\((.+)\)',
         r'\1->notEmpty() implies (\2)',
         'guarded_implication_isEmpty_paren'),
        
        # Pattern: X = null or P  →  X <> null implies P
        (r'(\w+(?:\.\w+)*)\s*=\s*null\s+or\s+(.+)',
         r'\1 <> null implies \2',
         'guarded_implication_null_eq'),
        
        # Pattern: null = X or P  →  X <> null implies P
        (r'null\s*=\s*(\w+(?:\.\w+)*)\s+or\s+(.+)',
         r'\1 <> null implies \2',
         'guarded_implication_null_eq_rev'),
        
        # Pattern: X->size() = 0 or P  →  X->notEmpty() implies P
        (r'(\w+(?:\.\w+)*)->size\(\)\s*=\s*0\s+or\s+(.+)',
         r'\1->notEmpty() implies \2',
         'guarded_implication_size_zero'),
        
        # Pattern: X->size() < 1 or P  →  X->notEmpty() implies P
        (r'(\w+(?:\.\w+)*)->size\(\)\s*<\s*1\s+or\s+(.+)',
         r'\1->notEmpty() implies \2',
         'guarded_implication_size_lt_one'),
        
        # ═══════════════════════════════════════════════════════════════════
        # 2. Nested Collection Guarded Implications
        # ═══════════════════════════════════════════════════════════════════
        
        # Pattern: self.X->isEmpty() or self.X->...  →  self.X->notEmpty() implies self.X->...
        (r'(self\.\w+)->isEmpty\(\)\s+or\s+(\1->.+)',
         r'\1->notEmpty() implies \2',
         'guarded_self_collection'),
        
        # ═══════════════════════════════════════════════════════════════════
        # 3. Boolean Logic Normalization
        # ═══════════════════════════════════════════════════════════════════
        
        # Pattern: not (A and B)  →  not A or not B (De Morgan)
        (r'not\s*\(\s*(.+?)\s+and\s+(.+?)\s*\)',
         r'not (\1) or not (\2)',
         'demorgan_and'),
        
        # Pattern: not (A or B)  →  not A and not B (De Morgan)
        (r'not\s*\(\s*(.+?)\s+or\s+(.+?)\s*\)',
         r'not (\1) and not (\2)',
         'demorgan_or'),
        
        # Pattern: not not P  →  P (double negation)
        (r'not\s+not\s+(.+)',
         r'\1',
         'double_negation'),
        
        # ═══════════════════════════════════════════════════════════════════
        # 4. Comparison Normalization
        # ═══════════════════════════════════════════════════════════════════
        
        # Pattern: not (X = Y)  →  X <> Y
        (r'not\s*\(\s*(.+?)\s*=\s*(.+?)\s*\)',
         r'\1 <> \2',
         'not_equals'),
        
        # Pattern: not (X <> Y)  →  X = Y
        (r'not\s*\(\s*(.+?)\s*<>\s*(.+?)\s*\)',
         r'\1 = \2',
         'not_not_equals'),
        
        # ═══════════════════════════════════════════════════════════════════
        # 5. Collection Property Normalization
        # ═══════════════════════════════════════════════════════════════════
        
        # Pattern: not X->notEmpty()  →  X->isEmpty()
        (r'not\s+(\w+(?:\.\w+)*)->notEmpty\(\)',
         r'\1->isEmpty()',
         'not_notEmpty'),
        
        # Pattern: not X->isEmpty()  →  X->notEmpty()
        (r'not\s+(\w+(?:\.\w+)*)->isEmpty\(\)',
         r'\1->notEmpty()',
         'not_isEmpty'),
        
        # Pattern: X->size() > 0  →  X->notEmpty()
        (r'(\w+(?:\.\w+)*)->size\(\)\s*>\s*0',
         r'\1->notEmpty()',
         'size_gt_zero'),
        
        # Pattern: X->size() >= 1  →  X->notEmpty()
        (r'(\w+(?:\.\w+)*)->size\(\)\s*>=\s*1',
         r'\1->notEmpty()',
         'size_gte_one'),
    ]
    
    def __init__(self, enable_logging: bool = False):
        """
        Initialize the OCL normalizer.
        
        Args:
            enable_logging: If True, logs each normalization transformation
        """
        self.enable_logging = enable_logging
        self.compiled_rules = [
            (re.compile(pattern, re.IGNORECASE), replacement, name)
            for pattern, replacement, name in self.NORMALIZATION_RULES
        ]
    
    def normalize(self, constraint_text: str) -> str:
        """
        Apply all normalization rules to the constraint text.
        
        Args:
            constraint_text: The original OCL constraint text
            
        Returns:
            The normalized OCL constraint text
        """
        original = constraint_text
        normalized = constraint_text
        transformations_applied = []
        
        # Apply each normalization rule
        for pattern, replacement, rule_name in self.compiled_rules:
            new_text = pattern.sub(replacement, normalized)
            if new_text != normalized:
                transformations_applied.append(rule_name)
                normalized = new_text
        
        # Log transformations if enabled
        if self.enable_logging and transformations_applied:
            print(f"[OCL Normalizer] Applied {len(transformations_applied)} transformations:")
            for rule_name in transformations_applied:
                print(f"  • {rule_name}")
            if original != normalized:
                print(f"  Original:   {original[:80]}...")
                print(f"  Normalized: {normalized[:80]}...")
        
        return normalized
    
    def normalize_with_context(self, constraint_text: str, 
                               context_class: str,
                               xmi_metadata: Optional[dict] = None) -> str:
        """
        Apply normalization with XMI context awareness.
        
        This method can use multiplicity information from XMI to apply
        more sophisticated normalization (e.g., recognizing that a 
        collection with multiplicity [1..*] is always non-empty).
        
        Args:
            constraint_text: The original OCL constraint text
            context_class: The context class name
            xmi_metadata: Optional XMI metadata dictionary
            
        Returns:
            The normalized OCL constraint text
        """
        # First apply standard normalization
        normalized = self.normalize(constraint_text)
        
        # TODO: Add XMI-aware transformations here
        # Example: If XMI says association has multiplicity [1..*],
        #          can replace "X->notEmpty()" with "true" (always holds)
        
        return normalized
    
    def get_applied_rules(self, constraint_text: str) -> list:
        """
        Get the list of normalization rules that would be applied.
        
        Args:
            constraint_text: The OCL constraint text to analyze
            
        Returns:
            List of rule names that match the constraint
        """
        applied_rules = []
        normalized = constraint_text
        
        for pattern, replacement, rule_name in self.compiled_rules:
            if pattern.search(normalized):
                applied_rules.append(rule_name)
                normalized = pattern.sub(replacement, normalized)
        
        return applied_rules


# Convenience function for quick normalization
def normalize_ocl(constraint_text: str, enable_logging: bool = False) -> str:
    """
    Convenience function to normalize OCL constraint text.
    
    Args:
        constraint_text: The OCL constraint to normalize
        enable_logging: If True, logs transformations applied
        
    Returns:
        Normalized OCL constraint text
    """
    normalizer = OCLNormalizer(enable_logging=enable_logging)
    return normalizer.normalize(constraint_text)


if __name__ == "__main__":
    # Test normalization on example patterns
    print("OCL Normalizer Test Suite")
    print("=" * 80)
    
    test_cases = [
        "self.rentals->isEmpty() or self.rentals->forAll(r | r.status = 'active')",
        "self.branch = null or self.branch.city = 'NYC'",
        "self.cars->size() = 0 or self.cars->forAll(c | c.available)",
        "not (self.startDate > self.endDate)",
        "self.vehicles->size() > 0",
        "not self.customers->isEmpty()",
        "self.pickupWindow->isEmpty() or (self.pickupWindow.start < self.pickupWindow.end and self.pickupBranch <> null)",
    ]
    
    normalizer = OCLNormalizer(enable_logging=True)
    
    for i, test in enumerate(test_cases, 1):
        print(f"\nTest {i}:")
        print(f"Original:   {test}")
        normalized = normalizer.normalize(test)
        print(f"Normalized: {normalized}")
        rules = normalizer.get_applied_rules(test)
        print(f"Rules:      {', '.join(rules)}")
        print("-" * 80)
